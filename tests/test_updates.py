import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from urllib.request import Request
import zipfile

from observer.updates import (
    API, GitHubUpdates, Release, SafeRedirect, UpdateError, extract_verified,
    select_release, version_key,
)


def release(tag='v0.1.0-preview.4', preview=True):
    return dict(tag_name=tag, prerelease=preview, draft=False, body='Notes', assets=[
        dict(id=1, name=f'Observer-{tag[1:]}-windows-x64.zip', size=100, digest='')])


def bundle(extra=None):
    files = {'Observer.exe': b'exe', '_internal/build-info.json': b'{"version":"0.1.0-preview.4"}'}
    inventory = [dict(path=p, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for p, b in files.items()]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for p, b in files.items():
            archive.writestr('Observer/' + p, b)
        archive.writestr('Observer/files.sha256.json', json.dumps(inventory))
        if extra:
            archive.writestr(*extra)
    return stream.getvalue()


class FakeOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        return io.BytesIO(self.responses.pop(0))


class UpdateTests(unittest.TestCase):
    def test_semantic_order_not_lexical_and_no_downgrade(self):
        self.assertGreater(version_key('v0.1.0-preview.10'), version_key('0.1.0-preview.9'))
        self.assertGreater(version_key('0.1.0'), version_key('0.1.0rc3'))
        self.assertIsNone(select_release([release('v0.1.0-preview.2')], '0.1.0-preview.3', True))

    def test_explicit_preview_opt_in_and_drafts_excluded(self):
        draft = dict(release('v9.0.0', False), draft=True)
        items = [release(), release('v0.1.0', False), draft]
        self.assertEqual(select_release(items, '0.0.9').tag, 'v0.1.0')
        self.assertEqual(select_release([release()], '0.1.0-preview.3', True).tag, 'v0.1.0-preview.4')
        self.assertIsNone(select_release([release()], '0.0.1'))

    def test_missing_compatible_asset_is_actionable(self):
        item = release()
        item['assets'] = []
        with self.assertRaises(UpdateError):
            select_release([item], '0.1.0-preview.3', True)

    def test_authorization_not_forwarded_to_asset_cdn(self):
        request = Request(API, headers={'Authorization': 'Bearer secret'})
        redirected = SafeRedirect().redirect_request(request, None, 302, '', {},
                        'https://release-assets.githubusercontent.com/file')
        self.assertIsNone(redirected.get_header('Authorization'))
        with self.assertRaises(UpdateError):
            SafeRedirect().redirect_request(request, None, 302, '', {}, 'http://evil.test/file')

    def test_download_checksum_fallback_and_stage(self):
        data = bundle()
        checksum = hashlib.sha256(data).hexdigest()
        name = 'Observer-0.1.0-preview.4-windows-x64.zip'
        opener = FakeOpener([f'{checksum}  {name}\n'.encode(), data])
        client = GitHubUpdates('secret', opener)
        item = Release('v0.1.0-preview.4', '', '', 1, name, len(data), '', 2)
        with tempfile.TemporaryDirectory() as tmp:
            app = client.stage(item, Path(tmp))
            self.assertEqual((app / 'Observer.exe').read_bytes(), b'exe')
            self.assertTrue((app.parent.parent / 'receipt.json').exists())
        self.assertTrue(opener.requests[0].full_url.startswith(API + '/assets/'))

    def test_corrupt_download_never_produces_launchable_app(self):
        data = bundle()
        client = GitHubUpdates(opener=FakeOpener([data]))
        item = Release('v0.1.0-preview.4', '', '', 1, 'x.zip', len(data), 'sha256:' + '0' * 64, None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UpdateError):
                client.stage(item, Path(tmp))
            self.assertFalse(list(Path(tmp).rglob('Observer.exe')))
            self.assertFalse(list(Path(tmp).rglob('*.partial')))

    def test_unsafe_zip_paths_and_unrecorded_executable_rejected(self):
        for path in ('../outside.exe', 'Observer/../bad.exe', 'Observer/C:bad.exe',
                     'Observer/NUL.txt', 'Observer/./bad', 'Observer/extra.exe'):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / 'a.zip'
                archive.write_bytes(bundle((path, b'bad')))
                with self.assertRaises(UpdateError):
                    extract_verified(archive, Path(tmp) / 'unpacked')
                self.assertFalse((Path(tmp) / 'outside.exe').exists())

    def test_release_and_internal_version_must_match(self):
        data = bundle()
        item = Release('v0.2.0', '', '', 1, 'x.zip', len(data),
                       'sha256:' + hashlib.sha256(data).hexdigest(), None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(UpdateError, '版本'):
                GitHubUpdates(opener=FakeOpener([data])).stage(item, Path(tmp))

    def test_cancelled_download_not_staged(self):
        data = bundle()
        item = Release('v0.1.0-preview.4', '', '', 1, 'x.zip', len(data), 'sha256:' + '0' * 64, None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(UpdateError, '取消'):
                GitHubUpdates(opener=FakeOpener([data])).stage(item, Path(tmp), cancelled=lambda: True)

    def test_paginated_release_check(self):
        opener = FakeOpener([json.dumps([release()]).encode()])
        result = GitHubUpdates(opener=opener).check('0.1.0-preview.3', True)
        self.assertEqual(result.tag, 'v0.1.0-preview.4')
