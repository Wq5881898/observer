import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('observer_build', Path(__file__).resolve().parents[1]/'scripts/build_windows.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class PackagingNoticeTests(unittest.TestCase):
    def test_vendor_third_party_notice_bytes_and_duplicate_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'LICENSE').write_text('project terms')
            (root/'NOTICE').write_text('project notice')
            vendor = root/'vendor'
            vendor.mkdir()
            raw = b'Vendor third-party attribution\r\n\xff'
            (vendor/'ThirdPartyNotices.txt').write_bytes(raw)
            (vendor/'module.py').write_text('not a notice')
            dist = SimpleNamespace(metadata={'Name': 'onnxruntime'}, version='1.29.0',
                files=[Path('ThirdPartyNotices.txt'), Path('module.py')],
                locate_file=lambda p: vendor/p)
            output = root/'out'
            rows = build.collect_licenses([dist, dist], output, root)
            self.assertEqual(len(rows), 1)
            copied = output/'onnxruntime/ThirdPartyNotices.txt'
            self.assertEqual(copied.read_bytes(), raw)
            self.assertEqual(rows[0]['license_file_sha256']['onnxruntime/ThirdPartyNotices.txt'], hashlib.sha256(raw).hexdigest())
            self.assertFalse((output/'onnxruntime/module.py').exists())
            self.assertEqual((output/'Observer/LICENSE').read_text(), 'project terms')

    def test_missing_project_terms_fail_instead_of_silent_incomplete_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                build.collect_licenses([], root/'out', root)

    def test_existing_evidence_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                build.collect_licenses([], root, root)
