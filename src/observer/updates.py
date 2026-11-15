"""GitHub portable updates: download, verify and stage alongside the running app."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import zipfile

from observer import __version__
from observer.deployment import is_frozen, resource_root

REPOSITORY = 'Wq5881898/observer'
RELEASES_URL = f'https://github.com/{REPOSITORY}/releases'
API = f'https://api.github.com/repos/{REPOSITORY}/releases'
MAX_ARCHIVE = 1024 * 1024 * 1024
MAX_EXPANDED = 2 * MAX_ARCHIVE


class UpdateError(RuntimeError):
    pass


def version_key(value):
    match = re.fullmatch(r'v?(\d+)\.(\d+)\.(\d+)(?:[-.]?(alpha|beta|preview|rc)[.-]?(\d+))?', value)
    if not match:
        raise UpdateError('版本号格式不支持')
    major, minor, patch, label, number = match.groups()
    return (int(major), int(minor), int(patch),
            {'alpha': 0, 'beta': 1, 'preview': 2, 'rc': 3, None: 4}[label], int(number or 0))


def current_version():
    if is_frozen():
        info = json.loads((resource_root() / 'build-info.json').read_text(encoding='utf-8'))
        return str(info['version']), str(info['build_id'])
    return __version__, '源码运行'


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        host = parsed.hostname or ''
        if parsed.scheme != 'https' or not (host in ('api.github.com', 'github.com')
                or host.endswith('.githubusercontent.com')):
            raise UpdateError('更新下载重定向到了不支持的地址')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and parsed.netloc != urlparse(req.full_url).netloc:
            redirected.remove_header('Authorization')
        return redirected


@dataclass(frozen=True)
class Release:
    tag: str
    notes: str
    page: str
    asset_id: int
    asset_name: str
    size: int
    digest: str
    checksum_id: int | None


def select_release(releases, current, include_preview=False):
    candidates = []
    for item in releases:
        if item.get('draft') or (item.get('prerelease') and not include_preview):
            continue
        try:
            key = version_key(item['tag_name'])
        except (UpdateError, KeyError, TypeError):
            continue
        if not include_preview and key[3] != 4:
            continue
        candidates.append((key, item))
    if not candidates:
        return None
    key, item = max(candidates, key=lambda pair: pair[0])
    if key <= version_key(current):
        return None
    tag = item['tag_name']
    assets = item.get('assets', [])
    expected = f'Observer-{tag.removeprefix("v")}-windows-x64.zip'
    found = [a for a in assets if a.get('name') == expected and a.get('state', 'uploaded') == 'uploaded']
    if len(found) != 1:
        raise UpdateError(f'{tag} 缺少兼容的 Windows x64 更新包，请查看发布页')
    asset = found[0]
    checksums = [a for a in assets if a.get('name') == 'SHA256SUMS.txt']
    return Release(tag, str(item.get('body') or ''),
                   f'{RELEASES_URL}/tag/{tag}', int(asset['id']), expected,
                   int(asset['size']), str(asset.get('digest') or ''),
                   int(checksums[0]['id']) if len(checksums) == 1 else None)


class GitHubUpdates:
    def __init__(self, token='', opener=None):
        self._token = token.strip()
        self._opener = opener or build_opener(SafeRedirect())

    def _open(self, url, binary=False):
        headers = {'User-Agent': 'Observer-Updater', 'Accept':
                   'application/octet-stream' if binary else 'application/vnd.github+json',
                   'X-GitHub-Api-Version': '2022-11-28'}
        if self._token:
            headers['Authorization'] = f'Bearer {self._token}'
        try:
            return self._opener.open(Request(url, headers=headers), timeout=20)
        except HTTPError as exc:
            if exc.code in (401, 403, 404):
                raise UpdateError('暂时无法获取版本，请稍后重试或打开 GitHub 发布页。') from None
            raise UpdateError(f'GitHub 请求失败（HTTP {exc.code}），请稍后重试') from None
        except (URLError, OSError):
            raise UpdateError('连接 GitHub 失败，请检查网络后重试') from None

    def check(self, current, include_preview=False, cancelled=lambda: False):
        # Scan published releases, including preview only when requested.
        items = []
        for page in range(1, 11):
            if cancelled():
                raise UpdateError('已取消更新检查')
            with self._open(f'{API}?per_page=100&page={page}') as response:
                raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise UpdateError('GitHub 发布列表过大')
            batch = json.loads(raw)
            if not isinstance(batch, list):
                raise UpdateError('GitHub 发布列表格式错误')
            items.extend(batch)
            if len(batch) < 100:
                return select_release(items, current, include_preview)
        raise UpdateError('发布列表超过检查范围，请查看 GitHub 发布页')

    def stage(self, release, destination, progress=lambda done, total: None,
              cancelled=lambda: False):
        if not 0 < release.size <= MAX_ARCHIVE:
            raise UpdateError('更新包大小不受支持')
        expected = release.digest.removeprefix('sha256:') if release.digest.startswith('sha256:') else ''
        if not re.fullmatch('[0-9a-fA-F]{64}', expected):
            if not release.checksum_id:
                raise UpdateError('发布未提供 SHA-256，无法验证更新包')
            with self._open(f'{API}/assets/{release.checksum_id}', True) as response:
                content = response.read(65537)
            if len(content) > 65536:
                raise UpdateError('校验文件过大')
            matches = re.findall(r'^([0-9a-fA-F]{64})\s+\*?' + re.escape(release.asset_name) + r'\s*$',
                                 content.decode('utf-8-sig'), re.M)
            if len(matches) != 1:
                raise UpdateError('发布校验文件中缺少对应更新包')
            expected = matches[0]
        folder = Path(destination) / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        archive = folder / 'update.zip.partial'
        digest = hashlib.sha256()
        total = 0
        try:
            with self._open(f'{API}/assets/{release.asset_id}', True) as response, archive.open('xb') as target:
                while True:
                    if cancelled():
                        raise UpdateError('已取消下载')
                    block = response.read(256 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > release.size:
                        raise UpdateError('下载大小与发布信息不符')
                    digest.update(block)
                    target.write(block)
                    progress(total, release.size)
            if total != release.size or digest.hexdigest().lower() != expected.lower():
                raise UpdateError('更新包校验失败，保留当前版本')
            app = extract_verified(archive, folder / 'unpacked', cancelled)
            info = json.loads((app / '_internal/build-info.json').read_text(encoding='utf-8'))
            if version_key(str(info['version'])) != version_key(release.tag):
                raise UpdateError('更新包内部版本与 Release 标签不符')
            archive.rename(folder / 'update.zip')
            (folder / 'receipt.json').write_text(json.dumps({
                'tag': release.tag, 'sha256': digest.hexdigest(), 'app': str(app)}, indent=2), encoding='utf-8')
            return app
        finally:
            # Only this operation's partial download, never installed app/user data.
            archive.unlink(missing_ok=True)


def extract_verified(archive, destination, cancelled=lambda: False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 20000 or sum(e.file_size for e in entries) > MAX_EXPANDED:
            raise UpdateError('更新包解压大小超出限制')
        seen = set()
        for entry in entries:
            name = entry.filename
            parts = name.rstrip('/').split('/')
            if (not parts or name.startswith('/') or '\\' in name or ':' in name
                    or any(not p or p in ('.', '..') or p.endswith(('.', ' ')) for p in parts)
                    or any(p.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                        *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))} for p in parts)
                    or name.casefold() in seen or stat.S_ISLNK(entry.external_attr >> 16)):
                raise UpdateError('更新包包含不安全或重复路径')
            seen.add(name.casefold())
        for entry in entries:
            if cancelled():
                raise UpdateError('已取消解压')
            bundle.extract(entry, destination)
    app = destination / 'Observer'
    if not (app / 'Observer.exe').is_file():
        raise UpdateError('更新包缺少 Observer.exe')
    manifest = json.loads((app / 'files.sha256.json').read_text(encoding='utf-8'))
    recorded = set()
    for row in manifest:
        name = row['path']
        file = (app / name).resolve()
        if not file.is_relative_to(app.resolve()) or name in recorded or not file.is_file():
            raise UpdateError('程序文件清单无效')
        recorded.add(name)
        if file.stat().st_size != row['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            raise UpdateError('程序文件完整性检查失败')
    actual = {p.relative_to(app).as_posix() for p in app.rglob('*') if p.is_file()}
    if actual != recorded | {'files.sha256.json'}:
        raise UpdateError('更新包含未登记的程序文件')
    return app
