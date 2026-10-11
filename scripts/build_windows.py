"""Build fresh, auditable onedir candidates using the prepared build environment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.build_id):
        parser.error('build-id must be a simple directory name')
    if sys.platform != 'win32' or sys.version_info[:2] != (3, 11) or platform.machine() != 'AMD64':
        raise RuntimeError('Windows x64 Python 3.11 required')
    for name, version in re.findall(r'^([\w-]+)==([^\s\\]+)',
                                   (ROOT/'packaging/requirements.lock').read_text(), re.M):
        if metadata.version(name) != version:
            raise RuntimeError(f'Build tool mismatch: {name} != {version}')
    from observer.product_cli import verify_frozen_models, load_frozen_config
    from observer.deployment import source_identity
    verify_frozen_models(ROOT/'models')
    load_frozen_config()
    stage = ROOT/'build/phase-f'/args.build_id
    evidence = ROOT/'artifacts/phase-f'/args.build_id
    dist = ROOT/'dist'/args.build_id
    for path in (stage, evidence, dist):
        path.mkdir(parents=True, exist_ok=False)
    licenses = stage/'licenses'
    licenses.mkdir()
    packages = []
    for distribution in sorted(metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
        name = distribution.metadata['Name']
        row = {'name': name, 'version': distribution.version,
               'license': distribution.metadata.get('License-Expression') or distribution.metadata.get('License'),
               'review': 'pending; inventory is not redistribution approval', 'license_files': []}
        for file in distribution.files or []:
            if file.name.lower().startswith(('license', 'copying', 'notice')):
                source = Path(distribution.locate_file(file))
                if source.is_file():
                    target = licenses/name/str(file).replace('../', '__/')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    row['license_files'].append(target.relative_to(licenses).as_posix())
        packages.append(row)
    write_json(licenses/'inventory.json', packages)
    source_files = {p.relative_to(ROOT).as_posix(): sha(p)
                    for p in sorted((ROOT/'src/observer').glob('*.py'))}
    inputs = [ROOT/'pyproject.toml', ROOT/'uv.lock', ROOT/'src/observer/default_config.json',
              ROOT/'scripts/build_windows.py', ROOT/'scripts/build_windows.ps1',
              ROOT/'scripts/test_windows_candidate.py', ROOT/'scripts/test_candidate_readonly.ps1']
    inputs += sorted(p for p in (ROOT/'packaging').rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    inputs += [ROOT/'models'/n for n in ('manifest.json', 'face_landmarker.task',
              'pose_landmarker_lite.task', 'efficientdet_lite0_int8.tflite',
              'yolo/observer-phone-s304-e2b.onnx')]
    info = {'build_id': args.build_id, 'version': metadata.version('observer-focus'),
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'platform': platform.platform(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_worktree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
            'source_tree_sha256': source_identity()['source_tree_sha256'],
            'source_files': source_files,
            'inputs': {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
            'packages': [{k: v for k, v in p.items() if k in ('name', 'version')} for p in packages],
            'status': 'internal candidate; E gates and redistribution review pending'}
    write_json(stage/'build-info.json', info)
    write_json(evidence/'build-info.json', info)
    command = [sys.executable, '-m', 'PyInstaller', '--clean', '--noconfirm',
               '--distpath', str(dist), '--workpath', str(stage/'work'), str(ROOT/'packaging/Observer.spec')]
    write_json(evidence/'command.json', command)
    env = dict(os.environ, OBSERVER_BUILD_STAGE=str(stage),
               PYINSTALLER_CONFIG_DIR=str(stage/'pyinstaller-cache'))
    # Dependency scanning must not pick up DLLs from unrelated desktop tooling.
    env['PATH'] = os.pathsep.join([str(Path(sys.executable).parent),
                                 str(Path(os.environ['SystemRoot'])/'System32'),
                                 os.environ['SystemRoot']])
    for key in ('PYTHONPATH', 'PYTHONHOME', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH',
                'QML2_IMPORT_PATH'):
        env.pop(key, None)
    write_json(evidence/'build-environment.json', {'PATH': env['PATH'], 'python': sys.executable})
    with (evidence/'build.log').open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    write_json(evidence/'build-result.json', {'exit_code': result.returncode})
    if result.returncode:
        raise RuntimeError(f'PyInstaller failed: {evidence / "build.log"}')
    if any(sha(ROOT/name) != value for name, value in {**source_files, **info['inputs']}.items()):
        raise RuntimeError('Build inputs changed during construction; candidate provenance invalid')
    candidate = dist/'Observer'
    shutil.copyfile(ROOT/'packaging/START-HERE.txt', candidate/'START-HERE.txt')
    shutil.copyfile(ROOT/'packaging/REDISTRIBUTION.md', candidate/'REDISTRIBUTION.md')
    files = [{'path': p.relative_to(candidate).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(candidate.rglob('*')) if p.is_file()]
    # Manifest excludes itself; duplicate outside candidate preserves the build evidence.
    write_json(candidate/'files.sha256.json', files)
    write_json(evidence/'files.sha256.json', files)
    print(json.dumps({'candidate': str(candidate), 'files': len(files),
                      'bytes': sum(f['bytes'] for f in files)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
