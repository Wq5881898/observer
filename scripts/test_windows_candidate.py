"""Outside-process checks for a supplied onedir candidate; camera-free."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--native-qt', action='store_true')
    parser.add_argument('--resources-only', action='store_true')
    args = parser.parse_args()
    if not args.run_id.replace('-', '').replace('_', '').isalnum():
        parser.error('simple run-id required')
    candidate = args.candidate.resolve()
    evidence = ROOT/'artifacts/phase-f'/args.run_id
    evidence.mkdir(parents=True, exist_ok=False)
    work = ROOT/'build/phase-f-tests'/args.run_id
    work.mkdir(parents=True, exist_ok=False)
    cwd = work/'任意 工作目录'
    cwd.mkdir()
    local_appdata = work/'隔离 用户数据'
    data = local_appdata/'Observer/product'
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'QT_PLUGIN_PATH',
                'QT_QPA_PLATFORM_PLUGIN_PATH', 'QML2_IMPORT_PATH'):
        env.pop(key, None)
    env.update(PATH=str(Path(os.environ['SystemRoot'])/'System32'),
               QT_QPA_PLATFORM='windows' if args.native_qt else 'offscreen',
               LOCALAPPDATA=str(local_appdata), TEMP=str(work/'temp'), TMP=str(work/'temp'))
    (work/'temp').mkdir()
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'candidate': str(candidate),
              'sanitized_path': env['PATH'], 'camera_opened': False, 'runs': []}

    def execute(exe, label, *flags, expected=0):
        output = evidence/f'{label}.json'
        command = [str(exe), *flags, '--data-dir', str(data), '--report', str(output)]
        with (evidence/f'{label}.stdout.log').open('wb') as stdout, \
                (evidence/f'{label}.stderr.log').open('wb') as stderr:
            result = subprocess.run(command, env=env, cwd=cwd, stdout=stdout, stderr=stderr,
                                    timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
        report['runs'].append({'label': label, 'command': command, 'exit_code': result.returncode})
        if result.returncode != expected or not output.is_file():
            raise RuntimeError(f'{label}: exit={result.returncode}, expected={expected}, report={output.exists()}')
        return json.loads(output.read_text(encoding='utf-8'))

    try:
        inventory = json.loads((candidate/'files.sha256.json').read_text(encoding='utf-8'))
        recorded = {item['path'] for item in inventory}
        actual = {p.relative_to(candidate).as_posix() for p in candidate.rglob('*') if p.is_file()}
        if actual != recorded | {'files.sha256.json'}:
            raise RuntimeError('Unrecorded or missing candidate files')
        suspicious = []
        for row in inventory:
            path = candidate/row['path']
            if digest(path) != row['sha256']:
                raise RuntimeError(f'File digest mismatch: {row["path"]}')
            relative = Path(row['path'])
            user_tree = relative.parts and relative.parts[0] in {'data', 'logs', 'evidence', 'tests', 'training', '.git', '.venv'}
            if (user_tree
                    or relative.name.startswith('.env')
                    or relative.suffix.lower() in {'.db', '.sqlite', '.sqlite3', '.mp4', '.log'}
                    or relative.name.startswith(('run_e3', 'test_e3', 'run_e5'))):
                suspicious.append(row['path'])
        report['audit'] = {'file_count': len(inventory), 'bytes': sum(r['bytes'] for r in inventory),
                           'suspicious_paths': suspicious}
        if suspicious:
            raise RuntimeError('Candidate privacy audit requires inspection')
        execute(candidate/'Observer-diagnostic.exe', 'original-native', '--check-resources', '--load-native')
        if args.resources_only:
            report['status'] = 'passed'
            return
        relocated = work/'中文 空格程序'/ 'Observer'
        shutil.copytree(candidate, relocated)
        first = execute(relocated/'Observer-diagnostic.exe', 'relocated-desktop', '--self-test', '--load-native')
        original_hashes = {p.relative_to(relocated).as_posix(): digest(p)
                           for p in relocated.rglob('*') if p.is_file()}
        second = execute(candidate/'Observer.exe', 'replacement-windowed', '--self-test')
        third = execute(relocated/'Observer.exe', 'rollback-windowed', '--self-test')
        for before, after in ((first, second), (second, third)):
            old, new = before['desktop'], after['desktop']
            if not set(old['session_ids']).issubset(new['session_ids']):
                raise RuntimeError('History lost when changing executable directory')
            if any(new['evidence_hashes'].get(k) != v for k, v in old['evidence_hashes'].items()):
                raise RuntimeError('Evidence changed when changing executable directory')
        report['same_build_replacement_and_return'] = 'passed (not cross-schema rollback)'
        after_hashes = {p.relative_to(relocated).as_posix(): digest(p)
                       for p in relocated.rglob('*') if p.is_file()}
        if original_hashes != after_hashes:
            raise RuntimeError('Application wrote into its executable directory')
        target = relocated/'_internal/models/yolo/observer-phone-s304-e2b.onnx'
        backup = target.with_suffix('.held')
        target.rename(backup)
        try:
            missing = execute(relocated/'Observer-diagnostic.exe', 'missing-model', '--check-resources', expected=1)
            if missing['status'] != 'failed':
                raise RuntimeError('Missing model was accepted')
        finally:
            backup.rename(target)
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (evidence/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
