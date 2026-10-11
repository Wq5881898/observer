"""Inject resource faults into a marked test copy and restore every byte."""
from datetime import datetime, timezone
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('copy', type=Path)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.replace('-', '').replace('_', '').isalnum():
        parser.error('simple run-id required')
    copy = args.copy.resolve()
    if not copy.is_relative_to(ROOT/'build/phase-f-tests'):
        parser.error('Only isolated candidate copies under build/phase-f-tests are allowed')
    evidence = ROOT/'artifacts/phase-f'/args.run_id
    evidence.mkdir(exist_ok=False)
    data = ROOT/'build/phase-f-tests'/args.run_id/'isolated-data'
    env = dict(os.environ, PATH=str(Path(os.environ['SystemRoot'])/'System32'), QT_QPA_PLATFORM='offscreen')
    for key in ('PYTHONPATH', 'PYTHONHOME', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        env.pop(key, None)
    results = []
    targets = {'configuration': copy/'_internal/observer/default_config.json',
               'face-model': copy/'_internal/models/face_landmarker.task'}
    before = {key: digest(path) for key, path in targets.items()}
    for label, target in targets.items():
        raw = target.read_bytes()
        try:
            # Change one byte in a copy only; original C resources never touched.
            target.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            report = evidence/f'{label}.json'
            command = [str(copy/'Observer-diagnostic.exe'), '--check-resources', '--load-native',
                       '--data-dir', str(data), '--report', str(report)]
            result = subprocess.run(command, cwd=evidence, env=env, capture_output=True,
                                    timeout=45, creationflags=subprocess.CREATE_NO_WINDOW)
            (evidence/f'{label}.stderr.log').write_bytes(result.stderr)
            detail = json.loads(report.read_text(encoding='utf-8'))
            results.append({'label': label, 'command': command, 'exit_code': result.returncode,
                            'status': detail['status'], 'error': detail.get('error')})
            if result.returncode != 1 or detail['status'] != 'failed':
                raise RuntimeError(f'{label} accepted changed resource')
        finally:
            target.write_bytes(raw)
    # Test a native module load error with the diagnostic entry, so errors remain
    # observable without an unattended GUI message box.
    native = copy/'_internal/PySide6/QtWidgets.pyd'
    held = native.with_suffix('.held')
    native.rename(held)
    try:
        command = [str(copy/'Observer-diagnostic.exe'), 'product', '--data-dir', str(data), 'gui', '--demo']
        result = subprocess.run(command, cwd=evidence, env=env, capture_output=True,
                                timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        (evidence/'missing-native.stdout.log').write_bytes(result.stdout)
        (evidence/'missing-native.stderr.log').write_bytes(result.stderr)
        if result.returncode != 1 or not result.stdout+result.stderr:
            raise RuntimeError('Missing Qt module failed without a diagnostic')
        results.append({'label': 'missing-native', 'command': command, 'exit_code': result.returncode})
    finally:
        held.rename(native)
    # No failed validation should have reached the store/session owner.
    databases = [str(p) for p in data.rglob('*') if p.suffix in ('.db', '.sqlite', '.sqlite3')]
    restored = all(digest(path) == before[label] for label, path in targets.items())
    summary = {'time_utc': datetime.now(timezone.utc).isoformat(), 'results': results,
               'resources_restored': restored, 'created_databases': databases,
               'launcher_log_exists': (data/'logs/launcher.log').is_file(),
               'administrator_token': bool(ctypes.windll.shell32.IsUserAnAdmin()),
               'status': 'passed' if restored and not databases else 'failed'}
    (evidence/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    if summary['status'] != 'passed':
        raise RuntimeError('Resource fault cleanup failed')


if __name__ == '__main__':
    main()
