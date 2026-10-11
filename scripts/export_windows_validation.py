"""Export a verified internal onedir candidate and standalone PowerShell checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--export-id', required=True)
    args = parser.parse_args()
    if not args.export_id.replace('-', '').replace('_', '').isalnum():
        parser.error('simple export-id required')
    candidate = args.candidate.resolve()
    inventory_file = candidate / 'files.sha256.json'
    inventory = json.loads(inventory_file.read_text(encoding='utf-8'))
    recorded = {row['path'] for row in inventory}
    if len(recorded) != len(inventory):
        raise RuntimeError('Duplicate inventory entries')
    actual = {p.relative_to(candidate).as_posix() for p in candidate.rglob('*') if p.is_file()}
    if actual != recorded | {'files.sha256.json'}:
        raise RuntimeError('Unrecorded or missing candidate files')
    for row in inventory:
        file = (candidate / row['path']).resolve()
        if not file.is_relative_to(candidate) or file.stat().st_size != row['bytes'] or digest(file) != row['sha256']:
            raise RuntimeError(f'Candidate integrity mismatch: {row["path"]}')
    evidence = ROOT / 'artifacts/phase-f' / args.export_id
    archive = ROOT / 'dist' / f'{args.export_id}.zip'
    temporary = archive.with_suffix('.zip.partial')
    if archive.exists() or temporary.exists() or evidence.exists():
        raise RuntimeError('Fresh export-id required; previous evidence must not be overwritten')
    evidence.mkdir(parents=True)
    tools = {
        'validate_windows_portable.ps1': ROOT / 'scripts/validate_windows_portable.ps1',
        'PORTABLE-VALIDATION.txt': ROOT / 'packaging/PORTABLE-VALIDATION.txt',
    }
    started = datetime.now(timezone.utc).isoformat()
    # Keep the original candidate byte-for-byte, including its own manifest.
    # Do not add the validation runner to the frozen resource/input identity.
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in sorted(actual):
            bundle.write(candidate / name, 'Observer/' + name)
        for name, file in tools.items():
            bundle.write(file, name)
    with zipfile.ZipFile(temporary) as bundle:
        bad = bundle.testzip()
        if bad:
            raise RuntimeError(f'Archive CRC mismatch: {bad}')
        for row in inventory:
            with bundle.open('Observer/' + row['path']) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                    raise RuntimeError(f'Archived candidate digest mismatch: {row["path"]}')
        for name, file in tools.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != digest(file):
                raise RuntimeError(f'Validation tool changed while exporting: {name}')
        if hashlib.sha256(bundle.read('Observer/files.sha256.json')).hexdigest() != digest(inventory_file):
            raise RuntimeError('Archived manifest changed while exporting')
    temporary.rename(archive)
    receipt = {
        'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'exported and archive integrity verified; independent machine run pending',
        'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive),
        'candidate_build_id': json.loads((candidate / '_internal/build-info.json').read_text())['build_id'],
        'candidate_manifest_sha256': digest(inventory_file), 'candidate_files': len(inventory),
        'tools': {name: digest(file) for name, file in tools.items()},
        'archive_crc_verified': True, 'archived_candidate_hashes_verified': True,
        'camera_opened': False, 'redistribution_review': 'pending; internal candidate only',
    }
    (evidence / 'transfer-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
