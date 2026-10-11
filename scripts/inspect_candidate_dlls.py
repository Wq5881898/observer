"""Read-only PE import/export comparison for a candidate's Qt dependencies."""
import argparse
import json
from pathlib import Path
import pefile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.candidate.resolve()/'_internal'
    files = {}
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in ('.dll', '.pyd'):
            files.setdefault(path.name.lower(), []).append(path)
    report = []
    for path in (root/'PySide6').glob('*'):
        if path.suffix.lower() not in ('.dll', '.pyd'):
            continue
        pe = pefile.PE(str(path), fast_load=True, max_symbol_exports=65536)
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
        for entry in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', []):
            name = entry.dll.decode().lower()
            for dependency in files.get(name, []):
                other = pefile.PE(str(dependency), fast_load=True, max_symbol_exports=65536)
                other.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']])
                exports = getattr(other, 'DIRECTORY_ENTRY_EXPORT', None)
                names = {symbol.name for symbol in exports.symbols} if exports else set()
                missing = [symbol.name.decode(errors='replace') for symbol in entry.imports
                           if symbol.name and symbol.name not in names]
                if missing:
                    report.append({'importer': str(path.relative_to(root)),
                                   'dependency': str(dependency.relative_to(root)), 'missing_exports': missing})
                other.close()
        pe.close()
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
