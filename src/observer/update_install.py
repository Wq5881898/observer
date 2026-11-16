"""Complete an already verified update after the old session owner exits."""
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys


def upgrade_arguments(args):
    values = ['product', '--data-dir', str(Path(args.data_dir).resolve()), 'gui']
    if args.demo:
        values.append('--demo')
    if args.profile:
        values.extend(['--profile', str(Path(args.profile).resolve())])
    if getattr(args, 'no_profile', False):
        values.append('--no-profile')
    return values


def update_shortcut(app, arguments):
    if os.name != 'nt':
        return
    # Environment values are data, never interpolated into PowerShell code.
    script = """
$ErrorActionPreference = 'Stop'
$path = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Observer.lnk'
if (-not (Test-Path -LiteralPath $path)) { exit 0 }
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($path)
$match = [regex]::Match($link.Arguments, '(?:^|\s)--data-dir\s+(?:"([^"]+)"|(\S+))')
$sameData = $false
if ($match.Success) {
    $value = if ($match.Groups[1].Success) { $match.Groups[1].Value } else { $match.Groups[2].Value }
    $sameData = [IO.Path]::IsPathRooted($value) -and [IO.Path]::GetFullPath($value) -eq $env:OBSERVER_UPDATE_DATA
}
if ($link.TargetPath -ne $env:OBSERVER_PREVIOUS_APP -and -not ($sameData -and [IO.Path]::GetFileName($link.TargetPath) -eq 'Observer.exe')) { exit 0 }
$link.TargetPath = $env:OBSERVER_UPDATE_APP
$link.Arguments = $env:OBSERVER_UPDATE_ARGS
$link.WorkingDirectory = $env:OBSERVER_UPDATE_CWD
$link.Save()
"""
    env = dict(os.environ, OBSERVER_PREVIOUS_APP=sys.executable,
               OBSERVER_UPDATE_APP=str(app/'Observer.exe'),
               OBSERVER_UPDATE_CWD=str(app),
               OBSERVER_UPDATE_DATA=arguments[2],
               OBSERVER_UPDATE_ARGS=subprocess.list2cmdline(arguments))
    powershell = Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    subprocess.run([str(powershell), '-NoProfile', '-NonInteractive', '-Command', script],
                   env=env, check=True, timeout=10, capture_output=True,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


def finish_upgrade(app, args):
    app = Path(app).resolve()
    data = Path(args.data_dir).resolve()
    folder = data/'updates'
    folder.mkdir(parents=True, exist_ok=True)
    command = [str(app/'Observer.exe'), *upgrade_arguments(args)]
    receipt = {'app': str(app), 'status': 'starting'}
    try:
        database = data/'observer.sqlite3'
        if database.exists():
            backup = folder/('before-upgrade-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.sqlite3')
            with closing(sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)) as source:
                with closing(sqlite3.connect(backup)) as target:
                    source.backup(target)
            receipt['backup'] = str(backup)
        subprocess.Popen(command, cwd=app, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        receipt['status'] = 'launched'
    except Exception:
        receipt['status'] = 'failed; original executable retained'
        (folder/'restart-error.txt').write_text('更新启动失败，已保留原程序。请从原入口重试。', encoding='utf-8')
        # Restore the original program with the same history/profile flags.
        try:
            subprocess.Popen([sys.executable, *upgrade_arguments(args)],
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception:
            pass
    else:
        try:
            update_shortcut(app, command[1:])
        except Exception:
            receipt['shortcut'] = 'unchanged; could not update'
        else:
            receipt['shortcut'] = 'checked'
    (folder/'last-upgrade.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt
