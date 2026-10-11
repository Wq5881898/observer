"""Run the GUI bootloader with null stdio, as required for shell launches."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


class StartupInfo(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('lpReserved', wintypes.LPWSTR),
                ('lpDesktop', wintypes.LPWSTR), ('lpTitle', wintypes.LPWSTR),
                ('dwX', wintypes.DWORD), ('dwY', wintypes.DWORD),
                ('dwXSize', wintypes.DWORD), ('dwYSize', wintypes.DWORD),
                ('dwXCountChars', wintypes.DWORD), ('dwYCountChars', wintypes.DWORD),
                ('dwFillAttribute', wintypes.DWORD), ('dwFlags', wintypes.DWORD),
                ('wShowWindow', wintypes.WORD), ('cbReserved2', wintypes.WORD),
                ('lpReserved2', ctypes.POINTER(ctypes.c_byte)),
                ('hStdInput', wintypes.HANDLE), ('hStdOutput', wintypes.HANDLE), ('hStdError', wintypes.HANDLE)]


class ProcessInfo(ctypes.Structure):
    _fields_ = [('hProcess', wintypes.HANDLE), ('hThread', wintypes.HANDLE),
                ('dwProcessId', wintypes.DWORD), ('dwThreadId', wintypes.DWORD)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.replace('-', '').replace('_', '').isalnum():
        parser.error('simple run-id required')
    evidence = ROOT/'artifacts/phase-f'/args.run_id
    evidence.mkdir(exist_ok=False)
    local = ROOT/'build/phase-f-tests'/args.run_id/'local-appdata'
    data = local/'Observer/product'
    env = dict(os.environ, PATH=str(Path(os.environ['SystemRoot'])/'System32'),
               LOCALAPPDATA=str(local), QT_QPA_PLATFORM='offscreen')
    for name in ('PYTHONPATH', 'PYTHONHOME', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        env.pop(name, None)
    command = [str(args.candidate.resolve()/'Observer.exe'), '--self-test',
               '--data-dir', str(data), '--report', str(evidence/'desktop.json')]
    startup = StartupInfo()
    startup.cb = ctypes.sizeof(startup)
    startup.dwFlags = 0x100 | 0x1  # STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    process = ProcessInfo()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateProcessW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p,
        ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
        ctypes.POINTER(StartupInfo), ctypes.POINTER(ProcessInfo)]
    kernel.CreateProcessW.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    block = ctypes.create_unicode_buffer('\0'.join(f'{k}={v}' for k,v in sorted(env.items(), key=lambda p:p[0].upper()))+'\0\0')
    ok = kernel.CreateProcessW(None, ctypes.create_unicode_buffer(subprocess.list2cmdline(command)),
        None, None, False, 0x08000000 | 0x400, block, str(evidence),
        ctypes.byref(startup), ctypes.byref(process))  # CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        deadline = time.monotonic()+60
        while kernel.WaitForSingleObject(process.hProcess, 200) == 258:
            if time.monotonic() >= deadline:
                kernel.TerminateProcess(process.hProcess, 1)  # Only this owned child.
                raise RuntimeError('Null-stdio GUI diagnostic timed out')
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(process.hProcess, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        details = json.loads((evidence/'desktop.json').read_text(encoding='utf-8'))
        log = (data/'logs/launcher.log').read_text(encoding='utf-8')
        logged_result = '"status": "passed"' in log
        summary = {'time_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
                   'stdio_handles': 'NULL; no inherited handles', 'exit_code': code.value,
                   'desktop_status': details['status'], 'result_written_to_launcher_log': logged_result,
                   'administrator_token': bool(ctypes.windll.shell32.IsUserAnAdmin())}
        (evidence/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary))
        if code.value or details['status'] != 'passed' or not logged_result:
            raise RuntimeError('Windowed launcher failed without console streams')
    finally:
        kernel.CloseHandle(process.hThread)
        kernel.CloseHandle(process.hProcess)


if __name__ == '__main__':
    main()
