"""Small packaged entry, without development/calibration CLI commands."""
from __future__ import annotations

import argparse
import ctypes
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

from observer.deployment import default_data_dir


def entry_arguments(argv):
    return list(argv) if argv else ['product', 'gui']


def log_directory(argv):
    # Honor test/user isolation even before argparse or Qt has loaded.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data-dir', type=Path)
    args, _ = parser.parse_known_args(argv)
    return (args.data_dir or default_data_dir()).resolve() / 'logs'


class LogStream:
    def __init__(self, logger, level):
        self.logger, self.level = logger, level

    def write(self, text):
        if text.strip():
            self.logger.log(self.level, text.rstrip())
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    windowed = sys.stderr is None
    logger = logging.getLogger('observer.launcher')
    log_path = None
    try:
        log_path = log_directory(argv) / 'launcher.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log_path, maxBytes=512*1024, backupCount=1,
                                      encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Do not log argv: it may contain a personal task title or profile path.
        logger.info('Launcher started')
        if sys.stdout is None:
            sys.stdout = LogStream(logger, logging.INFO)
        if sys.stderr is None:
            sys.stderr = LogStream(logger, logging.ERROR)
        if argv and argv[0] in ('--check-resources', '--self-test'):
            from frozen_diagnostics import run
            return run(argv)
        from observer.product_cli import register_parser, dispatch
        parser = argparse.ArgumentParser(prog='Observer')
        register_parser(parser.add_subparsers(dest='command', required=True))
        args = parser.parse_args(entry_arguments(argv))
        code = dispatch(args)
        if code:
            raise RuntimeError(f'程序返回错误码 {code}；详细原因见日志或诊断控制台')
        return 0
    except SystemExit as exc:
        if not exc.code:
            return 0
        logger.error('Invalid command line: %s', exc.code)
        if windowed:
            ctypes.windll.user32.MessageBoxW(None, f'启动参数无效。日志：{log_path}', 'Observer', 0x10)
        return int(exc.code)
    except Exception as exc:
        logger.exception('Launcher failed')
        if windowed:
            ctypes.windll.user32.MessageBoxW(
                None, f'Observer 启动失败：{exc}\n日志：{log_path or "无法创建日志"}', 'Observer', 0x10)
        elif sys.stderr is not None:
            print(f'Observer: {exc}; log: {log_path}', file=sys.stderr)
        return 1
