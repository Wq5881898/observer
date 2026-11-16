from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from observer.update_install import finish_upgrade, upgrade_arguments


class UpdateInstallTests(unittest.TestCase):
    def test_closed_database_backed_up_before_launch_and_flags_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); app = root/'new app'; app.mkdir()
            args = SimpleNamespace(data_dir=root/'history', demo=True, profile=root/'profile.json', no_profile=False)
            args.data_dir.mkdir()
            with closing(sqlite3.connect(args.data_dir/'observer.sqlite3')) as db:
                db.execute('CREATE TABLE proof(value TEXT)'); db.execute("INSERT INTO proof VALUES('keep')"); db.commit()
            def launch(command, **kwargs):
                backups = list((args.data_dir/'updates').glob('before-upgrade-*.sqlite3'))
                self.assertEqual(len(backups), 1)
                with closing(sqlite3.connect(backups[0])) as copy:
                    self.assertEqual(copy.execute('SELECT * FROM proof').fetchall(), [('keep',)])
                self.assertEqual(command, [str(app/'Observer.exe'), *upgrade_arguments(args)])
            with patch('observer.update_install.subprocess.Popen', side_effect=launch) as start, \
                    patch('observer.update_install.update_shortcut') as shortcut:
                result = finish_upgrade(app, args)
            self.assertEqual(result['status'], 'launched'); start.assert_called_once()
            shortcut.assert_called_once_with(app, upgrade_arguments(args))
            self.assertIn('--demo', upgrade_arguments(args)); self.assertIn('--profile', upgrade_arguments(args))

    def test_launch_failure_retains_original_and_attempts_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(data_dir=Path(tmp), demo=False, profile=None, no_profile=True)
            with patch('observer.update_install.subprocess.Popen', side_effect=[OSError('fail'), None]) as start, \
                    patch('observer.update_install.update_shortcut') as shortcut:
                result = finish_upgrade(Path(tmp)/'new', args)
            self.assertIn('failed', result['status']); self.assertEqual(start.call_count, 2)
            shortcut.assert_not_called()
            self.assertIn('--no-profile', start.call_args.args[0])
            self.assertTrue((Path(tmp)/'updates/restart-error.txt').is_file())

    def test_shortcut_failure_does_not_undo_successful_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(data_dir=Path(tmp), demo=False, profile=None)
            with patch('observer.update_install.subprocess.Popen') as start, \
                    patch('observer.update_install.update_shortcut', side_effect=OSError('denied')):
                result = finish_upgrade(Path(tmp)/'new', args)
            start.assert_called_once(); self.assertEqual(result['status'], 'launched')
            self.assertIn('unchanged', result['shortcut'])
