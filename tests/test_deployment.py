import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from observer import deployment
from observer.frozen_entry import entry_arguments, log_directory
from observer.product_cli import apply_default_profile
from types import SimpleNamespace


class DeploymentTests(unittest.TestCase):
    def test_source_defaults_remain_relative(self):
        self.assertEqual(deployment.default_data_dir(), Path('data/product'))
        self.assertEqual(deployment.default_models_dir(), Path('models'))
        self.assertEqual(deployment.default_profile(), Path('data/calibration/default.json'))

    def test_frozen_paths_ignore_working_directory(self):
        with tempfile.TemporaryDirectory() as root, \
                patch.object(deployment.sys, 'frozen', True, create=True), \
                patch.object(deployment, '__file__', str(Path(root)/'包 _internal/observer/deployment.py')), \
                patch.dict(os.environ, {'LOCALAPPDATA': root}):
            self.assertEqual(deployment.default_models_dir(), Path(root)/'包 _internal/models')
            self.assertEqual(deployment.default_data_dir(), Path(root)/'Observer/product')
            self.assertEqual(deployment.default_profile(), Path(root)/'Observer/calibration/default.json')

    def test_frozen_identity_requires_build_record(self):
        with tempfile.TemporaryDirectory() as root, \
                patch.object(deployment.sys, 'frozen', True, create=True), \
                patch.object(deployment, 'resource_root', return_value=Path(root)):
            with self.assertRaises(FileNotFoundError):
                deployment.source_identity()
            path = Path(root)/'build-info.json'
            path.write_text(json.dumps({'source_tree_sha256': 'a'*64, 'build_id': 'test',
                                        'source_files': {'src/observer/a.py': 'b'*64}}))
            identity = deployment.source_identity()
            self.assertEqual(identity['source_identity_origin'], 'build-input-manifest')
            self.assertEqual(identity['build_info_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_text(json.dumps({'source_tree_sha256': 'a'*64, 'source_files': {}}))
            with self.assertRaises(ValueError):
                deployment.source_identity()

    def test_source_digest_is_not_empty(self):
        self.assertNotEqual(deployment.source_identity()['source_tree_sha256'], hashlib.sha256().hexdigest())

    def test_missing_or_relative_local_appdata_rejected(self):
        for value in ('', 'relative'):
            with patch.dict(os.environ, {'LOCALAPPDATA': value}):
                with self.assertRaises(RuntimeError):
                    deployment.user_root()

    def test_default_profile_respects_override_and_opt_out(self):
        with tempfile.TemporaryDirectory() as root:
            profile = Path(root)/'calibration.json'
            profile.write_text('{}')
            args = SimpleNamespace(demo=False, profile=None, no_profile=False)
            with patch('observer.product_cli.default_profile', return_value=profile):
                self.assertEqual(apply_default_profile(args), profile)
                args.profile = Path('explicit.json')
                self.assertEqual(apply_default_profile(args), Path('explicit.json'))
                args.profile, args.no_profile = None, True
                self.assertIsNone(apply_default_profile(args))

    def test_double_click_and_explicit_cli(self):
        self.assertEqual(entry_arguments([]), ['product', 'gui'])
        args = ['product', '--data-dir', '中文 路径', 'gui', '--demo']
        self.assertEqual(entry_arguments(args), args)
        self.assertEqual(log_directory(args), Path('中文 路径').resolve()/'logs')
