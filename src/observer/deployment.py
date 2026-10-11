"""Frozen deployment paths and provenance; source CLI defaults stay compatible."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


def is_frozen() -> bool:
    return bool(getattr(sys, 'frozen', False))


def resource_root() -> Path:
    return Path(__file__).resolve().parent.parent if is_frozen() else Path('.')


def user_root() -> Path:
    local = os.environ.get('LOCALAPPDATA')
    if not local or not Path(local).is_absolute():
        raise RuntimeError('LOCALAPPDATA 必须指向用户可写的绝对路径')
    return Path(local) / 'Observer'


def default_data_dir() -> Path:
    return user_root() / 'product' if is_frozen() else Path('data/product')


def default_profile() -> Path:
    return user_root() / 'calibration/default.json' if is_frozen() else Path('data/calibration/default.json')


def default_models_dir() -> Path:
    return resource_root() / 'models'


def source_identity() -> dict:
    if is_frozen():
        raw = (resource_root() / 'build-info.json').read_bytes()
        info = json.loads(raw)
        digest = info['source_tree_sha256']
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or not info.get('source_files')):
            raise ValueError('构建来源记录缺少有效源码摘要')
        return {'source_tree_sha256': digest,
                'source_identity_origin': 'build-input-manifest',
                'build_info_sha256': hashlib.sha256(raw).hexdigest(),
                'build_id': info['build_id']}
    digest = hashlib.sha256()
    files = sorted(Path(__file__).parent.glob('*.py'))
    if not files:
        raise RuntimeError('源码身份不可读取')
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {'source_tree_sha256': digest.hexdigest(),
            'source_identity_origin': 'source-files'}
