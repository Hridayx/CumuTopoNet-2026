"""Small filesystem and configuration primitives shared by CLI stages."""
from __future__ import annotations
import contextlib
import json
import os
from pathlib import Path
import tempfile
import time
import yaml


class IntegrityError(RuntimeError):
    """Scientific data/configuration cannot safely be used."""


@contextlib.contextmanager
def atomic_target(path, mode='wb'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, mode) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, obj):
    with atomic_target(path, 'w') as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_once(path, obj):
    """Caller must hold its stage lock if concurrent writers are possible."""
    path = Path(path)
    if path.exists():
        if read_json(path) != obj:
            raise IntegrityError(f'Immutable artifact differs: {path}')
    else:
        write_json(path, obj)


@contextlib.contextmanager
def directory_lock(path):
    import socket
    import psutil
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as e:
        raise IntegrityError(f'Lock exists: {path}. Check owner/process before recovering.') from e
    write_json(path/'owner.json', {'host': socket.gethostname(), 'pid': os.getpid(),
               'process_created': psutil.Process().create_time(), 'created': time.time()})
    try:
        yield
    finally:
        (path/'owner.json').unlink()
        path.rmdir()


def merge(base, overrides):
    result = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge(result[k], v)
        else:
            result[k] = v
    return result


DEFAULTS = {
    'dataset': {'path': '', 'session_metadata': None, 'windows_per_recording': 1024,
                'parent_length': 1024, 'sample_seed': 42, 'sample_rate': 60000000,
                'class_aliases': {}, 'mode_aliases': {}, 'condition_aliases': {}},
    'workspace': './work/recovery',
    'features': {'length': 128, 'tda_points': 64, 'tda_dim': 3, 'tda_delay': 5,
                 'pi_size': 20, 'pi_sigma': 0.1, 'tda_channels': [0, 1]},
    'training': {'batch_size': 256, 'max_epochs': 30, 'min_epochs': 10,
                 'patience': 6, 'weight_decay': 0.0001, 'lr': 0.001,
                 'seeds': [17, 42, 91], 'phase_augmentation': True,
                 'temperature': 0.07, 'checkpoint_steps': 100, 'amp': True},
    'budget': {'hours': 72, 'reserve_hours': 12, 'core_target_hours': 48,
               'safety_factor': 1.5, 'gpu_workers': 1},
    'runtime': {'cpus': 8, 'archive_readers': 2, 'feature_workers': 6,
                'heartbeat_seconds': 60, 'monitor_seconds': 60},
}


def load_config(path):
    path = Path(path).resolve()
    with path.open() as f:
        supplied = yaml.safe_load(f) or {}
    cfg = merge(DEFAULTS, supplied)
    for field in [('workspace',), ('dataset', 'path'), ('dataset', 'session_metadata')]:
        parent = cfg if len(field) == 1 else cfg[field[0]]
        key = field[-1]
        if parent.get(key):
            p = Path(os.path.expandvars(str(parent[key]))).expanduser()
            candidate = path.parent / p if not p.is_absolute() else p
            parent[key] = str(candidate.resolve())
    cfg['_config_path'] = str(path)
    if cfg['dataset']['parent_length'] != 1024:
        raise ValueError('The matched-location protocol requires 1024-sample parent windows.')
    if cfg['dataset']['windows_per_recording'] not in (512, 1024) and not cfg.get('fixture', False):
        raise ValueError('Production supports 512 or 1024 windows/recording; use fixture: true for tests.')
    if cfg['runtime']['archive_readers'] + cfg['runtime']['feature_workers'] > cfg['runtime']['cpus']:
        raise ValueError('Archive readers + feature workers exceed available CPUs.')
    if cfg['budget']['hours'] <= cfg['budget']['reserve_hours']:
        raise ValueError('Budget must exceed evaluation reserve.')
    return cfg


def workspace(cfg):
    p = Path(cfg['workspace'])
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_event(root, kind, details):
    """One immutable event per file; no concurrent appends on network filesystems."""
    import uuid
    write_json(Path(root) / 'events' / f'{time.time_ns()}-{uuid.uuid4().hex}.json',
               {'time': time.time(), 'kind': kind, 'details': details})
