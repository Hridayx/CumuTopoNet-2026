"""Local environment, dataset, and CUDA preflight checks."""
from __future__ import annotations
import importlib.metadata
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import uuid
import torch
from .common import IntegrityError, write_json, read_json, workspace
from .queue import worker_identity


def gpu_identity(device='cuda:0'):
    if not torch.cuda.is_available():
        raise IntegrityError('CUDA is unavailable on this machine.')
    index = torch.device(device).index or 0
    props = torch.cuda.get_device_properties(index)
    identifier = getattr(props, 'uuid', None)
    if identifier is None:
        visible = [item.strip() for item in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if item.strip()]
        selected = visible[index] if index < len(visible) else str(index)
        if selected.startswith(('GPU-', 'MIG-')):
            identifier = selected
        else:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=20, check=True)
                inventory = dict(row.strip().split(', ', 1) for row in result.stdout.splitlines()
                                 if ', ' in row)
                identifier = inventory.get(selected)
            except (OSError, subprocess.SubprocessError):
                pass
        if not identifier:
            raise IntegrityError('Cannot obtain stable GPU UUID; refuse an unsafe duplicate-worker check.')
    return {'host': socket.gethostname(), 'uuid': str(identifier), 'name': props.name,
            'total_memory_bytes': props.total_memory, 'logical_device': str(device)}


def environment_snapshot():
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'host': socket.gethostname(), 'packages': packages, 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'threads': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']}}


def doctor(cfg, cuda=False, worker_name='local', verify_workers=False, device='cuda:0'):
    from .data import inventory, open_recording
    root = workspace(cfg)
    snap = environment_snapshot()
    if sys.version_info[:2] not in [(3, 11), (3, 12)]:
        raise IntegrityError('Use the pinned Python 3.11/3.12 environment.')
    required = {'numpy': '2.2.6', 'scipy': '1.16.1', 'scikit-learn': '1.7.1', 'ripser': '0.6.12',
                'PyYAML': '6.0.2', 'matplotlib': '3.10.5', 'torch': '2.8.0', 'threadpoolctl': '3.6.0', 'psutil': '7.0.0'}
    for name, version in required.items():
        if importlib.metadata.version(name).split('+')[0] != version:
            raise IntegrityError(f'Dependency {name} differs from pinned version {version}.')
    inv = inventory(cfg)
    with open_recording(inv['source'], inv['records'][0]) as f:
        if len(f.read(8192)) != 8192:
            raise IntegrityError('Dataset stream is not readable.')
    nonce = uuid.uuid4().hex
    marker = root/'environment'/f'storage-{nonce}.json'
    write_json(marker, {'nonce': nonce})
    if read_json(marker)['nonce'] != nonce:
        raise IntegrityError('Shared-workspace atomic write/read failed.')
    snap.update(time=time.time(), worker_name=worker_name, identity=worker_identity(worker_name),
                dataset_source=inv['source'], dataset_recordings=len(inv['records']),
                dataset_bytes=inv['total_bytes'], storage_marker=str(marker))
    if cuda:
        snap['gpu'] = gpu_identity(device)
        x = torch.randn(256, 256, device=device, requires_grad=True)
        loss = (x @ x.T).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
            raise IntegrityError('CUDA forward/backward computation failed.')
    if verify_workers:
        prior = [read_json(p) for p in (root/'environment').glob('doctor-*.json')]
        gpu_markers = [p for p in prior if p.get('gpu')]
        keys = {(p['gpu']['host'], p['gpu']['uuid']) for p in gpu_markers}
        if len(keys) < cfg['budget']['gpu_workers']:
            raise IntegrityError('Run doctor --cuda once for each configured local GPU worker.')
        for p in gpu_markers:
            if not Path(p['storage_marker']).exists():
                raise IntegrityError('A local worker storage marker is not visible.')
        snap['verified_gpu_workers'] = len(keys)
    safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in worker_name)
    write_json(root/'environment'/f'doctor-{safe_name}.json', snap)
    return snap
