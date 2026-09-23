"""Runtime measurements before dataset freeze; no validation/test scores."""
from __future__ import annotations
import time
import numpy as np
import torch
from cumutoponet import SUPCON_WEIGHT
from .common import workspace, write_json, read_json, IntegrityError
from .data import inventory, sample_offsets, read_selected
from .features import feature_spec, extract_features
from .models import build_model, capacity_width, supcon_loss
from .training import make_inputs, seed_everything, rng_state, restore_rng, scaled_optimizer_step
from .environment import environment_snapshot, gpu_identity

PRINCIPAL = ['hoc', 'temporal', 'tda', 'hoc_temporal', 'hoc_tda', 'temporal_tda',
             'full', 'raw_iq', 'lstm', 'wide_temporal', 'full_supcon']


def cpu_benchmark(cfg):
    from threadpoolctl import threadpool_limits
    root = workspace(cfg)
    if (root/'data_plan.json').exists():
        raise IntegrityError('Benchmark precedes production data freeze; use existing measurements after freeze.')
    inv = inventory(cfg)
    read_rates, feature_costs = [], {128: [], 1024: []}
    with threadpool_limits(limits=1):
        for record in sorted(inv['records'], key=lambda r: -r['bytes'])[:2]:
            count = min(32, record['samples']//1024)
            offsets = sample_offsets(record['samples'], count, 1024, 42)
            started = time.monotonic()
            raw = read_selected(inv['source'], record, offsets)
            elapsed = time.monotonic()-started
            read_rates.append(record['bytes']/max(elapsed, 1e-9))
            for length in [128, 1024]:
                spec = feature_spec(cfg, length)
                begin = (1024-length)//2
                started = time.monotonic()
                extract_features(raw[:, begin:begin+length], spec)
                feature_costs[length].append((time.monotonic()-started)/count)
    inventory_records = [{key: record[key] for key in ('id', 'path', 'bytes', 'samples', 'source_metadata')}
                         for record in inv['records']]
    result = {'time': time.time(), 'inventory_source': inv['source'],
              'inventory_records': inventory_records, 'recordings': len(inv['records']),
              'total_bytes': inv['total_bytes'], 'read_bytes_per_second': min(read_rates),
              'feature_seconds_per_window': {str(k): max(v) for k, v in feature_costs.items()},
              'environment': environment_snapshot(),
              'note': 'Full sequential source reads, one feature thread; production parallel efficiency conservatively discounted.'}
    write_json(root/'benchmarks'/'cpu.json', result)
    return result


def gpu_benchmark(cfg, device='cuda:0', steps=5):
    root = workspace(cfg)
    if (root/'data_plan.json').exists():
        raise IntegrityError('GPU benchmark precedes production dataset freeze.')
    if not device.startswith('cuda') and not cfg.get('fixture', False):
        raise IntegrityError('Production estimates require an allocated CUDA device.')
    seed_everything(42)
    width, capacity = capacity_width()
    batch = cfg['training']['batch_size']
    rng = np.random.default_rng(42)
    times = {}
    for length in [128, 1024]:
        spec = feature_spec(cfg, length)
        raw = rng.normal(size=(batch, length)) + 1j*rng.normal(size=(batch, length))
        # Only feature values are synthetic; full actual tensor shapes and optimizer are benchmarked.
        arrays = {'raw': raw, 'hoc': rng.normal(size=(batch, 6)).astype('float32'),
                  'pi': rng.normal(size=(batch, 2, 20, 20)).astype('float32'),
                  'scaler': {'mean': [0.]*6, 'scale': [1.]*6}}
        models = PRINCIPAL if length == 128 else ['full', 'temporal', 'hoc_temporal', 'lstm']
        for name in models:
            model = build_model(name, width if name == 'wide_temporal' else 1.).to(device)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            enabled = device.startswith('cuda') and cfg['training']['amp']
            scaler = torch.amp.GradScaler('cuda', enabled=enabled)
            labels = torch.arange(batch, device=device) % 7
            values, inference, amp_overflows = [], [], 0
            if device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats()
            for i in range(steps+2):
                retries = 0
                while True:
                    if device.startswith('cuda'):
                        torch.cuda.synchronize()
                    started = time.monotonic()
                    batch_rng = rng_state()
                    inputs = make_inputs(arrays, np.arange(batch), device, phase_seed=i)
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=enabled):
                        logits, embedding = model(inputs)
                    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
                    if name == 'full_supcon':
                        loss += SUPCON_WEIGHT*supcon_loss(embedding, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite benchmark training loss.')
                    scaler.scale(loss).backward()
                    if scaled_optimizer_step(optimizer, scaler, model.parameters()):
                        retries += 1
                        amp_overflows += 1
                        if retries > 8:
                            raise FloatingPointError('AMP overflow persisted during GPU benchmark.')
                        restore_rng(batch_rng)
                        continue
                    break
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                elapsed = time.monotonic()-started
                started = time.monotonic()
                model.eval()
                with torch.inference_mode():
                    model(make_inputs(arrays, np.arange(batch), device))
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                if i >= 2:
                    values.append(elapsed)
                    inference.append(time.monotonic()-started)
            times[f'{name}:{length}'] = {'train_step_seconds': float(np.quantile(values, .95)),
                'validation_step_seconds': float(np.quantile(inference, .95)),
                'batch_size': batch, 'steps': steps,
                'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
                'amp_overflows': amp_overflows}
            del model, optimizer
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
    result = {'time': time.time(), 'timings': times, 'width': width, 'capacity_control': capacity,
              'environment': environment_snapshot(),
              'gpu': gpu_identity(device) if device.startswith('cuda') else None}
    write_json(root/'benchmarks'/'gpu.json', result)
    return result
