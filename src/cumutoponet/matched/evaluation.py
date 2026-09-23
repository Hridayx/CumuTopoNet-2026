"""Post-freeze held-out evaluation, paired uncertainty, and measured processing costs."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time
import numpy as np
import psutil
import torch
from .common import IntegrityError, read_json, write_json, write_once, workspace, atomic_target
from .data import load_arrays, save_array, LABELS, MODES, CONDITIONS
from .features import normalize, hoc_features, temporal_features, persistence_images, add_noise, feature_spec, extract_features
from .models import build_model, BRANCHES
from .training import predict, make_inputs, select_arrays
from .metrics import classification_metrics, paired_group_interval
from .queue import JobQueue


def _synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def waveform_inputs(wave, model_name, scaler, spec, device):
    z = normalize(wave)
    values = {}
    branches = BRANCHES.get(model_name, ('hoc', 'temporal', 'tda') if model_name == 'full_supcon' else ('temporal',))
    if model_name in ['lstm', 'raw_iq']:
        values['iq'] = np.stack([z.real, z.imag])[None].astype('float32')
    else:
        if 'hoc' in branches:
            values['hoc'] = ((hoc_features(z)-np.asarray(scaler['mean']))/np.asarray(scaler['scale']))[None].astype('float32')
        if 'temporal' in branches:
            values['temporal'] = temporal_features(z)[None]
        if 'tda' in branches:
            values['tda'] = persistence_images(z, spec)[None]
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in values.items()}


@torch.inference_mode()
def latency(model, arrays, model_name, device, batch_size=256, repeats=50):
    from threadpoolctl import threadpool_limits
    model.eval()
    indices = np.flatnonzero(arrays['splits'] == 'train')[:max(repeats, batch_size)]
    one = make_inputs(arrays, indices[:1], device)
    for _ in range(10):
        model(one)
    _synchronize(device)
    network, features, total = [], [], []
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    with threadpool_limits(limits=1):
        for i in range(repeats):
            _synchronize(device)
            start = time.perf_counter()
            model(one)
            _synchronize(device)
            network.append(time.perf_counter()-start)
            start = time.perf_counter()
            inputs = waveform_inputs(arrays['raw'][indices[i % len(indices)]], model_name,
                                     arrays['scaler'], arrays['feature_spec'], 'cpu')
            feature_end = time.perf_counter()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(inputs)
            _synchronize(device)
            features.append(feature_end-start)
            total.append(time.perf_counter()-start)
    batch_indices = np.resize(indices, batch_size)
    batch_inputs = make_inputs(arrays, batch_indices, device)
    for _ in range(3):
        model(batch_inputs)
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(10):
        model(batch_inputs)
    _synchronize(device)
    batch_seconds = (time.perf_counter()-start)/10
    def stats(values):
        return {'p50_ms': float(np.quantile(values, .5)*1000),
                'p95_ms': float(np.quantile(values, .95)*1000),
                'mean_ms': float(np.mean(values)*1000)}
    return {'network_batch1': stats(network), 'feature_batch1': stats(features), 'end_to_end_batch1': stats(total),
            'network_batch_throughput_windows_s': batch_size/batch_seconds,
            'batch_size': batch_size, 'repeats': repeats,
            'end_to_end_serial_windows_s': 1/float(np.mean(total)),
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
            'process_rss_bytes': psutil.Process().memory_info().rss,
            'precision': 'float32 network; complex128 moments; one CPU feature thread',
            'scope': 'resident raw windows through features, transfer, network; excludes RF hardware and disk ingestion',
            'device': str(device)}


def require_evaluation(cfg, allow_partial=False):
    from .protocol import validate_selection
    validate_selection(cfg)
    root = workspace(cfg)
    protocol = read_json(root/'protocol.json')
    states = JobQueue(root).status()
    missing = [j for j in protocol['core_job_ids'] if states.get(j, {}).get('state') != 'completed']
    if missing and not allow_partial:
        raise IntegrityError(f'{len(missing)} principal fits are unfinished; evaluation is blocked unless explicitly marked --partial.')
    if any(states[j]['state'] == 'running' for j in protocol['core_job_ids']):
        if allow_partial:
            raise IntegrityError('Do not evaluate partial core work while core training is still running.')
    return root, protocol, states, missing


def evaluate(cfg, device='cuda:0', allow_partial=False, worker_index=0, worker_count=1, measure_latency=True):
    root, protocol, states, missing = require_evaluation(cfg, allow_partial)
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError('Invalid evaluation worker shard.')
    torch.set_num_threads(1)
    jobs = [j for j, s in states.items() if s['state'] == 'completed' and
            read_json(root/'queue'/j/'spec.json')['phase'] in ['core', 'optional-long', 'optional-topology']]
    evaluated, arrays_cache = [], {}
    for jid in sorted(jobs)[worker_index::worker_count]:
        spec = read_json(root/'queue'/jid/'spec.json')
        result = read_json(root/'runs'/jid/'result.json')
        checkpoint = root/'runs'/jid/'best.pt'
        if result.get('spec') != spec or not checkpoint.is_file():
            raise IntegrityError('Checkpoint and run specification do not match the queue entry.')
        key = spec['feature_key']
        if key not in arrays_cache:
            arrays_cache.clear()
            arrays_cache[key] = load_arrays(cfg, spec['length'], spec['points'], spec['channels'])
        arrays = arrays_cache[key]
        test = np.flatnonzero(arrays['splits'] == 'test')
        out = root/'evaluation'/jid
        context = {'protocol_id': protocol['protocol_id'], 'spec': spec,
                   'feature_key': key, 'data_plan_id': arrays['data_plan_id'],
                   'fixture': protocol['fixture']}
        if (out/'complete.json').exists():
            previous = read_json(out/'complete.json')
            if previous['context'] != context:
                raise IntegrityError('Completed evaluation has incompatible settings.')
            for filename in previous['files']:
                if not (out/filename).is_file():
                    raise IntegrityError(f'Completed evaluation is missing {filename}.')
            evaluated.append(jid)
            continue
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state.get('spec') != spec:
            raise IntegrityError('Checkpoint contents do not match run spec.')
        model = build_model(spec['model'], spec['width']).to(device)
        model.load_state_dict(state['model'])
        probabilities = predict(model, arrays, test, device, cfg['training']['batch_size'])
        out.mkdir(parents=True, exist_ok=True)
        save_array(out/'probabilities.npy', probabilities)
        with atomic_target(out/'observations.npz') as f:
            np.savez(f, **{k: arrays[k][test] for k in ['ids', 'labels', 'groups', 'records', 'modes', 'conditions']})
        # Metrics are computed after reopening exactly the persisted evidence.
        obs = np.load(out/'observations.npz', allow_pickle=False)
        predictions = np.load(out/'probabilities.npy', allow_pickle=False).argmax(axis=1)
        metrics = classification_metrics(obs['labels'], predictions)
        subgroups = {}
        for field, values in [('modes', list(MODES.values())), ('conditions', list(CONDITIONS.values()))]:
            subgroups[field] = {}
            for value in values:
                mask = obs[field] == value
                subgroups[field][value] = classification_metrics(obs['labels'][mask], predictions[mask])
        # All expected class/mode/interference cells, including unsupported cells.
        cells = []
        for label in range(7):
            for mode in MODES.values():
                for condition in CONDITIONS.values():
                    mask = (obs['labels'] == label) & (obs['modes'] == mode) & (obs['conditions'] == condition)
                    cells.append({'label': LABELS[label], 'mode': mode, 'condition': condition,
                                  'support': int(mask.sum()),
                                  'recall': float((predictions[mask] == label).mean()) if mask.any() else None})
        write_json(out/'metrics.json', {'overall': metrics, 'subgroups': subgroups, 'condition_cells': cells})
        measured = latency(model, arrays, spec['model'], device, cfg['training']['batch_size']) if measure_latency else {
            'status': 'not-measured', 'reason': 'latency explicitly disabled; no deployment claims permitted'}
        write_json(out/'latency.json', measured)
        write_json(out/'complete.json', {'context': context, 'job_id': jid,
                   'files': ['observations.npz', 'probabilities.npy', 'metrics.json', 'latency.json']})
        evaluated.append(jid)
        del model, state
    write_json(root/'evaluation'/f'worker-{worker_index}-status.json', {'completed': evaluated, 'missing_core': missing,
               'worker_count': worker_count, 'protocol_id': protocol['protocol_id']})
    return {'evaluated': len(evaluated), 'missing_core': missing}


def _noise_features(task):
    from threadpoolctl import threadpool_limits
    raw, start_index, snr, snr_index, seed, spec = task
    with threadpool_limits(limits=1):
        seeds = [int(np.random.SeedSequence([seed, snr_index, start_index + offset]).generate_state(1)[0])
                 for offset in range(len(raw))]
        noisy = np.stack([add_noise(x, snr, local_seed) for x, local_seed in zip(raw, seeds)])
        hoc, pi = extract_features(noisy, spec)
    return noisy.astype('complex64'), hoc, pi


def robustness(cfg, device='cuda:0', features_only=False):
    root, protocol, states, missing = require_evaluation(cfg)
    decision = read_json(root/'optional'/'robustness.json')
    if decision['state'] != 'ready-for-evaluation' or decision['protocol_id'] != protocol['protocol_id']:
        raise IntegrityError('Robustness batch was not admitted by the budget policy.')
    arrays = load_arrays(cfg)
    arrays = select_arrays(arrays, arrays['splits'] == 'test')
    settings = protocol['optional_robustness']
    outputs = []
    for snr_index, snr in enumerate(settings['snr_db']):
        directory = root/'robustness'/f'snr-{snr}'
        request = {'protocol_id': protocol['protocol_id'], 'snr_db': snr,
                   'observations': len(arrays['ids']), 'noise_seed': settings['noise_seed'],
                   'feature_key': arrays['feature_key'], 'data_plan_id': arrays['data_plan_id']}
        if (directory/'features.json').exists():
            marker = read_json(directory/'features.json')
            if marker['request'] != request:
                raise IntegrityError('Corrupted-waveform cache is incompatible.')
            for name in marker['files']:
                if not (directory/name).is_file():
                    raise IntegrityError(f'Corrupted-waveform cache is missing {name}.')
        else:
            if not features_only:
                raise IntegrityError('Run prepare --robustness before robustness evaluation.')
            tasks = [(arrays['raw'][i:i+64], i, snr, snr_index, settings['noise_seed'], arrays['feature_spec'])
                     for i in range(0, len(arrays['ids']), 64)]
            with ProcessPoolExecutor(cfg['runtime']['feature_workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
                pieces = list(pool.map(_noise_features, tasks, chunksize=1))
            for j, name in enumerate(['raw', 'hoc', 'pi']):
                save_array(directory/f'{name}.npy', np.concatenate([p[j] for p in pieces]))
            write_json(directory/'features.json', {'request': request,
                       'files': [f'{name}.npy' for name in ['raw', 'hoc', 'pi']]})
        if features_only:
            outputs.append(str(directory))
            continue
        corrupted = dict(arrays)
        for key in ['raw', 'hoc', 'pi']:
            corrupted[key] = np.load(directory/f'{key}.npy', allow_pickle=False)
        for jid in protocol['core_job_ids']:
            spec = read_json(root/'queue'/jid/'spec.json')
            if spec['model'] not in settings['models']:
                continue
            out = directory/jid
            if (out/'complete.json').exists():
                complete = read_json(out/'complete.json')
                if complete['request'] != request or complete['spec'] != spec:
                    raise IntegrityError('Completed robustness output settings differ.')
                probabilities = np.load(out/'probabilities.npy', mmap_mode='r', allow_pickle=False)
                if probabilities.shape != (len(arrays['ids']), 7) or not np.isfinite(probabilities).all():
                    raise IntegrityError('Completed robustness probabilities are invalid.')
                outputs.append(str(out))
                continue
            result = read_json(root/'runs'/jid/'result.json')
            checkpoint = root/'runs'/jid/'best.pt'
            if result.get('spec') != spec or not checkpoint.is_file():
                raise IntegrityError('Robustness checkpoint does not match its run specification.')
            model = build_model(spec['model'], spec['width']).to(device)
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            if state.get('spec') != spec:
                raise IntegrityError('Robustness checkpoint contents do not match the run specification.')
            model.load_state_dict(state['model'])
            probabilities = predict(model, corrupted, np.arange(len(corrupted['labels'])), device, cfg['training']['batch_size'])
            save_array(out/'probabilities.npy', probabilities)
            write_json(out/'metrics.json', classification_metrics(corrupted['labels'], probabilities.argmax(axis=1)))
            write_json(out/'complete.json', {'request': request, 'spec': spec,
                       'files': ['probabilities.npy', 'metrics.json']})
            outputs.append(str(out))
            del model, state
    return {'robustness_outputs': len(outputs)}
