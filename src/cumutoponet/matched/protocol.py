"""Budget and validation-only experiment selection. Never reads test outputs."""
from __future__ import annotations
from pathlib import Path
import math
import time
import uuid
from cumutoponet import SUPCON_WEIGHT
from .common import (IntegrityError, read_json, write_json,
                     write_once, workspace, directory_lock, append_event)
from .features import feature_key, feature_spec
from .models import capacity_width
from .benchmark import PRINCIPAL
from .queue import JobQueue

GATES = ['math', 'data', 'models', 'training', 'orchestration', 'reporting']


def validate_selection(cfg):
    selection = read_json(workspace(cfg)/'selection_plan.json')
    if (selection['training'] != cfg['training'] or selection['features'] != cfg['features'] or
            selection['budget'] != cfg['budget']):
        raise IntegrityError('Selection protocol changed after pilot start.')
    return selection


def require_gates(cfg):
    if cfg.get('fixture', False):
        return
    root = workspace(cfg)
    for gate in GATES:
        p = root/'gates'/f'{gate}.json'
        if not p.exists() or read_json(p).get('status') != 'passed':
            raise IntegrityError(f'Gate {gate} has not passed: cumutopo matched-128 doctor -c CONFIG --tests {gate}')
    markers = [read_json(p) for p in (root/'environment').glob('doctor-*.json')]
    required = int(cfg['budget'].get('gpu_workers', 1))
    if len({(p['gpu']['host'], p['gpu']['uuid']) for p in markers
            if p.get('gpu')}) < required:
        raise IntegrityError(f'Run doctor --cuda once for each of the {required} configured local GPU workers.')


def fit_seconds(gpu, model, length, n_train, n_validation, epochs):
    timing = gpu['timings'][f'{model}:{length}']
    return epochs*(math.ceil(n_train/timing['batch_size'])*timing['train_step_seconds'] +
                   math.ceil(n_validation/timing['batch_size'])*timing['validation_step_seconds'])


def budget_plan(cfg):
    root = workspace(cfg)
    cpu, gpu = read_json(root/'benchmarks'/'cpu.json'), read_json(root/'benchmarks'/'gpu.json')
    path = root/'budget_decision.json'
    if path.exists():
        previous = read_json(path)
        if previous['benchmarks'] != {'cpu': cpu, 'gpu': gpu}:
            raise IntegrityError('Budget decision already frozen with different benchmarks.')
        return previous
    if (root/'data_plan.json').exists():
        raise IntegrityError('Cannot change sample count after production data freeze.')
    # Exact family partition proportions from the inventory, rather than assumed 60/20/20.
    from .data import inventory, make_partitions
    inv = inventory(cfg)
    inventory_records = [{key: record[key] for key in ('id', 'path', 'bytes', 'samples', 'source_metadata')}
                         for record in inv['records']]
    if inv['source'] != cpu['inventory_source'] or inventory_records != cpu['inventory_records']:
        raise IntegrityError('Inventory changed after the CPU benchmark.')
    folds = make_partitions(inv['records'])
    train_records = sum(folds[r['id']] >= 2 for r in inv['records'])
    val_records = sum(folds[r['id']] == 1 for r in inv['records'])
    safety = cfg['budget']['safety_factor']
    limit = min(cfg['budget']['core_target_hours'], cfg['budget']['hours']-cfg['budget']['reserve_hours'])
    estimates = {}
    candidates = [cfg['dataset']['windows_per_recording']]
    if candidates[0] == 1024:
        candidates.append(512)
    for count in candidates:
        windows = count*len(inv['records'])
        # 50% parallel efficiency for readers, 70% for feature processes.
        read_time = cpu['total_bytes']/(cpu['read_bytes_per_second']*max(1, cfg['runtime']['archive_readers']*.5))
        feature_time = windows*cpu['feature_seconds_per_window']['128']/max(1, cfg['runtime']['feature_workers']*.7)
        n, v = train_records*count, val_records*count
        core_times = [fit_seconds(gpu, m, 128, n, v, cfg['training']['max_epochs']) for m in PRINCIPAL
                      for _ in cfg['training']['seeds']]
        pilot_epochs = 1 if cfg.get('fixture', False) else 10
        pilot_times = [fit_seconds(gpu, m, 128, n, v, pilot_epochs)
                       for m in ['full', 'temporal', 'raw_iq', 'lstm'] for _ in range(2)]
        # Separate dependent stages; add the longest fit for scheduling imbalance.
        def parallel_time(jobs):
            workers = cfg['budget']['gpu_workers']
            return sum(jobs)/workers + max(jobs)*(1-1/workers)
        gpu_seconds = sum(parallel_time(j) for j in [pilot_times, core_times])
        total = safety*(read_time+feature_time+gpu_seconds)
        estimates[str(count)] = {'estimated_hours': total/3600, 'read_seconds': read_time,
                                'feature_seconds': feature_time, 'gpu_wall_seconds': gpu_seconds,
                                'safety_factor': safety}
        if total <= limit*3600:
            break
    selected = int(list(estimates)[-1])
    now = time.time()
    result = {'windows_per_recording': selected, 'estimates': estimates,
              'feasible': estimates[str(selected)]['estimated_hours'] <= limit,
              'core_limit_hours': limit, 'started_at': now,
              'training_deadline': now+(cfg['budget']['hours']-cfg['budget']['reserve_hours'])*3600,
              'overall_deadline': now+cfg['budget']['hours']*3600,
              'benchmarks': {'cpu': cpu, 'gpu': gpu},
              'rationale': 'Smallest permitted matched sample count needed to fit conservative measured budget; no accuracy examined.'}
    write_once(path, result)
    append_event(root, 'budget-frozen', result)
    return result


def base_spec(cfg, model, seed, lr, phase, length=128, width=1., points=64, channels=None):
    root = workspace(cfg)
    data = read_json(root/'data_plan.json')
    spec = feature_spec(cfg, length, points, channels)
    training = dict(cfg['training'])
    if phase.startswith('pilot'):
        epochs = 1 if cfg.get('fixture', False) else 10
        training.update(max_epochs=epochs, min_epochs=epochs)
    return {'phase': phase, 'model': model, 'seed': seed, 'lr': lr, 'width': width,
            'supcon_weight': SUPCON_WEIGHT if model == 'full_supcon' else 0.,
            'training': training, 'length': length, 'points': points,
            'channels': channels or [0, 1], 'feature_key': feature_key(spec),
            'data_plan_id': data['plan_id'], 'fixture': cfg.get('fixture', False)}


def start_pilots(cfg):
    require_gates(cfg)
    root = workspace(cfg)
    if not read_json(root/'budget_decision.json')['feasible']:
        raise IntegrityError('Even reduced sampling exceeds measured budget. Resolve capacity/time before starting; do not remove controls silently.')
    q = JobQueue(root)
    # Freeze all nonlocal settings at the first use of validation outcomes.
    selection = {'training': cfg['training'], 'features': cfg['features'], 'budget': cfg['budget'],
                 'lr_grid': [.0003, .001], 'supcon_weight': SUPCON_WEIGHT,
                 'criterion': 'validation macro-F1; ties select smaller hyperparameter',
                 'principal': PRINCIPAL, 'data_plan_id': read_json(root/'data_plan.json')['plan_id']}
    write_once(root/'selection_plan.json', selection)
    jobs = []
    for model in ['full', 'temporal', 'raw_iq', 'lstm']:
        for lr in [.0003, .001]:
            jobs.append(q.enqueue(base_spec(cfg, model, 17, lr, 'pilot-lr'), priority=1))
    return {'stage': 'pilot-lr', 'jobs': jobs}


def phase_results(root, phase):
    q = JobQueue(root)
    selected = [(jid, state, read_json(q.path/jid/'spec.json')) for jid, state in q.status().items()
                if read_json(q.path/jid/'spec.json')['phase'] == phase]
    if not selected or any(state['state'] != 'completed' for _, state, _ in selected):
        return None
    results = []
    for jid, state, spec in selected:
        result = read_json(Path(root)/'runs'/jid/'result.json')
        if result.get('spec') != spec:
            raise IntegrityError('Pilot result does not match its queued specification.')
        results.append((jid, spec, result))
    return results


def advance(cfg):
    root = workspace(cfg)
    with directory_lock(root/'.plan.lock'):
        selection_path = root/'selection_plan.json'
        if not selection_path.exists():
            return {'stage': 'needs-pilots', 'action': 'plan --pilots'}
        selection = validate_selection(cfg)
        protocol_path = root/'protocol.json'
        if protocol_path.exists():
            protocol = read_json(protocol_path)
            q = JobQueue(root)
            states = q.status()
            complete = all(states.get(j, {}).get('state') == 'completed' for j in protocol['core_job_ids'])
            return {'stage': 'core-completed' if complete else 'core-running',
                    'protocol_id': protocol['protocol_id']}
        pilots = phase_results(root, 'pilot-lr')
        if pilots is None:
            return {'stage': 'pilot-lr-running'}
        if len(pilots) != 8:
            raise IntegrityError('Expected exactly eight learning-rate pilot fits.')
        chosen = {}
        for model in ['full', 'temporal', 'raw_iq', 'lstm']:
            candidates = [(spec['lr'], result['best_validation_macro_f1']) for _, spec, result in pilots if spec['model'] == model]
            chosen[model] = sorted(candidates, key=lambda p: (-p[1], p[0]))[0][0]
        q = JobQueue(root)
        width, capacity = capacity_width()
        jobs, specs = [], []
        for i, seed in enumerate(cfg['training']['seeds']):
            for model in PRINCIPAL:
                lr = chosen.get(model, chosen['temporal'] if model == 'wide_temporal' else chosen['full'])
                spec = base_spec(cfg, model, seed, lr, 'core',
                                 width=width if model == 'wide_temporal' else 1.)
                jobs.append(q.enqueue(spec, priority=10+i))
                specs.append(spec)
        protocol = {'core_job_ids': jobs, 'core_specs': specs, 'selected_learning_rates': chosen,
                    'supcon_weight': SUPCON_WEIGHT, 'capacity_width': width,
                    'capacity_audit': capacity, 'selection_plan': selection,
                    'data_plan_id': selection['data_plan_id'], 'protocol_id': uuid.uuid4().hex,
                    'frozen_at': time.time(), 'fixture': cfg.get('fixture', False),
                    'optional_order': ['long', 'topology', 'robustness'],
                    'optional_long': {'models': ['full', 'temporal', 'hoc_temporal', 'lstm'], 'length': 1024},
                    'optional_topology': {'model': 'full', 'variants': [[64, [0]], [64, [1]], [32, [0, 1]], [96, [0, 1]]]},
                    'optional_robustness': {'models': ['full', 'temporal', 'hoc_temporal', 'lstm'],
                        'snr_db': [20, 10, 0], 'noise_seed': 20260910, 'noise': 'complex AWGN; all views recomputed'},
                    'evaluation': {'bootstrap_replicates': 2000, 'bootstrap_seed': 42,
                        'primary_contrasts': ['temporal', 'hoc_temporal', 'temporal_tda', 'raw_iq', 'lstm', 'wide_temporal'],
                        'interval_scope': 'fixed split and fixed seeds; recording-family resampling'}}
        write_once(protocol_path, protocol)
        append_event(root, 'protocol-frozen', {'protocol_id': protocol['protocol_id'],
                                               'principal_fits': len(jobs)})
        return {'stage': 'core-running', 'principal_fits': len(jobs),
                'protocol_id': protocol['protocol_id']}


def optional_batch(cfg, batch):
    root = workspace(cfg)
    validate_selection(cfg)
    protocol = read_json(root/'protocol.json')
    q, states = JobQueue(root), JobQueue(root).status()
    if not all(states[j]['state'] == 'completed' for j in protocol['core_job_ids']):
        raise IntegrityError('Finish all principal fits before optional work.')
    if batch not in protocol['optional_order']:
        raise ValueError('Optional batch must be long, topology or robustness.')
    for earlier in protocol['optional_order'][:protocol['optional_order'].index(batch)]:
        path = root/'optional'/f'{earlier}.json'
        if not path.exists():
            raise IntegrityError(f'Assess optional {earlier} before {batch}.')
        decision = read_json(path)
        if decision['state'] != 'budget-skipped' and (decision['state'] != 'queued' or
                not decision.get('job_ids') or not all(states.get(j, {}).get('state') == 'completed' for j in decision['job_ids'])):
            raise IntegrityError(f'Complete optional {earlier} first.')
    cpu, gpu = read_json(root/'benchmarks'/'cpu.json'), read_json(root/'benchmarks'/'gpu.json')
    data = read_json(root/'data_plan.json')
    count = data['windows_per_recording']
    n = count*sum(v >= 2 for v in data['partitions'].values())
    v = count*sum(v == 1 for v in data['partitions'].values())
    specs, features = [], []
    if batch == 'long':
        for model in protocol['optional_long']['models']:
            for seed in cfg['training']['seeds']:
                specs.append(base_spec(cfg, model, seed, protocol['selected_learning_rates'].get(model,
                    protocol['selected_learning_rates']['full']), 'optional-long', length=1024))
    elif batch == 'topology':
        for points, channels in protocol['optional_topology']['variants']:
            for seed in cfg['training']['seeds']:
                specs.append(base_spec(cfg, 'full', seed, protocol['selected_learning_rates']['full'],
                    'optional-topology', points=points, channels=channels))
    costs = []
    for s in specs:
        # Point sensitivity changes CPU preprocessing, not tensor/network size.
        cost = fit_seconds(gpu, s['model'], s['length'], n, v, cfg['training']['max_epochs'])
        costs.append(cost)
        fs = feature_spec(cfg, s['length'], s['points'], s['channels'])
        if fs not in features:
            features.append(fs)
    feature_seconds = sum(count*len(data['partitions'])*cpu['feature_seconds_per_window'][str(f['length'])]
                          *(f['tda_points']/64)**3/max(1, cfg['runtime']['feature_workers']*.7) for f in features)
    if batch == 'robustness':
        n_test = count*sum(fold == 0 for fold in data['partitions'].values())
        feature_seconds = 3*n_test*cpu['feature_seconds_per_window']['128']/max(1, cfg['runtime']['feature_workers']*.7)
        costs = [3*len(cfg['training']['seeds'])*math.ceil(n_test/cfg['training']['batch_size'])*
                 gpu['timings'][f'{m}:128']['validation_step_seconds'] for m in protocol['optional_robustness']['models']]
    estimated = cfg['budget']['safety_factor']*(feature_seconds + sum(costs)/cfg['budget']['gpu_workers'] + max(costs, default=0))
    deadline = read_json(root/'budget_decision.json')['training_deadline']
    decision_path = root/'optional'/f'{batch}.json'
    if decision_path.exists() and read_json(decision_path)['state'] in ('queued', 'budget-skipped', 'ready-for-evaluation'):
        return read_json(decision_path)
    if time.time()+estimated > deadline:
        decision = {'batch': batch, 'state': 'budget-skipped', 'estimated_seconds': estimated,
                    'reason': 'Entire preregistered optional batch does not fit before evaluation reserve.', 'job_ids': []}
    else:
        missing = [f for f in features if not (root/'cache'/feature_key(f)/'complete.json').exists()]
        if missing:
            decision = {'batch': batch, 'state': 'needs-features', 'estimated_seconds': estimated,
                        'feature_requests': [{'length': f['length'], 'points': f['tda_points'], 'channels': f['tda_channels']} for f in missing],
                        'job_ids': []}
        else:
            ids = [q.enqueue(s, priority=30+protocol['optional_order'].index(batch)) for s in specs]
            decision = {'batch': batch, 'state': 'ready-for-evaluation' if batch == 'robustness' else 'queued',
                        'estimated_seconds': estimated, 'job_ids': ids,
                        'protocol_id': protocol['protocol_id']}
    write_json(decision_path, decision)
    append_event(root, 'optional-decision', decision)
    return decision
