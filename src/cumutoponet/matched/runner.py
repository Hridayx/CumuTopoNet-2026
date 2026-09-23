"""One local process per GPU plus a validation-only monitor."""
from __future__ import annotations
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import threading
import time
import traceback
import torch
from .common import IntegrityError, workspace, write_json, read_json, append_event
from .queue import JobQueue, worker_identity, gpu_lease, process_evidence
from .environment import gpu_identity
from .data import load_arrays
from .training import train_run, training_arrays
from .protocol import require_gates, advance


def failure_category(error):
    if isinstance(error, OSError) and error.errno in {
            errno.EIO, errno.ETIMEDOUT, errno.ECONNRESET, errno.ESTALE, errno.ENETUNREACH, errno.EHOSTUNREACH}:
        return 'transient'
    if isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower():
        return 'oom'
    if isinstance(error, FloatingPointError) or 'non-finite' in str(error).lower():
        return 'numerical'
    if isinstance(error, IntegrityError):
        return 'integrity'
    return 'configuration-or-code'


def worker(cfg, name, device='cuda:0', once=False):
    require_gates(cfg)
    if not device.startswith('cuda') and not cfg.get('fixture', False):
        raise IntegrityError('CPU training is restricted to explicitly marked fixtures.')
    torch.set_num_threads(1)
    root, owner = workspace(cfg), worker_identity(name)
    identity = gpu_identity(device) if device.startswith('cuda') else {'host': owner['host'], 'uuid': f'fixture-cpu-{name}'}
    q = JobQueue(root)
    current, heartbeat_error = {'claim': None}, []
    stop = threading.Event()
    worker_file = root/'workers'/f'{owner["host"]}-{owner["pid"]}.json'
    def heartbeat():
        while not stop.is_set():
            try:
                claim = current['claim']
                write_json(worker_file, {'owner': owner, 'gpu': identity, 'time': time.time(),
                                        'job': claim['id'] if claim else None})
                if claim:
                    q.heartbeat(claim)
            except Exception as e:
                # A completed claim may close between the local snapshot and the heartbeat.
                if current['claim'] is claim:
                    heartbeat_error.append(str(e))
            stop.wait(cfg['runtime']['heartbeat_seconds'])
    with gpu_lease(root, identity, owner):
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        arrays_cache = {}
        try:
            while True:
                budget = read_json(root/'budget_decision.json')
                if time.time() >= budget['training_deadline']:
                    q.stop_for_budget()
                    return {'state': 'budget-stopped'}
                if heartbeat_error:
                    raise IntegrityError('Worker heartbeat failed: ' + heartbeat_error[-1])
                blocks = list((root/'stage_blocks').glob('*.json')) if (root/'stage_blocks').exists() else []
                if blocks:
                    return {'state': 'blocked', 'reason': 'A scientific stage requires diagnosis.',
                            'blocks': [read_json(p) for p in blocks]}
                # The planner commits the final manifest after enqueueing all core
                # jobs. Workers must wait through that brief transaction window.
                allowed = None if (root/'protocol.json').exists() else ['pilot-lr']
                claim = q.claim(owner, allowed_phases=allowed)
                if claim is None:
                    if once:
                        return {'state': 'idle'}
                    protocol = root/'protocol.json'
                    optional_pending = [s for s in q.status().values() if s['state'] in ('queued', 'running', 'retryable-failure')]
                    if protocol.exists() and not optional_pending:
                        write_json(worker_file, {'owner': owner, 'gpu': identity, 'time': time.time(),
                                                'job': None, 'state': 'idle-awaiting-optional-or-evaluation'})
                    stop.wait(10)
                    continue
                current['claim'] = claim
                spec = claim['spec']
                output = root/'runs'/claim['id']
                try:
                    if spec['phase'] == 'core' and not (root/'protocol.json').exists():
                        raise IntegrityError('Principal experiments require a frozen protocol.')
                    key = spec['feature_key']
                    if key not in arrays_cache:
                        # Keep only one representation in RAM per worker.
                        arrays_cache.clear()
                        arrays_cache[key] = training_arrays(load_arrays(cfg, spec['length'], spec['points'], spec['channels']))
                    result = train_run(spec, arrays_cache[key], output, device, deadline=budget['training_deadline'])
                    # Clear heartbeat target before closing its claim directory.
                    current['claim'] = None
                    if result['status'] == 'interrupted':
                        if result.get('budget_stop'):
                            q.finish(claim, 'budget-skipped', {'reason': 'checkpoint saved at evaluation reserve'})
                        else:
                            q.fail(claim, 'transient', f'Termination signal {result.get("signal")}; checkpoint preserved')
                        return result
                    q.finish(claim, 'completed', {'result': str(output/'result.json')})
                except Exception as e:
                    current['claim'] = None
                    category = failure_category(e)
                    diagnostic = {'category': category, 'message': str(e), 'traceback': traceback.format_exc(),
                                  'job': claim['id'], 'time': time.time(), 'owner': owner}
                    write_json(output/f'failure-{time.time_ns()}.json', diagnostic)
                    q.fail(claim, category, str(e))
                    # Numerical/OOM/integrity failures stop this worker for diagnosis.
                    if category != 'transient':
                        write_json(root/'stage_blocks'/f'{spec["phase"]}.json', diagnostic)
                        return {'state': 'blocked', **diagnostic}
                if once:
                    return {'state': 'one-job-finished', 'job': claim['id']}
        finally:
            current['claim'] = None
            stop.set()
            thread.join(timeout=5)
            write_json(worker_file, {'owner': owner, 'gpu': identity, 'time': time.time(), 'state': 'stopped'})


def monitor(cfg, recover=False, advance_protocol=False):
    root, q = workspace(cfg), JobQueue(workspace(cfg))
    findings = q.recover(stale_seconds=3*cfg['runtime']['heartbeat_seconds']) if recover else []
    gpu_recovery = []
    stage_recovery = []
    if recover:
        stage_locks = [root/'.prepare.lock', root/'.plan.lock']
        stage_locks += [root/'queue'/'.enqueue.lock']
        stage_locks += list((root/'queue').glob('*/.recover.lock'))
        for directory in stage_locks:
            owner_path = directory/'owner.json'
            if not owner_path.exists():
                continue
            owner = read_json(owner_path)
            evidence = process_evidence(owner)
            if evidence['alive'] is False:
                target = root/'recovered_stage_locks'/f'{directory.name}-{time.time_ns()}'
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    directory.rename(target)
                    stage_recovery.append({'lock': str(directory), 'evidence': evidence})
                except FileNotFoundError:
                    pass
    if recover and (root/'gpu_locks').exists():
        for directory in (root/'gpu_locks').iterdir():
            p = directory/'owner.json'
            if not p.exists():
                gpu_recovery.append({'lock': str(directory), 'action': 'needs-owner-evidence'})
                continue
            owner = read_json(p)
            evidence = process_evidence(owner)
            if evidence['alive'] is False:
                # Move to an evidence directory; recovery never kills a process.
                target = root/'recovered_gpu_locks'/f'{directory.name}-{time.time_ns()}'
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    directory.rename(target)
                    gpu_recovery.append({'lock': str(directory), 'recovered': True, 'evidence': evidence})
                except FileNotFoundError:
                    pass
    stage = None
    if advance_protocol:
        try:
            stage = advance(cfg)
        except IntegrityError as e:
            stage = {'state': 'blocked', 'message': str(e)}
    states = q.status()
    counts = {}
    for state in states.values():
        counts[state['state']] = counts.get(state['state'], 0)+1
    result = {'time': time.time(), 'counts': counts, 'jobs': states, 'recovery_findings': findings,
              'gpu_recovery': gpu_recovery, 'stage_recovery': stage_recovery, 'protocol': stage,
              'stage_blocks': [read_json(p) for p in (root/'stage_blocks').glob('*.json')] if (root/'stage_blocks').exists() else [],
              'workers': [read_json(p) for p in (root/'workers').glob('*.json')] if (root/'workers').exists() else []}
    write_json(root/'status.json', result)
    append_event(root, 'monitor', {'counts': counts, 'protocol': stage})
    return result
