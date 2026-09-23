"""Atomic filesystem queue for concurrent local worker processes."""
from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import re
import socket
import time
import uuid
import psutil
from .common import (IntegrityError, write_json, read_json, write_once,
                     directory_lock, append_event)

def worker_identity(name):
    return {'name': name, 'host': socket.gethostname(), 'pid': os.getpid(),
            'process_created': psutil.Process().create_time(), 'started': time.time()}


def process_evidence(owner):
    if owner.get('host') == socket.gethostname():
        try:
            process = psutil.Process(owner['pid'])
            same = abs(process.create_time()-owner.get('process_created', process.create_time())) < .1
            alive = process.is_running() and process.status() != psutil.STATUS_ZOMBIE and same
            return {'alive': alive, 'source': 'same-host PID and process creation time'}
        except psutil.NoSuchProcess:
            return {'alive': False, 'source': 'same-host PID absent'}
        except psutil.AccessDenied:
            return {'alive': None, 'source': 'process inspection denied'}
    return {'alive': None, 'source': 'remote host cannot be inspected; do not reclaim'}


class JobQueue:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root/'queue'
        self.path.mkdir(parents=True, exist_ok=True)

    def enqueue(self, spec, priority=10):
        with directory_lock(self.path/'.enqueue.lock'):
            existing = [directory for directory in sorted(self.path.iterdir())
                        if directory.is_dir() and (directory/'spec.json').exists()]
            for directory in existing:
                if read_json(directory/'spec.json') == spec:
                    return directory.name
            phase = re.sub(r'[^A-Za-z0-9_.-]+', '-', str(spec['phase'])).strip('-')
            model = re.sub(r'[^A-Za-z0-9_.-]+', '-', str(spec['model'])).strip('-')
            sequence = len(existing) + 1
            while True:
                jid = f'{priority:03d}-{phase}-{model}-{sequence:04d}'
                directory = self.path/jid
                if not directory.exists():
                    break
                sequence += 1
            directory.mkdir()
            write_once(directory/'spec.json', spec)
            write_json(directory/'state.json', {'state': 'queued', 'attempts': 0,
                       'priority': priority, 'updated': time.time()})
        return jid

    def status(self):
        return {d.name: read_json(d/'state.json') for d in sorted(self.path.iterdir())
                if d.is_dir() and (d/'state.json').exists()}

    def claim(self, owner, allowed_phases=None):
        for jid, state in self.status().items():
            if state['state'] not in ('queued', 'retryable-failure'):
                continue
            directory = self.path/jid
            spec = read_json(directory/'spec.json')
            if allowed_phases and spec['phase'] not in allowed_phases:
                continue
            try:
                (directory/'claim').mkdir()
            except FileExistsError:
                continue
            latest = read_json(directory/'state.json')
            if latest['state'] not in ('queued', 'retryable-failure'):
                (directory/'claim').rmdir()
                continue
            token = uuid.uuid4().hex
            write_json(directory/'claim'/'owner.json', {**owner, 'token': token})
            write_json(directory/'claim'/'heartbeat.json', {'time': time.time()})
            latest.update(state='running', attempts=latest['attempts']+1, updated=time.time(), owner=owner)
            write_json(directory/'state.json', latest)
            append_event(self.root, 'claim', {'job': jid, 'owner': owner, 'attempt': latest['attempts']})
            return {'id': jid, 'spec': spec, 'token': token}
        return None

    def heartbeat(self, claim):
        self._check_owner(claim)
        write_json(self.path/claim['id']/'claim'/'heartbeat.json', {'time': time.time()})

    def _check_owner(self, claim):
        owner = read_json(self.path/claim['id']/'claim'/'owner.json')
        if owner['token'] != claim['token']:
            raise IntegrityError('Worker no longer owns this job claim.')

    def finish(self, claim, state, details=None):
        if state not in ('completed', 'retryable-failure', 'blocked', 'budget-skipped'):
            raise ValueError('Invalid terminal/recovery state.')
        self._check_owner(claim)
        directory = self.path/claim['id']
        current = read_json(directory/'state.json')
        if current['state'] != 'running':
            raise IntegrityError('Only a running owner can finish a job.')
        current.update(state=state, details=details or {}, updated=time.time())
        write_json(directory/'state.json', current)
        # Preserve owner/heartbeat evidence rather than delete it.
        (directory/'claim').rename(directory/f'claim-closed-{claim["token"]}')
        append_event(self.root, state, {'job': claim['id'], 'details': details or {}})

    def fail(self, claim, category, message):
        attempts = read_json(self.path/claim['id']/'state.json')['attempts']
        state = 'retryable-failure' if category == 'transient' and attempts <= 2 else 'blocked'
        self.finish(claim, state, {'category': category, 'message': message, 'automatic_retry_limit': 2})

    def recover(self, stale_seconds=180):
        findings = []
        for jid, state in self.status().items():
            directory = self.path/jid
            claim_dir = directory/'claim'
            if not claim_dir.exists():
                continue
            owner_path = claim_dir/'owner.json'
            heartbeat_path = claim_dir/'heartbeat.json'
            if not owner_path.exists() or not heartbeat_path.exists():
                findings.append({'job': jid, 'action': 'blocked', 'reason': 'Incomplete claim; owner unknown. Manual audit required.'})
                continue
            age = time.time()-read_json(heartbeat_path)['time']
            if age <= stale_seconds:
                continue
            owner = read_json(owner_path)
            evidence = process_evidence(owner)
            findings.append({'job': jid, 'heartbeat_age_seconds': age, 'evidence': evidence})
            if evidence['alive'] is not False:
                continue
            try:
                with directory_lock(directory/'.recover.lock'):
                    # Recheck after acquiring recovery ownership.
                    if read_json(owner_path) != owner or process_evidence(owner)['alive'] is not False:
                        continue
                    claim_dir.rename(directory/f'claim-recovered-{uuid.uuid4().hex}')
                    current = read_json(directory/'state.json')
                    if current['state'] in ('running', 'queued', 'retryable-failure'):
                        next_state = 'retryable-failure' if current['attempts'] <= 2 else 'blocked'
                        current.update(state=next_state, updated=time.time(), details={'recovery_evidence': evidence})
                        write_json(directory/'state.json', current)
                    append_event(self.root, 'recovered', {'job': jid, 'evidence': evidence})
            except IntegrityError:
                continue
        return findings

    def stop_for_budget(self, phases=None):
        for jid, state in self.status().items():
            directory = self.path/jid
            spec = read_json(directory/'spec.json')
            if phases and spec['phase'] not in phases:
                continue
            if state['state'] in ('queued', 'retryable-failure'):
                # Use the same atomic claim path as workers to avoid a claim/skip race.
                try:
                    (directory/'claim').mkdir()
                except FileExistsError:
                    continue
                latest = read_json(directory/'state.json')
                if latest['state'] in ('queued', 'retryable-failure'):
                    latest.update(state='budget-skipped', updated=time.time(), details={'reason': 'evaluation reserve reached'})
                    write_json(directory/'state.json', latest)
                (directory/'claim').rmdir()


@contextmanager
def gpu_lease(root, identity, owner):
    label = re.sub(r'[^A-Za-z0-9_.-]+', '-',
                   f"{identity['host']}-{identity['uuid']}").strip('-')
    directory = Path(root)/'gpu_locks'/label
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError as e:
        raise IntegrityError(f'GPU worker already registered: {directory}. Monitor must verify its owner before recovery.') from e
    token = uuid.uuid4().hex
    write_json(directory/'owner.json', {**owner, 'gpu_identity': identity, 'token': token})
    try:
        yield
    finally:
        if (directory/'owner.json').exists() and read_json(directory/'owner.json')['token'] == token:
            (directory/'owner.json').unlink()
            directory.rmdir()
