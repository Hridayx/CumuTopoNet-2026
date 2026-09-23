from concurrent.futures import ThreadPoolExecutor
import os
import socket
import time
from cumutoponet.matched.queue import JobQueue, process_evidence, worker_identity, gpu_lease
from cumutoponet.matched.common import write_json, read_json, atomic_target, IntegrityError
import pytest


def test_two_workers_cannot_claim_one_job_and_completed_evidence_is_immutable(tmp_path):
    q = JobQueue(tmp_path)
    job = q.enqueue({'phase': 'pilot', 'model': 'hoc', 'seed': 17}, priority=1)
    with ThreadPoolExecutor(2) as pool:
        claims = list(pool.map(lambda name: q.claim(worker_identity(name)), ['a', 'b']))
    claims = [c for c in claims if c is not None]
    assert len(claims) == 1
    q.finish(claims[0], 'completed', {'answer': 42})
    assert q.claim(worker_identity('c')) is None
    assert q.status()[job]['state'] == 'completed'


def test_dead_owner_recovery_and_retry_limit(tmp_path):
    q = JobQueue(tmp_path)
    jid = q.enqueue({'phase': 'pilot', 'model': 'hoc', 'seed': 17})
    for attempt in range(3):
        c = q.claim(worker_identity('worker'))
        assert c is not None
        q.fail(c, 'transient', 'test I/O interruption')
    assert q.status()[jid]['state'] == 'blocked'
    assert q.claim(worker_identity('last')) is None


def test_stale_heartbeat_alone_never_reclaims_live_or_unknown_owner(tmp_path):
    q = JobQueue(tmp_path)
    jid = q.enqueue({'phase': 'pilot', 'model': 'hoc', 'seed': 91})
    c = q.claim(worker_identity('live'))
    write_json(tmp_path/'queue'/jid/'claim'/'heartbeat.json', {'time': 0})
    report = q.recover(stale_seconds=1)
    assert report[0]['evidence']['alive'] is True
    assert q.status()[jid]['state'] == 'running'
    assert process_evidence({'host': 'unreachable-remote-host', 'pid': 1})['alive'] is None
    owner = read_json(tmp_path/'queue'/jid/'claim'/'owner.json')
    owner['pid'] = 99999999
    write_json(tmp_path/'queue'/jid/'claim'/'owner.json', owner)
    q.recover(stale_seconds=1)
    assert q.status()[jid]['state'] == 'retryable-failure'


def test_interrupted_atomic_write_preserves_previous_artifact(tmp_path):
    target = tmp_path/'artifact'
    target.write_bytes(b'previous-complete-evidence')
    try:
        with atomic_target(target) as f:
            f.write(b'partial')
            raise OSError('interruption')
    except OSError:
        pass
    assert target.read_bytes() == b'previous-complete-evidence'


def test_gpu_lease_uses_physical_uuid_not_logical_device_number(tmp_path):
    identity = {'host': 'node', 'uuid': 'physical-gpu', 'logical_device': 'cuda:0'}
    with gpu_lease(tmp_path, identity, worker_identity('a')):
        with pytest.raises(IntegrityError, match='already registered'):
            with gpu_lease(tmp_path, {**identity, 'logical_device': 'cuda:1'}, worker_identity('b')):
                pytest.fail('same physical GPU received two workers')


def test_budget_stop_does_not_steal_running_claim(tmp_path):
    q = JobQueue(tmp_path)
    running = q.enqueue({'phase': 'core', 'model': 'hoc', 'seed': 17})
    c = q.claim(worker_identity('a'))
    pending = q.enqueue({'phase': 'core', 'model': 'lstm', 'seed': 17})
    q.stop_for_budget()
    assert q.status()[running]['state'] == 'running'
    assert q.status()[pending]['state'] == 'budget-skipped'
    q.finish(c, 'completed')


def test_worker_waits_for_protocol_commit_before_claiming_core_jobs(tmp_path):
    from cumutoponet.matched.common import DEFAULTS, merge
    from cumutoponet.matched.runner import worker
    cfg = merge(DEFAULTS, {'workspace': str(tmp_path), 'fixture': True})
    write_json(tmp_path/'budget_decision.json', {'training_deadline': time.time()+600})
    q = JobQueue(tmp_path)
    jid = q.enqueue({'phase': 'core', 'model': 'hoc', 'seed': 17})
    # The planner commits several job files before the final protocol manifest.
    result = worker(cfg, 'fixture', device='cpu', once=True)
    assert result['state'] == 'idle'
    assert q.status()[jid]['state'] == 'queued'
