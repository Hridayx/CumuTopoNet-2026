from __future__ import annotations

import subprocess
import time
import traceback

from cumutoponet.matched.environment import gpu_identity
from cumutoponet.matched.queue import JobQueue, gpu_lease, worker_identity

from .common import IntegrityError, atomic_json, workspace
from .training import train_run


def _gpu_record():
    command = ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {"inventory": result.stdout.strip().splitlines(), "returncode": result.returncode}
    except (OSError, subprocess.TimeoutExpired):
        return {"inventory": [], "returncode": None}


def _worker(cfg, name, device="cuda:0", once=False):
    root = workspace(cfg)
    queue = JobQueue(root)
    owner = worker_identity(name)
    status_path = root / "workers" / f"{name}.json"
    atomic_json(status_path, {"state": "starting", "owner": owner, "gpu": _gpu_record(),
                              "updated": time.time()})
    while True:
        allowed = []
        if ((root / "preparation_complete.json").exists()
                and (root / "protocol.json").exists()):
            allowed.extend(["legacy", "core"])
        if ((root / "preparation_complete.json").exists()
                and (root / "extended_manifest.json").exists()):
            from .extended import EXTENDED_PHASES, advance_lstm
            allowed.extend(sorted(EXTENDED_PHASES))
            advance_lstm(cfg)
        if ((root / "sensitivity_complete.json").exists()
                and (root / "protocol.json").exists()):
            allowed.append("tda-sensitivity")
        claim = queue.claim(owner, allowed_phases=allowed) if allowed else None
        if claim is None:
            counts = {}
            for state in queue.status().values():
                counts[state["state"]] = counts.get(state["state"], 0) + 1
            atomic_json(status_path, {"state": "idle", "owner": owner, "allowed_phases": allowed,
                                      "queue_counts": counts, "updated": time.time()})
            if once:
                return {"state": "idle", "counts": counts}
            time.sleep(int(cfg["runtime"]["heartbeat_seconds"]))
            continue
        atomic_json(status_path, {"state": "running", "owner": owner, "job": claim["id"],
                                  "spec": claim["spec"], "updated": time.time()})
        try:
            result = train_run(cfg, claim["spec"], claim["id"], device,
                               heartbeat=lambda: queue.heartbeat(claim))
            if result.get("status") == "interrupted":
                queue.fail(claim, "transient", "Worker received a termination signal; checkpoint preserved.")
                return result
            queue.finish(claim, "completed", {"result": str(root / "runs" / claim["id"] / "result.json"),
                                               "test_accuracy": None if result["test"] is None else result["test"]["accuracy"]})
        except Exception as error:
            category = "transient" if isinstance(error, (OSError, torch_cuda_oom())) else "scientific"
            message = f"{type(error).__name__}: {error}"
            queue.fail(claim, category, message)
            atomic_json(root / "diagnostics" / f"{claim['id']}-{time.time_ns()}.json",
                        {"job": claim["id"], "category": category, "message": message,
                         "traceback": traceback.format_exc(), "time": time.time()})
            if category != "transient":
                if once:
                    return {"state": "blocked", "job": claim["id"], "message": message}


def worker(cfg, name, device="cuda:0", once=False):
    if not device.startswith("cuda"):
        raise IntegrityError("Production training requires a CUDA device.")
    root = workspace(cfg)
    owner = worker_identity(name)
    identity = gpu_identity(device)
    with gpu_lease(root, identity, owner):
        return _worker(cfg, name, device, once)


def torch_cuda_oom():
    try:
        import torch
        return torch.cuda.OutOfMemoryError
    except Exception:
        return MemoryError


def status(cfg, recover=False):
    root = workspace(cfg)
    queue = JobQueue(root)
    findings = queue.recover(stale_seconds=180) if recover else []
    counts = {}
    for state in queue.status().values():
        counts[state["state"]] = counts.get(state["state"], 0) + 1
    workers = {}
    for path in sorted((root / "workers").glob("*.json")) if (root / "workers").exists() else []:
        import json
        workers[path.stem] = json.loads(path.read_text())
    result = {"time": time.time(), "queue_counts": counts, "workers": workers,
              "base_prepared": (root / "preparation_complete.json").exists(),
              "sensitivity_prepared": (root / "sensitivity_complete.json").exists(),
              "recovery_findings": findings}
    atomic_json(root / "status.json", result)
    return result
