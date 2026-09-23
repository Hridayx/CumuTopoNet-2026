from __future__ import annotations

import time
import uuid

from cumutoponet import SUPCON_WEIGHT
from cumutoponet.matched.queue import JobQueue

from .common import IntegrityError, read_json, workspace, write_once
from .data import TDA_VARIANTS
from .models import model_summary


CORE_MODELS = ["hoc", "temporal", "tda", "hoc_temporal", "hoc_tda",
               "temporal_tda", "full", "full_ce", "wide_temporal", "raw_iq", "lstm"]
TASKS = ["drone7", "synthetic9"]


def _base(cfg, phase, profile, task, model, seed, **extra):
    return {"phase": phase, "profile": profile, "task": task, "model": model,
            "seed": int(seed), "training": dict(cfg["training"]),
            "evaluate_test": True, **extra}


def experiment_specs(cfg):
    specs = []
    manual = [1., 1.2, 1., 2., 2., 1., 1., .3, .3]
    specs.extend([
        _base(cfg, "legacy", "legacy", "synthetic9", "full", 42,
              recipe="v2.5", sampler_power=1., supcon_weight=SUPCON_WEIGHT,
              manual_class_weights=manual),
        _base(cfg, "legacy", "legacy", "synthetic9", "full", 42,
              recipe="v2.6a", sampler_power=.5, supcon_weight=SUPCON_WEIGHT),
    ])
    for task in TASKS:
        for model in CORE_MODELS:
            weight = 0. if model in ("full_ce", "raw_iq", "lstm") else SUPCON_WEIGHT
            specs.append(_base(cfg, "core", "corrected", task, model, 42,
                               sampler_power=1., supcon_weight=weight))
    for variant in TDA_VARIANTS:
        training = {**cfg["training"], "max_epochs": 15, "patience": 5}
        spec = _base(cfg, "tda-sensitivity", "corrected", "drone7", "tda", 42,
                     sampler_power=1., supcon_weight=SUPCON_WEIGHT, sensitivity=variant,
                     evaluate_test=False)
        spec["training"] = training
        specs.append(spec)
    return specs


def _priority(spec):
    if spec["phase"] == "legacy":
        return 10
    if spec["phase"] == "core" and spec["model"] == "full":
        return 20
    if spec["phase"] == "core":
        return 30
    return 50


def commit_plan(cfg):
    root = workspace(cfg)
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        plan = read_json(protocol_path)
        if plan.get("suite") != "standard-1024" or plan.get("specs") != experiment_specs(cfg):
            raise IntegrityError("Existing standard plan uses different settings.")
        return plan
    specs = experiment_specs(cfg)
    queue = JobQueue(root)
    ids = [queue.enqueue(spec, _priority(spec)) for spec in specs]
    counts = {phase: sum(spec["phase"] == phase for spec in specs)
              for phase in sorted({spec["phase"] for spec in specs})}
    plan = {"status": "committed", "created_at": time.time(),
            "protocol_id": uuid.uuid4().hex, "suite": "standard-1024",
            "test_policy": "Sensitivity runs do not evaluate the test partition.",
            "architecture": model_summary(), "supcon_weight": SUPCON_WEIGHT,
            "counts": {**counts, "total": len(specs)}, "job_ids": ids, "specs": specs}
    write_once(protocol_path, plan)
    return plan
