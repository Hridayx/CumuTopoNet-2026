"""Frozen extended 1024-sample experiment suite."""
from __future__ import annotations

import math
import time
import uuid

import numpy as np
from sklearn.preprocessing import RobustScaler

from cumutoponet import SUPCON_WEIGHT
from cumutoponet.matched.queue import JobQueue

from .common import IntegrityError, read_json, workspace, write_once
from .data import LABELS, SPLIT_CODE, entry_manifest


HELDOUT_CONDITIONS = ("BT", "WiFi", "BTandWiFi")
EXTENDED_PHASES = {
    "extended-lstm-pilot", "extended-lstm-final", "extended-standard-seed",
    "extended-spectrogram-standard", "extended-heldout",
}
MANIFEST_NAME = "extended_manifest.json"
SELECTION_NAME = "extended_lstm_selection.json"


def _training(cfg, **changes):
    return {**cfg["training"], **changes}


def _spec(cfg, phase, model, seed, *, supcon_weight, evaluate_test=True,
          heldout_condition=None, training=None):
    value = {
        "phase": phase,
        "profile": "corrected",
        "task": "drone7",
        "model": model,
        "seed": int(seed),
        "sampler_power": 1.0,
        "supcon_weight": float(supcon_weight),
        "evaluate_test": bool(evaluate_test),
        "training": dict(training or cfg["training"]),
        "protocol": "extended-1024-v1",
    }
    if heldout_condition is not None:
        if heldout_condition not in HELDOUT_CONDITIONS:
            raise IntegrityError(f"Unsupported held-out condition: {heldout_condition}")
        value["heldout_condition"] = heldout_condition
    return value


def _priority(spec):
    if spec["phase"] == "extended-lstm-pilot":
        return 1
    if spec["phase"] == "extended-lstm-final":
        return 2
    if spec["phase"] == "extended-standard-seed":
        return 10
    if spec["phase"] == "extended-heldout":
        return 20
    return 30


def extended_specs(cfg):
    specs = []
    for lr in (.0003, .001):
        spec = _spec(
            cfg, "extended-lstm-pilot", "lstm", 17, supcon_weight=0,
            evaluate_test=False,
            training=_training(cfg, amp=False, lr=lr, max_epochs=10, patience=10),
        )
        spec["selection_group"] = "lstm-fp32-lr"
        specs.append(spec)
    for condition in HELDOUT_CONDITIONS:
        for model in ("temporal", "hoc_temporal"):
            specs.append(_spec(
                cfg, "extended-heldout", model, 42, supcon_weight=SUPCON_WEIGHT,
                heldout_condition=condition,
            ))
        specs.append(_spec(
            cfg, "extended-heldout", "hoc", 42, supcon_weight=SUPCON_WEIGHT,
            heldout_condition=condition,
        ))
        specs.append(_spec(
            cfg, "extended-heldout", "spectrogram_cnn", 42, supcon_weight=0,
            heldout_condition=condition,
        ))
    for model, weight in (("temporal", SUPCON_WEIGHT), ("hoc_temporal", SUPCON_WEIGHT),
                          ("full", SUPCON_WEIGHT), ("raw_iq", 0)):
        specs.append(_spec(
            cfg, "extended-standard-seed", model, 17, supcon_weight=weight,
        ))
        specs.append(_spec(
            cfg, "extended-standard-seed", model, 91, supcon_weight=weight,
        ))
    specs.append(_spec(
        cfg, "extended-spectrogram-standard", "spectrogram_cnn", 42,
        supcon_weight=0,
    ))
    for seed in (17, 91):
        specs.append(_spec(
            cfg, "extended-standard-seed", "spectrogram_cnn", seed,
            supcon_weight=0,
        ))
    return specs


def _fit_heldout_scaler(cfg, condition):
    root = workspace(cfg)
    target = root / "scalers" / f"corrected-drone7-heldout-{condition}.json"
    if target.exists():
        return read_json(target)
    entries = entry_manifest(cfg, "corrected", "drone7")
    candidates = []
    for entry in entries:
        if entry["condition"] == condition:
            continue
        local = np.flatnonzero(entry["split"] == SPLIT_CODE["train"])
        if len(local):
            candidates.append((entry, local))
    totals = np.cumsum([len(local) for _, local in candidates])
    if not len(totals):
        raise IntegrityError(f"No scaler training observations remain for {condition}.")
    seed = {"BT": 1801, "WiFi": 1802, "BTandWiFi": 1803}[condition]
    rng = np.random.default_rng(seed)
    count = min(int(cfg["features"]["scaler_windows"]), int(totals[-1]))
    chosen = np.sort(rng.choice(int(totals[-1]), count, replace=False))
    values = []
    arrays = {}
    for global_index in chosen:
        shard = int(np.searchsorted(totals, global_index, side="right"))
        previous = 0 if shard == 0 else int(totals[shard - 1])
        entry, local = candidates[shard]
        if entry["id"] not in arrays:
            arrays[entry["id"]] = np.load(
                root / "cache" / "records" / entry["id"] / "hoc_corrected.npy",
                mmap_mode="r", allow_pickle=False,
            )
        values.append(np.asarray(arrays[entry["id"]][local[int(global_index) - previous]],
                                 dtype=np.float32))
    scaler = RobustScaler().fit(np.stack(values))
    scale = np.where(np.abs(scaler.scale_) < 1e-8, 1., scaler.scale_)
    result = {
        "profile": "corrected", "task": "drone7",
        "heldout_condition": condition, "count": count,
        "center": scaler.center_.tolist(), "scale": scale.tolist(),
        "selection_seed": seed,
        "policy": "fit only on original training records from seen conditions",
    }
    write_once(target, result)
    return result


def _heldout_audit(cfg):
    entries = entry_manifest(cfg, "corrected", "drone7")
    observed_cells = {(entry["label_name"], entry["mode"], entry["condition"])
                      for entry in entries}
    possible_cells = {(label, mode, condition) for label in LABELS
                      for mode in ("Flying", "Hovering", "SwitchedOn")
                      for condition in ("Clean",) + HELDOUT_CONDITIONS}
    result = {"absent_cells": [list(cell) for cell in sorted(possible_cells - observed_cells)],
              "conditions": {}}
    for condition in HELDOUT_CONDITIONS:
        split_ids = {name: set() for name in ("train", "validation", "test")}
        windows = {name: [0] * len(LABELS) for name in split_ids}
        for entry in entries:
            if entry["condition"] == condition:
                split_ids["test"].add(entry["id"])
                windows["test"][entry["label"]] += int(entry["count"])
                continue
            for name in ("train", "validation"):
                count = int(np.count_nonzero(entry["split"] == SPLIT_CODE[name]))
                if count:
                    split_ids[name].add(entry["id"])
                    windows[name][entry["label"]] += count
        if any(split_ids[a] & split_ids[b] for a, b in
               (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise IntegrityError(f"Held-out recording overlap for {condition}.")
        if any(value == 0 for values in windows.values() for value in values):
            raise IntegrityError(f"Held-out protocol lacks class support for {condition}.")
        result["conditions"][condition] = {
            "record_counts": {name: len(ids) for name, ids in split_ids.items()},
            "window_counts_by_class": {name: dict(zip(LABELS, values))
                                       for name, values in windows.items()},
            "record_ids": {name: sorted(ids) for name, ids in split_ids.items()},
        }
    if len(result["absent_cells"]) != 6:
        raise IntegrityError(f"Expected six absent cells, found {len(result['absent_cells'])}.")
    return result


def _manifest(cfg, jobs):
    root = workspace(cfg)
    data_plan = read_json(root / "data_plan.json")
    audit = _heldout_audit(cfg)
    value = {
        "status": "frozen", "created_at": time.time(),
        "manifest_id": uuid.uuid4().hex,
        "protocol": "extended-1024-v1",
        "data_plan_id": data_plan["plan_id"],
        "scientific_policy": {
            "window": 1024, "stride": 512, "task": "drone7",
            "selection": "validation loss only; test once after selection",
            "heldout_conditions": list(HELDOUT_CONDITIONS),
            "heldout_clean": False,
            "lstm_lr_candidates": [.0003, .001],
            "full_supcon_weight": SUPCON_WEIGHT,
        },
        "heldout_audit": audit,
        "jobs": jobs,
        "lstm_final_template": {
            "seed": 42, "evaluate_test": True, "amp": False, "max_epochs": 50,
            "patience": 10, "selection": "minimum finite pilot validation loss; tie -> 0.0003",
        },
    }
    return value


def commit_extended(cfg):
    root = workspace(cfg)
    if not (root / "preparation_complete.json").exists():
        raise IntegrityError("Prepared full-window cache is unavailable.")
    for condition in HELDOUT_CONDITIONS:
        _fit_heldout_scaler(cfg, condition)
    manifest_path = root / MANIFEST_NAME
    queue = JobQueue(root)
    specs = extended_specs(cfg)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (manifest.get("data_plan_id") != read_json(root / "data_plan.json")["plan_id"]
                or [row["spec"] for row in manifest.get("jobs", [])] != specs):
            raise IntegrityError("Existing extended manifest uses different settings.")
    else:
        rows = []
        for spec in specs:
            priority = _priority(spec)
            rows.append({"job_id": queue.enqueue(spec, priority),
                         "priority": priority, "spec": spec})
        manifest = _manifest(cfg, rows)
        write_once(manifest_path, manifest)
    jobs = []
    for row in manifest["jobs"]:
        job = queue.enqueue(row["spec"], row["priority"])
        if job != row["job_id"]:
            raise IntegrityError("Frozen extended-suite job ID disagrees with queue ID.")
        jobs.append(job)
    return {"status": "committed", "manifest": str(manifest_path),
            "manifest_id": manifest["manifest_id"], "queued_now": jobs,
            "queued_now_count": len(jobs), "deferred_lstm_final": 1}


def advance_lstm(cfg):
    root = workspace(cfg)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        return {"state": "extended-plan-absent"}
    manifest = read_json(manifest_path)
    pilots = [row for row in manifest["jobs"]
              if row["spec"]["phase"] == "extended-lstm-pilot"]
    rows = []
    terminal = {"completed", "blocked", "budget-skipped"}
    for pilot in pilots:
        directory = root / "queue" / pilot["job_id"]
        state = read_json(directory / "state.json")
        row = {"job_id": pilot["job_id"], "lr": pilot["spec"]["training"]["lr"],
               "state": state["state"]}
        result_path = root / "runs" / pilot["job_id"] / "result.json"
        if result_path.exists():
            result = read_json(result_path)
            loss = result.get("best_validation_loss")
            if isinstance(loss, (int, float)) and math.isfinite(loss):
                row["validation_loss"] = float(loss)
        rows.append(row)
    if not all(row["state"] in terminal for row in rows):
        return {"state": "pilots-running", "pilots": rows}
    finite = [row for row in rows if "validation_loss" in row and row["state"] == "completed"]
    if not finite:
        selection = {"status": "blocked", "reason": "no finite completed LSTM pilot",
                     "pilots": rows}
        write_once(root / SELECTION_NAME, selection)
        return selection
    chosen = min(finite, key=lambda row: (row["validation_loss"], row["lr"]))
    selection = {"status": "selected", "selected_lr": chosen["lr"],
                 "selection_rule": "minimum finite validation loss; lower LR tie-break",
                 "pilots": rows}
    write_once(root / SELECTION_NAME, selection)
    final = _spec(
        cfg, "extended-lstm-final", "lstm", 42, supcon_weight=0,
        evaluate_test=True,
        training=_training(cfg, amp=False, lr=chosen["lr"], max_epochs=50, patience=10),
    )
    final["selection_artifact"] = SELECTION_NAME
    final["parent_pilot_job_ids"] = [row["job_id"] for row in rows]
    job_id = JobQueue(root).enqueue(final, _priority(final))
    return {**selection, "final_job_id": job_id}


def extended_status(cfg):
    root = workspace(cfg)
    manifest = read_json(root / MANIFEST_NAME)
    states = JobQueue(root).status()
    immediate = {}
    for row in manifest["jobs"]:
        state = states[row["job_id"]]["state"]
        immediate[state] = immediate.get(state, 0) + 1
    selection = read_json(root / SELECTION_NAME) if (root / SELECTION_NAME).exists() else None
    return {"manifest_id": manifest["manifest_id"], "immediate": immediate,
            "lstm_selection": selection}
