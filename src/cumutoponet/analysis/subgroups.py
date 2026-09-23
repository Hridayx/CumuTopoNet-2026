"""Compute per-condition/per-mode metrics and paired recording bootstraps."""
from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from cumutoponet.matched.metrics import paired_group_interval
from cumutoponet.full.common import atomic_json, workspace
from cumutoponet.full.dataset import ShardedDataset


def metrics(labels, predictions):
    return {
        "support": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", labels=np.arange(7),
                                    zero_division=0)),
    }


def model_label(spec):
    weight = float(spec.get("supcon_weight", 0))
    return f'{spec["model"]}:w{weight:g}'


def run(cfg, replicates=2000):
    root = workspace(cfg)
    dataset = ShardedDataset(cfg, "corrected", "drone7", "test", "temporal")
    labels = dataset.labels.astype(np.int64)
    entry = dataset.entry_index.astype(np.int64)
    records = np.asarray([dataset.entries[index]["id"] for index in entry])
    modes = np.asarray([dataset.entries[index]["mode"] for index in entry])
    conditions = np.asarray([dataset.entries[index]["condition"] for index in entry])
    runs = []
    for result_path in sorted((root / "runs").glob("*/result.json")):
        result = json.loads(result_path.read_text())
        spec = result.get("spec", {})
        probability_path = result_path.parent / "test_probabilities.npy"
        if (spec.get("task") != "drone7" or spec.get("heldout_condition")
                or not probability_path.exists() or result.get("test") is None):
            continue
        probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
        if len(probabilities) != len(labels):
            continue
        runs.append({"job_id": result_path.parent.name, "spec": spec,
                     "prediction": np.asarray(probabilities.argmax(axis=1), dtype=np.uint8)})
    if not runs:
        raise RuntimeError("No compatible standard-split Drone7 predictions were found.")
    rows = []
    for run in runs:
        common = {"job_id": run["job_id"], "model": model_label(run["spec"]),
                  "seed": int(run["spec"]["seed"])}
        rows.append({**common, "group_type": "overall", "group": "all",
                     **metrics(labels, run["prediction"])})
        for group_type, values, declared in (
            ("condition", conditions, ("Clean", "BT", "WiFi", "BTandWiFi")),
            ("mode", modes, ("Flying", "Hovering", "SwitchedOn")),
        ):
            for value in declared:
                mask = values == value
                rows.append({**common, "group_type": group_type, "group": value,
                             **metrics(labels[mask], run["prediction"][mask])})
    output = root / "analyses" / "subgroups"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "subgroup_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    by_key = {(model_label(run["spec"]), int(run["spec"]["seed"])): run for run in runs}
    reference_key = "temporal:w0.5"
    comparisons = {}
    reference_seeds = {seed for model, seed in by_key if model == reference_key}
    for model in sorted({model for model, _ in by_key} - {reference_key}):
        common_seeds = sorted(reference_seeds & {seed for key, seed in by_key if key == model})
        if not common_seeds:
            continue
        candidate = np.stack([by_key[(model, seed)]["prediction"] for seed in common_seeds])
        reference = np.stack([by_key[(reference_key, seed)]["prediction"] for seed in common_seeds])
        comparisons[f"{model}_minus_{reference_key}"] = {
            "candidate": model, "reference": reference_key, "seeds": common_seeds,
            "interval": paired_group_interval(labels, candidate, reference, records,
                                               replicates=replicates, seed=1809),
        }
    result = {
        "status": "completed", "runs": len(runs), "test_windows": len(labels),
        "recordings": int(len(np.unique(records))), "reference": reference_key,
        "files": [csv_path.name], "paired_comparisons": comparisons,
        "interpretation": "recording-cluster bootstrap conditional on the frozen split and available seeds",
    }
    atomic_json(output / "paired_bootstrap.json", result)
    return result
