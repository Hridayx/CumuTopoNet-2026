from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from cumutoponet.matched.queue import JobQueue

from .common import atomic_json, workspace
from .dataset import ShardedDataset


def report(cfg):
    root = workspace(cfg)
    rows = []
    for path in sorted((root / "runs").glob("*/result.json")) if (root / "runs").exists() else []:
        value = json.loads(path.read_text())
        test = value.get("test") or {}
        rows.append({"job_id": path.parent.name, "phase": value["spec"]["phase"],
                     "profile": value["spec"]["profile"], "task": value["spec"]["task"],
                     "model": value["spec"]["model"], "recipe": value["spec"].get("recipe", ""),
                     "seed": value["spec"]["seed"], "validation_loss": value["best_validation_loss"],
                     "accuracy": test.get("accuracy"), "balanced_accuracy": test.get("balanced_accuracy"),
                     "macro_f1": test.get("macro_f1"), "epochs": value["epochs"]})
    directory = root / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    columns = ["job_id", "phase", "profile", "task", "model", "recipe", "seed",
               "validation_loss", "accuracy", "balanced_accuracy", "macro_f1", "epochs"]
    with (directory / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)

    ensemble = None
    legacy = [row for row in rows if row["phase"] == "legacy" and row["accuracy"] is not None]
    if len(legacy) == 2:
        probabilities = [np.load(root / "runs" / row["job_id"] / "test_probabilities.npy") for row in legacy]
        dataset = ShardedDataset(cfg, "legacy", "synthetic9", "test", "full")
        labels = dataset.labels.astype(np.int64)
        prediction = np.mean(probabilities, axis=0).argmax(axis=1)
        ensemble = {"members": [row["job_id"] for row in legacy],
                    "accuracy": float(accuracy_score(labels, prediction)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
                    "macro_f1": float(f1_score(labels, prediction, average="macro"))}
    counts = {}
    for state in JobQueue(root).status().values():
        counts[state["state"]] = counts.get(state["state"], 0) + 1
    summary = {"completed_results": len(rows), "queue_counts": counts,
               "legacy_ensemble": ensemble}
    atomic_json(directory / "summary.json", summary)
    return summary
