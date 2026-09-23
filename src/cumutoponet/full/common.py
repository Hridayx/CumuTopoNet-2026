from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import yaml


class IntegrityError(RuntimeError):
    pass


def atomic_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_once(path, value) -> None:
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise IntegrityError(f"Immutable artifact differs: {path}")
        return
    atomic_json(path, value)


def event(root, kind, details) -> None:
    atomic_json(Path(root) / "events" / f"{time.time_ns()}-{os.getpid()}.json",
                {"time": time.time(), "kind": kind, "details": details})


DEFAULTS = {
    "dataset": {
        "path": "./data/DroneDetect_V2.zip",
        "sample_rate": 60_000_000,
        "window": 1024,
        "stride": 512,
        "caps": {
            "Air2S": 10_000, "Inspire2": 10_000,
            "MavicMini": 5_000, "MavicPro": 5_000, "MavicPro2": 5_000,
            "Phantom4": 15_000, "ParrotDisco": 15_000,
        },
        "background_windows": 50_000,
    },
    "workspace": "./work/full-1024",
    "training": {
        "batch_size": 256, "evaluation_batch_size": 1024,
        "max_epochs": 50, "patience": 10,
        "lr": 9.5060e-4, "weight_decay": 4.7485e-4,
        "label_smoothing": 0.05, "ema_decay": 0.999,
        "temperature": 0.07,
        "gradient_clip": 1.0, "seeds": [17, 42, 91],
        "num_workers": 6, "checkpoint_steps": 500, "amp": True,
    },
    "features": {
        "tda_dim": 3, "tda_delay": 5, "tda_points": 200,
        "pi_size": 20, "pi_sigma": 0.1,
        "pi_birth_max": 1.5, "pi_persistence_max": 1.0,
        "scaler_windows": 10_000, "hoc_clip": 10.0,
    },
    "runtime": {
        "cpus": 8, "feature_workers": 6, "archive_readers": 2,
        "heartbeat_seconds": 60,
    },
    "storage": {"minimum_free_gb": 100},
}


def _merge(base, override):
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path):
    path = Path(path).resolve()
    with path.open() as handle:
        supplied = yaml.safe_load(handle) or {}
    cfg = _merge(DEFAULTS, supplied)
    for keys in [("dataset", "path"), ("workspace",)]:
        value = cfg[keys[0]] if len(keys) == 1 else cfg[keys[0]][keys[1]]
        candidate = Path(os.path.expandvars(str(value))).expanduser()
        resolved = str((path.parent / candidate if not candidate.is_absolute() else candidate).resolve())
        if len(keys) == 1:
            cfg[keys[0]] = resolved
        else:
            cfg[keys[0]][keys[1]] = resolved
    cfg["_config_path"] = str(path)
    if cfg["dataset"]["window"] != 1024 or cfg["dataset"]["stride"] != 512:
        raise ValueError("The full-window workflow requires 1024-sample windows and stride 512.")
    if cfg["runtime"]["archive_readers"] + cfg["runtime"]["feature_workers"] > cfg["runtime"]["cpus"]:
        raise ValueError("Reader and feature worker counts exceed available CPUs.")
    return cfg


def workspace(cfg) -> Path:
    root = Path(cfg["workspace"])
    root.mkdir(parents=True, exist_ok=True)
    return root
