from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from cumutoponet.matched.data import inventory as base_inventory, read_selected

from .common import IntegrityError, atomic_json, read_json, workspace, write_once
from .features import corrected_hoc, legacy_hoc, legacy_scaler_hoc, topology


LABELS = ["Air2S", "Inspire2", "MavicMini", "MavicPro", "MavicPro2", "Phantom4", "ParrotDisco"]
BACKGROUND = ["background_awgn", "background_pink_gaussian"]
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}
SPLIT_NAME = np.array(["train", "validation", "test"])


TDA_VARIANTS = {
    "base": {}, "dim2": {"tda_dim": 2}, "dim4": {"tda_dim": 4},
    "delay2": {"tda_delay": 2}, "delay10": {"tda_delay": 10},
    "points100": {"tda_points": 100}, "points300": {"tda_points": 300},
    "sigma05": {"pi_sigma": .05}, "sigma20": {"pi_sigma": .2},
    "size16": {"pi_size": 16}, "size32": {"pi_size": 32},
}


def _save_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        np.save(temporary, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_npy_files(directory, names):
    directory = Path(directory)
    for name in names:
        path = directory / name
        if not path.is_file():
            raise IntegrityError(f"Missing cache file: {path}")
        try:
            np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise IntegrityError(f"Unreadable cache file: {path}") from error


def experiment_inventory(cfg):
    base = base_inventory(cfg)
    counts = {}
    cells = {}
    for record in base["records"]:
        counts[record["label_name"]] = counts.get(record["label_name"], 0) + 1
        key = (record["label_name"], record["mode"], record["condition"])
        cells.setdefault(key, []).append(record)
    if len(base["records"]) != 390 or any(len(records) != 5 for records in cells.values()):
        raise IntegrityError("Expected the verified 390-recording dataset with five files per observed condition cell.")
    if len(cells) != 78:
        raise IntegrityError(f"Expected 78 observed label/mode/interference cells, found {len(cells)}.")
    base["class_counts"] = counts
    base["cell_count"] = len(cells)
    return base


def corrected_record_splits(records):
    cells = {}
    for record in records:
        cells.setdefault((record["label_name"], record["mode"], record["condition"]), []).append(record)
    rng = np.random.default_rng(42)
    result = {}
    for key in sorted(cells):
        values = sorted(cells[key], key=lambda row: row["path"])
        order = rng.permutation(len(values))
        result[values[int(order[0])]["id"]] = "test"
        result[values[int(order[1])]["id"]] = "validation"
        for position in order[2:]:
            result[values[int(position)]["id"]] = "train"
    return result


def legacy_record_splits(records):
    # Preserve the legacy class and path ordering used by this workflow.
    ordered = sorted(records, key=lambda row: (row["label"], row["path"]))
    indices = np.arange(len(ordered))
    train, held = train_test_split(indices, test_size=.2,
                                   stratify=[row["label"] for row in ordered], random_state=42)
    return ({ordered[int(i)]["id"]: "train" for i in train},
            [ordered[int(i)] for i in held])


def _legacy_scaler(cfg, root, records):
    """Use the legacy global RandomState sequence for the scaler fit."""
    ordered = sorted(records, key=lambda row: (row["label"], row["path"]))
    positions = np.arange(len(ordered))
    train_position, _ = train_test_split(
        positions, test_size=.2, stratify=[row["label"] for row in ordered], random_state=42)
    train_records = [ordered[int(index)] for index in train_position]
    caps = cfg["dataset"]["caps"]
    record_start, cursor = {}, 0
    for label in range(7):
        for record in [row for row in ordered if row["label"] == label]:
            record_start[record["id"]] = cursor
            cursor += caps[record["label_name"]]
    pieces = [np.arange(record_start[row["id"]],
                        record_start[row["id"]] + caps[row["label_name"]], dtype=np.int64)
              for row in train_records]
    rng = np.random.RandomState(42)
    for _ in BACKGROUND:
        order = rng.permutation(cfg["dataset"]["background_windows"])
        count = int(.8 * len(order))
        pieces.append(cursor + order[:count])
        cursor += len(order)
    train_global = np.concatenate(pieces)
    rng.shuffle(train_global)
    selected = np.sort(rng.choice(train_global, cfg["features"]["scaler_windows"], replace=False))
    values = []
    for record in ordered:
        start = record_start[record["id"]]
        end = start + caps[record["label_name"]]
        local = selected[(selected >= start) & (selected < end)] - start
        if len(local):
            array = np.load(Path(root) / "cache" / "records" / record["id"] / "raw.npy",
                            mmap_mode="r", allow_pickle=False)
            values.append(legacy_scaler_hoc(array[local]))
    background_start = sum(caps[row["label_name"]] for row in ordered)
    for kind in BACKGROUND:
        end = background_start + cfg["dataset"]["background_windows"]
        local = selected[(selected >= background_start) & (selected < end)] - background_start
        if len(local):
            array = np.load(Path(root) / "cache" / "records" / kind / "raw.npy",
                            mmap_mode="r", allow_pickle=False)
            values.append(legacy_scaler_hoc(array[local]))
        background_start = end
    matrix = np.concatenate(values)
    if len(matrix) != len(selected):
        raise IntegrityError("Scaler index assembly failed.")
    scaler = RobustScaler().fit(matrix)
    scale = np.where(np.abs(scaler.scale_) < 1e-8, 1., scaler.scale_)
    write_once(Path(root) / "scalers" / "legacy-synthetic9.json",
               {"profile": "legacy", "task": "synthetic9", "count": len(matrix),
                "center": scaler.center_.tolist(), "scale": scale.tolist(),
                "selection_seed": 42,
                "selection": "legacy RandomState/global-index sequence"})


def _selected_offsets(record, cap):
    total = (int(record["samples"]) - 1024) // 512 + 1
    candidate = np.linspace(0, total - 1, int(cap), dtype=np.int64)
    offsets = candidate * 512
    if np.any(np.diff(offsets) < 1024):
        raise IntegrityError(f"Retained windows overlap in {record['path']}.")
    return offsets


def _free_space_gate(root, minimum_gb):
    free = shutil.disk_usage(root).free
    if free < float(minimum_gb) * 1e9:
        raise IntegrityError(f"Free storage {free/1e9:.1f} GB is below the {minimum_gb} GB reserve.")


def _prepare_record(args):
    source, root, record, cap, feature_cfg, minimum_free, archive_gate = args
    root = Path(root)
    directory = root / "cache" / "records" / record["id"]
    marker = directory / "meta.json"
    if marker.exists():
        metadata = read_json(marker)
        if metadata.get("record") != record or metadata.get("cap") != int(cap):
            raise IntegrityError(f"Cache metadata differs for {record['id']}.")
        _validate_npy_files(directory, metadata["files"])
        return metadata
    _free_space_gate(root, minimum_free)
    offsets = _selected_offsets(record, cap)
    with archive_gate:
        raw = read_selected(source, record, offsets, 1024)
    hoc_corrected = corrected_hoc(raw)
    hoc_legacy = legacy_hoc(raw)
    size = int(feature_cfg["pi_size"])
    tda_corrected = np.empty((len(raw), 2, size, size), dtype=np.float16)
    tda_legacy = np.empty_like(tda_corrected)
    for index, window in enumerate(raw):
        tda_corrected[index] = topology(window, feature_cfg, legacy=False).astype(np.float16)
        tda_legacy[index] = topology(window, feature_cfg, legacy=True).astype(np.float16)
    directory.mkdir(parents=True, exist_ok=True)
    values = {
        "raw.npy": raw.astype(np.complex64), "offsets.npy": offsets,
        "hoc_corrected.npy": hoc_corrected, "hoc_legacy.npy": hoc_legacy,
        "tda_corrected.npy": tda_corrected, "tda_legacy.npy": tda_legacy,
    }
    for name, value in values.items():
        _save_npy(directory / name, value)
    metadata = {
        "record": record, "cap": int(cap),
        "minimum_retained_spacing": int(np.diff(offsets).min()),
        "files": list(values),
    }
    atomic_json(marker, metadata)
    return metadata


def _background_waveforms(kind, count):
    seed = 42 if kind == "background_awgn" else 43
    rng = np.random.RandomState(seed)
    if kind == "background_awgn":
        i = rng.randn(count, 1024).astype(np.float32)
        q = rng.randn(count, 1024).astype(np.float32)
    else:
        frequencies = np.fft.rfftfreq(1024)
        frequencies[0] = frequencies[1]
        shape = 1 / np.sqrt(frequencies)
        i = np.fft.irfft(rng.randn(count, len(shape)) * shape, n=1024, axis=1).astype(np.float32)
        q = np.fft.irfft(rng.randn(count, len(shape)) * shape, n=1024, axis=1).astype(np.float32)
    value = (i + 1j * q).astype(np.complex64)
    power = np.mean(np.abs(value) ** 2, axis=1, keepdims=True)
    return value / np.sqrt(np.maximum(power, 1e-10))


def _prepare_background(args):
    root, kind, count, feature_cfg, minimum_free = args
    root = Path(root)
    directory = root / "cache" / "records" / kind
    marker = directory / "meta.json"
    if marker.exists():
        metadata = read_json(marker)
        if metadata.get("id") != kind or metadata.get("cap") != count:
            raise IntegrityError(f"Background cache metadata differs for {kind}.")
        _validate_npy_files(directory, metadata["files"])
        return metadata
    _free_space_gate(root, minimum_free)
    raw = _background_waveforms(kind, count)
    size = int(feature_cfg["pi_size"])
    tda_corrected = np.empty((count, 2, size, size), dtype=np.float16)
    tda_legacy = np.empty_like(tda_corrected)
    for index, window in enumerate(raw):
        tda_corrected[index] = topology(window, feature_cfg, legacy=False).astype(np.float16)
        tda_legacy[index] = topology(window, feature_cfg, legacy=True).astype(np.float16)
    directory.mkdir(parents=True, exist_ok=True)
    values = {"raw.npy": raw.astype(np.complex64),
              "hoc_corrected.npy": corrected_hoc(raw), "hoc_legacy.npy": legacy_hoc(raw),
              "tda_corrected.npy": tda_corrected, "tda_legacy.npy": tda_legacy}
    for name, value in values.items():
        _save_npy(directory / name, value)
    metadata = {"id": kind, "label_name": kind, "cap": count,
                "files": list(values)}
    atomic_json(marker, metadata)
    return metadata


def _legacy_local_splits(root, records, held_records, caps, background_count):
    root = Path(root)
    held_ids = {record["id"] for record in held_records}
    rng = np.random.default_rng(42)
    for label in range(7):
        selected = [record for record in held_records if record["label"] == label and record["id"] in held_ids]
        references = [(record["id"], local) for record in selected for local in range(caps[record["label_name"]])]
        order = rng.permutation(len(references))
        validation = set(int(i) for i in order[:len(order) // 2])
        by_record = {record["id"]: np.full(caps[record["label_name"]], SPLIT_CODE["test"], dtype=np.uint8)
                     for record in selected}
        for flat, (record_id, local) in enumerate(references):
            if flat in validation:
                by_record[record_id][local] = SPLIT_CODE["validation"]
        for record_id, values in by_record.items():
            _save_npy(root / "cache" / "records" / record_id / "legacy_split.npy", values)
    legacy_background_rng = np.random.RandomState(42)
    for kind in BACKGROUND:
        count = background_count
        initial = legacy_background_rng.permutation(count)
        train_n = int(.8 * count)
        values = np.full(count, SPLIT_CODE["train"], dtype=np.uint8)
        held = initial[train_n:]
        order = rng.permutation(len(held))
        half = len(held) // 2
        values[held[order[:half]]] = SPLIT_CODE["validation"]
        values[held[order[half:]]] = SPLIT_CODE["test"]
        _save_npy(root / "cache" / "records" / kind / "legacy_split.npy", values)


def _fit_scaler(root, entries, profile, task, feature_name, split_getter, sample_count):
    candidates = []
    for entry in entries:
        split = split_getter(entry)
        local = np.flatnonzero(split == SPLIT_CODE["train"])
        if len(local):
            candidates.append((entry, local))
    totals = np.cumsum([len(local) for _, local in candidates])
    rng = np.random.default_rng(42)
    chosen = np.sort(rng.choice(int(totals[-1]), min(sample_count, int(totals[-1])), replace=False))
    values = []
    active_entry = None
    array = None
    for global_index in chosen:
        shard = int(np.searchsorted(totals, global_index, side="right"))
        previous = 0 if shard == 0 else int(totals[shard - 1])
        entry, local = candidates[shard]
        if active_entry != entry["id"]:
            array = np.load(Path(root) / "cache" / "records" / entry["id"] / feature_name,
                            mmap_mode="r", allow_pickle=False)
            active_entry = entry["id"]
        values.append(np.asarray(array[local[int(global_index) - previous]], dtype=np.float32))
    scaler = RobustScaler().fit(np.stack(values))
    scale = np.where(np.abs(scaler.scale_) < 1e-8, 1., scaler.scale_)
    result = {"profile": profile, "task": task, "count": len(values),
              "center": scaler.center_.tolist(), "scale": scale.tolist(),
              "selection_seed": 42}
    write_once(Path(root) / "scalers" / f"{profile}-{task}.json", result)


def prepare(cfg):
    root = workspace(cfg)
    started = time.time()
    inv = experiment_inventory(cfg)
    records = inv["records"]
    caps = cfg["dataset"]["caps"]
    corrected = corrected_record_splits(records)
    legacy_train, held = legacy_record_splits(records)
    plan_values = {"inventory": inv, "caps": caps, "corrected_record_splits": corrected,
                   "legacy_train_records": legacy_train,
                   "legacy_held_record_ids": [r["id"] for r in held],
                   "features": cfg["features"]}
    plan_path = root / "data_plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if {key: plan[key] for key in plan_values} != plan_values:
            raise IntegrityError("Prepared data settings differ; use an empty workspace.")
    else:
        plan = {**plan_values, "plan_id": uuid.uuid4().hex}
        atomic_json(plan_path, plan)
    # Each process owns one recording from ZIP read through atomic cache commit.
    # A cross-process semaphore limits expensive concurrent ZIP streams without
    # limiting the number of records doing independent feature extraction.
    with multiprocessing.Manager() as manager:
        archive_gate = manager.BoundedSemaphore(cfg["runtime"]["archive_readers"])
        args = [(inv["source"], str(root), record, caps[record["label_name"]], cfg["features"],
                 cfg["storage"]["minimum_free_gb"], archive_gate) for record in records]
        with ProcessPoolExecutor(cfg["runtime"]["feature_workers"]) as workers:
            for _ in workers.map(_prepare_record, args, chunksize=1):
                pass
    background_args = [(str(root), kind, cfg["dataset"]["background_windows"], cfg["features"],
                        cfg["storage"]["minimum_free_gb"]) for kind in BACKGROUND]
    with ProcessPoolExecutor(2) as workers:
        for result in workers.map(_prepare_background, background_args):
            result
    _legacy_local_splits(root, records, held, caps, cfg["dataset"]["background_windows"])
    entries = [{"id": record["id"], "label": record["label"], "kind": "drone"} for record in records]
    entries9 = entries + [{"id": kind, "label": 7 + index, "kind": "background"}
                          for index, kind in enumerate(BACKGROUND)]
    records_by_id = {record["id"]: record for record in records}
    def corrected_values(entry):
        count = cfg["dataset"]["background_windows"] if entry["kind"] == "background" else caps[records_by_id[entry["id"]]["label_name"]]
        if entry["kind"] == "background":
            values = np.full(count, SPLIT_CODE["train"], dtype=np.uint8)
            rng = np.random.default_rng(42 + entry["label"])
            order = rng.permutation(count)
            values[order[:count // 5]] = SPLIT_CODE["test"]
            values[order[count // 5:2 * count // 5]] = SPLIT_CODE["validation"]
            return values
        return np.full(count, SPLIT_CODE[corrected[entry["id"]]], dtype=np.uint8)
    for task, task_entries in [("drone7", entries), ("synthetic9", entries9)]:
        _fit_scaler(root, task_entries, "corrected", task, "hoc_corrected.npy", corrected_values,
                    cfg["features"]["scaler_windows"])
    _legacy_scaler(cfg, root, records)
    manifest = {"status": "completed", "data_plan_id": plan["plan_id"], "records": len(records),
                "drone_windows": sum(caps[r["label_name"]] for r in records),
                "background_windows": 2 * cfg["dataset"]["background_windows"],
                "elapsed_seconds": time.time() - started, "completed_at": time.time()}
    write_once(root / "preparation_complete.json", manifest)
    return manifest


def entry_manifest(cfg, profile, task, sensitivity=None):
    root = workspace(cfg)
    plan = read_json(root / "data_plan.json")
    records = plan["inventory"]["records"]
    caps = plan["caps"]
    corrected = plan["corrected_record_splits"]
    held = set(plan["legacy_held_record_ids"])
    entries = []
    for record in records:
        count = caps[record["label_name"]]
        if sensitivity:
            count = min(250, count)
        if profile == "corrected":
            split = np.full(count, SPLIT_CODE[corrected[record["id"]]], dtype=np.uint8)
        elif record["id"] in held:
            original = np.load(root / "cache" / "records" / record["id"] / "legacy_split.npy", allow_pickle=False)
            positions = np.linspace(0, len(original) - 1, count, dtype=int) if sensitivity else np.arange(count)
            split = original[positions]
        else:
            split = np.zeros(count, dtype=np.uint8)
        entries.append({"id": record["id"], "label": record["label"], "label_name": record["label_name"],
                        "mode": record["mode"], "condition": record["condition"], "count": count,
                        "split": split, "kind": "drone", "record": record})
    if task == "synthetic9":
        for index, kind in enumerate(BACKGROUND):
            count = cfg["dataset"]["background_windows"]
            if profile == "legacy":
                split = np.load(root / "cache" / "records" / kind / "legacy_split.npy", allow_pickle=False)
            else:
                split = np.zeros(count, dtype=np.uint8)
                rng = np.random.default_rng(49 + index)
                order = rng.permutation(count)
                split[order[:count // 5]] = SPLIT_CODE["test"]
                split[order[count // 5:2 * count // 5]] = SPLIT_CODE["validation"]
            entries.append({"id": kind, "label": 7 + index, "label_name": kind,
                            "mode": "Synthetic", "condition": "Synthetic", "count": count,
                            "split": split, "kind": "background", "record": None})
    return entries


def prepare_sensitivity(cfg):
    root = workspace(cfg)
    if not (root / "preparation_complete.json").exists():
        raise IntegrityError("Base preparation is incomplete.")
    entries = entry_manifest(cfg, "corrected", "drone7", sensitivity=True)
    jobs = []
    for variant, changes in TDA_VARIANTS.items():
        spec = {**cfg["features"], **changes}
        for entry in entries:
            jobs.append((str(root), entry["id"], variant, spec, entry["count"]))
    with ProcessPoolExecutor(cfg["runtime"]["feature_workers"]) as pool:
        for _ in pool.map(_prepare_sensitivity_record, jobs, chunksize=1):
            pass
    marker = {"status": "completed", "variants": TDA_VARIANTS,
              "windows": sum(e["count"] for e in entries), "completed_at": time.time()}
    write_once(root / "sensitivity_complete.json", marker)
    return marker


def _prepare_sensitivity_record(args):
    root, record_id, variant, spec, count = args
    directory = Path(root) / "cache" / "sensitivity" / variant / record_id
    marker = directory / "meta.json"
    if marker.exists():
        result = read_json(marker)
        if result.get("variant") != variant or result.get("record_id") != record_id or result.get("spec") != spec:
            raise IntegrityError(f"Sensitivity cache metadata differs for {variant}/{record_id}.")
        _validate_npy_files(directory, result["files"])
        return result
    raw = np.load(Path(root) / "cache" / "records" / record_id / "raw.npy", mmap_mode="r", allow_pickle=False)
    positions = np.linspace(0, len(raw) - 1, count, dtype=int)
    values = np.stack([topology(raw[int(i)], spec, legacy=False) for i in positions]).astype(np.float16)
    directory.mkdir(parents=True, exist_ok=True)
    _save_npy(directory / "tda.npy", values)
    result = {"variant": variant, "record_id": record_id, "count": count,
              "spec": spec, "files": ["tda.npy"]}
    atomic_json(marker, result)
    return result
