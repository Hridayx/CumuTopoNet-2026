from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .common import IntegrityError, read_json, workspace
from .data import SPLIT_CODE, entry_manifest
from .features import scale_hoc
from .models import BRANCHES


def required_features(model_name):
    if model_name in {"lstm", "spectrogram_cnn"}:
        return {"raw"}
    if model_name == "raw_iq":
        return {"raw"}
    if model_name == "wide_temporal":
        return {"raw"}
    branches = set(BRANCHES[model_name])
    result = set()
    if "hoc" in branches:
        result.add("hoc")
    if "temporal" in branches:
        result.add("raw")
    if "tda" in branches:
        result.add("tda")
    return result


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_index(cfg, profile, task, split, sensitivity=None, heldout_condition=None):
    root = workspace(cfg)
    suffix = f"-{sensitivity}" if sensitivity else ""
    heldout_suffix = f"-heldout-{heldout_condition}" if heldout_condition else ""
    path = root / "indexes" / f"{profile}-{task}-{split}{suffix}{heldout_suffix}.npz"
    if path.exists():
        return path
    entries = entry_manifest(cfg, profile, task, sensitivity=sensitivity)
    code = SPLIT_CODE[split]
    def selected(entry):
        if heldout_condition is None:
            return np.flatnonzero(entry["split"] == code)
        is_heldout = entry["condition"] == heldout_condition
        if split == "test":
            return np.arange(entry["count"], dtype=np.int64) if is_heldout else np.empty(0, np.int64)
        if is_heldout:
            return np.empty(0, np.int64)
        return np.flatnonzero(entry["split"] == code)
    selections = [selected(entry) for entry in entries]
    counts = [len(values) for values in selections]
    total = sum(counts)
    entry_index = np.empty(total, dtype=np.uint16)
    local_index = np.empty(total, dtype=np.uint32)
    labels = np.empty(total, dtype=np.uint8)
    cursor = 0
    for index, (entry, count, local) in enumerate(zip(entries, counts, selections)):
        entry_index[cursor:cursor + count] = index
        local_index[cursor:cursor + count] = local
        labels[cursor:cursor + count] = entry["label"]
        cursor += count
    _atomic_npz(path, entry=entry_index, local=local_index, labels=labels)
    return path


class ShardedDataset(Dataset):
    """Lazy mmap view over per-record caches; safe in DataLoader subprocesses."""

    def __init__(self, cfg, profile, task, split, model_name, sensitivity=None,
                 heldout_condition=None):
        self.root = workspace(cfg)
        self.profile = profile
        self.task = task
        self.split_name = split
        self.sensitivity = sensitivity
        self.heldout_condition = heldout_condition
        self.entries = entry_manifest(cfg, profile, task, sensitivity=sensitivity)
        self.index_path = build_index(cfg, profile, task, split, sensitivity,
                                      heldout_condition)
        index = np.load(self.index_path, allow_pickle=False)
        self.entry_index = index["entry"]
        self.local_index = index["local"]
        self.labels = index["labels"]
        self.features = required_features(model_name)
        self.hoc_clip = float(cfg["features"]["hoc_clip"])
        scaler_name = f"{profile}-{task}"
        if heldout_condition and "hoc" in self.features:
            scaler_name += f"-heldout-{heldout_condition}"
        self.scaler = read_json(self.root / "scalers" / f"{scaler_name}.json") if "hoc" in self.features else None
        self._arrays = {}

    def __len__(self):
        return len(self.labels)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = {}
        return state

    def _array(self, entry, feature):
        key = (entry["id"], feature)
        if key not in self._arrays:
            if feature == "tda" and self.sensitivity:
                path = self.root / "cache" / "sensitivity" / self.sensitivity / entry["id"] / "tda.npy"
            else:
                names = {
                    "raw": "raw.npy",
                    "hoc": f"hoc_{self.profile}.npy",
                    "tda": f"tda_{self.profile}.npy",
                }
                path = self.root / "cache" / "records" / entry["id"] / names[feature]
            if not path.exists():
                raise IntegrityError(f"Missing prepared feature shard: {path}")
            self._arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._arrays[key]

    def _source_local(self, entry, local):
        if self.sensitivity:
            source_count = len(self._array(entry, "raw"))
            return int(np.linspace(0, source_count - 1, entry["count"], dtype=int)[local])
        return local

    def __getitem__(self, index):
        entry = self.entries[int(self.entry_index[index])]
        local = int(self.local_index[index])
        item = {"label": torch.tensor(int(self.labels[index]), dtype=torch.long)}
        if "raw" in self.features:
            source_local = self._source_local(entry, local)
            raw = np.array(self._array(entry, "raw")[source_local], dtype=np.complex64, copy=True)
            item["raw"] = torch.from_numpy(raw)
        if "hoc" in self.features:
            source_local = self._source_local(entry, local)
            hoc = np.asarray(self._array(entry, "hoc")[source_local], dtype=np.float32)
            item["hoc"] = torch.from_numpy(scale_hoc(hoc, self.scaler, self.hoc_clip).copy())
        if "tda" in self.features:
            tda = np.asarray(self._array(entry, "tda")[local], dtype=np.float32)
            item["tda"] = torch.from_numpy(np.log1p(tda).copy())
        return item

    def observation(self, index):
        entry = self.entries[int(self.entry_index[index])]
        local = int(self.local_index[index])
        source_local = self._source_local(entry, local) if self.sensitivity else local
        return {"entry_id": entry["id"], "local_index": source_local,
                "label": int(self.labels[index]), "label_name": entry["label_name"],
                "mode": entry["mode"], "condition": entry["condition"]}


class FixedSampler(Sampler):
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return (int(value) for value in self.values)

    def __len__(self):
        return len(self.values)


def balanced_order(labels, seed, epoch, power=1.0):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels)
    if np.any(counts == 0):
        raise IntegrityError("Training partition lacks one or more task classes.")
    weights = counts[labels].astype(np.float64) ** (-float(power))
    weights /= weights.sum()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), 1729]))
    return rng.choice(len(labels), len(labels), replace=True, p=weights).astype(np.uint32)


def make_inputs(batch, device, model_name):
    result = {}
    if "hoc" in batch:
        result["hoc"] = batch["hoc"].to(device, non_blocking=True)
    if "tda" in batch:
        result["tda"] = batch["tda"].to(device, non_blocking=True)
    if "raw" in batch:
        raw = batch["raw"].to(device, non_blocking=True)
        centered = raw - raw.mean(dim=1, keepdim=True)
        power = centered.abs().square().mean(dim=1, keepdim=True)
        z = centered / power.clamp_min(1e-24).sqrt()
        iq = torch.stack((z.real, z.imag), dim=1).float()
        if model_name == "lstm":
            result["raw"] = iq
        elif model_name == "spectrogram_cnn":
            window = torch.hann_window(128, periodic=True, device=device, dtype=z.real.dtype)
            spectrum = torch.stft(
                z, n_fft=128, hop_length=32, win_length=128, window=window,
                center=False, onesided=False, return_complex=True,
            )
            log_power = torch.log1p(spectrum.abs().square())
            log_power = torch.fft.fftshift(log_power, dim=1)
            mean = log_power.mean(dim=(1, 2), keepdim=True)
            std = log_power.std(dim=(1, 2), keepdim=True, unbiased=False)
            result["spectrogram"] = ((log_power - mean) / std.clamp_min(1e-6)).unsqueeze(1)
        elif model_name == "raw_iq":
            result["temporal"] = iq
        else:
            phase = torch.angle(z)
            increment = torch.zeros_like(phase)
            increment[:, 1:] = torch.angle(z[:, 1:] * z[:, :-1].conj())
            result["temporal"] = torch.stack((z.abs(), phase / torch.pi,
                                               increment / torch.pi), dim=1).float()
    return result
