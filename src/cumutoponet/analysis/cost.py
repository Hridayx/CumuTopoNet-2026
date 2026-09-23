"""Measure feature construction and neural inference cost for every model view."""
from __future__ import annotations

import time

import numpy as np
import torch

from cumutoponet.full.common import atomic_json, workspace
from cumutoponet.full.dataset import ShardedDataset, make_inputs
from cumutoponet.full.features import corrected_hoc, temporal, topology
from cumutoponet.full.models import build_model, parameter_count


MODELS = ("hoc", "temporal", "tda", "hoc_temporal", "temporal_tda", "full",
          "raw_iq", "wide_temporal", "lstm", "spectrogram_cnn")


def statistics(seconds):
    values = np.asarray(seconds) * 1000
    return {"mean_ms": float(values.mean()), "p50_ms": float(np.quantile(values, .5)),
            "p95_ms": float(np.quantile(values, .95))}


def run(cfg, device="cuda:0", batch_size=256, repeats=20):
    root = workspace(cfg)
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Production inference benchmark requires an allocated CUDA GPU.")
    base = ShardedDataset(cfg, "corrected", "drone7", "train", "full")
    indices = np.arange(batch_size)
    raw, hoc, tda = [], [], []
    for index in indices:
        item = base[int(index)]
        raw.append(item["raw"].numpy())
        hoc.append(item["hoc"].numpy())
        tda.append(item["tda"].numpy())
    batch = {"raw": torch.from_numpy(np.stack(raw)), "hoc": torch.from_numpy(np.stack(hoc)),
             "tda": torch.from_numpy(np.stack(tda)),
             "label": torch.zeros(batch_size, dtype=torch.long)}

    feature = {}
    waveform = np.asarray(raw[0])
    for name, operation, repeats in (
        ("hoc", lambda: corrected_hoc(waveform[None]), 100),
        ("temporal", lambda: temporal(waveform[None]), 100),
        ("tda", lambda: topology(waveform, cfg["features"], legacy=False), 10),
        ("spectrogram", lambda: make_inputs({"raw": batch["raw"][:1]}, torch.device("cpu"),
                                             "spectrogram_cnn"), 100),
    ):
        values = []
        for _ in range(repeats):
            started = time.perf_counter(); operation(); values.append(time.perf_counter() - started)
        feature[name] = {**statistics(values), "repeats": repeats, "batch": 1, "device": "CPU"}

    inference = {}
    torch.set_grad_enabled(False)
    for name in MODELS:
        model = build_model(name, 7).to(device).eval()
        inputs = make_inputs(batch, device, name)
        for _ in range(5):
            model(inputs)
        torch.cuda.synchronize(device)
        times = []
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeats):
            started = time.perf_counter(); model(inputs); torch.cuda.synchronize(device)
            times.append(time.perf_counter() - started)
        elapsed = float(np.mean(times))
        inference[name] = {
            **statistics(times), "batch": batch_size,
            "throughput_windows_s": float(batch_size / elapsed),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "parameters": parameter_count(model),
        }
        del model, inputs
        torch.cuda.empty_cache()
    result = {
        "status": "completed", "window": 1024, "batch_size": batch_size,
        "feature_extraction": feature, "network_inference": inference,
        "scope": "resident cached waveform/view through view construction or network; excludes RF acquisition and disk I/O",
        "gpu": torch.cuda.get_device_name(device),
    }
    output = root / "analyses" / "cost"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "result.json", result)
    return result
