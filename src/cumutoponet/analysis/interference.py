"""Feature-level OFDM/GFSK interference sweep for the frozen Drone7 split."""
from __future__ import annotations

import csv
import json

import numpy as np

from cumutoponet.full.common import atomic_json, workspace
from cumutoponet.full.dataset import ShardedDataset
from cumutoponet.full.features import corrected_hoc, normalize, temporal


SIR_DB = (-20, -10, 0, 10, 20)


def _ofdm(rng, count, length):
    nfft, cp = 64, 16
    symbols = int(np.ceil(length / (nfft + cp))) + 1
    bits = rng.integers(0, 4, size=(count, symbols, nfft))
    qpsk = np.exp(1j * (np.pi / 4 + bits * np.pi / 2))
    time = np.fft.ifft(qpsk, axis=-1) * np.sqrt(nfft)
    blocks = np.concatenate((time[..., -cp:], time), axis=-1)
    value = blocks.reshape(count, -1)[:, :length]
    offset = rng.uniform(-.08, .08, size=(count, 1))
    phase = rng.uniform(-np.pi, np.pi, size=(count, 1))
    n = np.arange(length)[None, :]
    return value * np.exp(1j * (2 * np.pi * offset * n + phase))


def _gfsk(rng, count, length):
    samples_per_symbol = 8
    symbols = int(np.ceil(length / samples_per_symbol)) + 8
    bits = 2 * rng.integers(0, 2, size=(count, symbols)) - 1
    impulses = np.repeat(bits, samples_per_symbol, axis=1)
    x = np.linspace(-3, 3, 6 * samples_per_symbol + 1)
    gaussian = np.exp(-.5 * x * x)
    gaussian /= gaussian.sum()
    shaped = np.stack([np.convolve(row, gaussian, mode="same") for row in impulses])
    phase = np.cumsum(shaped[:, :length] * (np.pi * .5 / samples_per_symbol), axis=1)
    offset = rng.uniform(-.03, .03, size=(count, 1))
    initial = rng.uniform(-np.pi, np.pi, size=(count, 1))
    n = np.arange(length)[None, :]
    return np.exp(1j * (phase + 2 * np.pi * offset * n + initial))


def _sample_clean_test(cfg, per_class):
    dataset = ShardedDataset(cfg, "corrected", "drone7", "test", "temporal")
    selected = []
    for label in range(7):
        candidates = [index for index in range(len(dataset))
                      if int(dataset.labels[index]) == label
                      and dataset.entries[int(dataset.entry_index[index])]["condition"] == "Clean"]
        if len(candidates) < per_class:
            raise RuntimeError(f"Only {len(candidates)} clean test windows for class {label}.")
        positions = np.linspace(0, len(candidates) - 1, per_class, dtype=int)
        selected.extend(candidates[int(position)] for position in positions)
    waves, labels, records = [], [], []
    for index in selected:
        entry = dataset.entries[int(dataset.entry_index[index])]
        local = int(dataset.local_index[index])
        waves.append(np.asarray(dataset._array(entry, "raw")[local], dtype=np.complex64))
        labels.append(int(dataset.labels[index]))
        records.append(entry["id"])
    return np.stack(waves), np.asarray(labels), np.asarray(records)


def _view_shifts(clean, mixed, hoc_scale, temporal_scale):
    clean_hoc, mixed_hoc = corrected_hoc(clean), corrected_hoc(mixed)
    hoc = np.sqrt(np.mean(((mixed_hoc - clean_hoc) / hoc_scale) ** 2, axis=1))
    clean_temporal, mixed_temporal = temporal(clean), temporal(mixed)
    difference = mixed_temporal - clean_temporal
    difference[:, 1:] = (difference[:, 1:] + 1) % 2 - 1
    temporal_shift = np.sqrt(np.mean((difference / temporal_scale[None, :, None]) ** 2,
                                     axis=(1, 2)))
    return hoc, temporal_shift


def _interval(values, seed):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(500, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), np.quantile(means, [.025, .975]).tolist()


def run(cfg, per_class=128):
    root = workspace(cfg)
    output = root / "analyses" / "controlled_interference"
    output.mkdir(parents=True, exist_ok=True)
    waves, labels, records = _sample_clean_test(cfg, per_class)
    clean_hoc = corrected_hoc(waves)
    clean_temporal = temporal(waves)
    hoc_scale = np.maximum(clean_hoc.std(axis=0), 1e-6)
    temporal_scale = np.maximum(clean_temporal.std(axis=(0, 2)), 1e-6)
    rows = []
    for kind_index, kind in enumerate(("wifi_ofdm", "bluetooth_gfsk")):
        for sir_index, sir in enumerate(SIR_DB):
            seed = 180000 + kind_index * 100 + sir_index
            rng = np.random.default_rng(seed)
            interference = _ofdm(rng, len(waves), waves.shape[1]) if kind == "wifi_ofdm" else _gfsk(
                rng, len(waves), waves.shape[1])
            signal_power = np.mean(np.abs(waves) ** 2, axis=1, keepdims=True)
            interference_power = np.mean(np.abs(interference) ** 2, axis=1, keepdims=True)
            scale = np.sqrt(signal_power / np.maximum(interference_power, 1e-24)
                            / (10 ** (sir / 10)))
            mixed = waves + scale * interference
            hoc, temporal_shift = _view_shifts(waves, mixed, hoc_scale, temporal_scale)
            for view, values in (("hoc", hoc), ("temporal", temporal_shift)):
                mean, ci = _interval(values, seed + (0 if view == "hoc" else 1000))
                rows.append({"interference": kind, "sir_db": sir, "view": view,
                             "mean_standardized_rms_shift": mean,
                             "ci95_low": ci[0], "ci95_high": ci[1],
                             "windows": len(values)})
    csv_path = output / "feature_shift.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), sharey=True)
    colors = {"hoc": "#087E8B", "temporal": "#D1495B"}
    labels_for_plot = {"hoc": "Cumulants", "temporal": "Temporal"}
    for axis, kind, title in zip(axes, ("wifi_ofdm", "bluetooth_gfsk"),
                                 ("Wi-Fi-like OFDM", "Bluetooth-like GFSK")):
        for view in ("hoc", "temporal"):
            selected = [row for row in rows if row["interference"] == kind and row["view"] == view]
            x = np.asarray([row["sir_db"] for row in selected])
            y = np.asarray([row["mean_standardized_rms_shift"] for row in selected])
            low = np.asarray([row["ci95_low"] for row in selected])
            high = np.asarray([row["ci95_high"] for row in selected])
            axis.plot(x, y, marker="o", linewidth=2, color=colors[view], label=labels_for_plot[view])
            axis.fill_between(x, low, high, color=colors[view], alpha=.16, linewidth=0)
        axis.set_title(title, fontsize=10, weight="bold")
        axis.set_xlabel("Signal-to-interference ratio (dB)")
        axis.grid(True, alpha=.22, linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Standardized feature displacement")
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout(pad=.8)
    png_path, pdf_path = output / "feature_shift.png", output / "feature_shift.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    request = {"sir_db": list(SIR_DB), "per_class": per_class,
               "classes": 7, "windows": len(waves), "seed_base": 180000,
               "source_record_ids": sorted(set(records.tolist())),
               "source_label_counts": np.bincount(labels, minlength=7).tolist(),
               "metric": "per-window RMS feature displacement normalized by clean-view dispersion",
               "normalization": "model-matched centering and unit-power waveform normalization",
               "data_plan_id": json.loads((root / "data_plan.json").read_text())["plan_id"]}
    result = {"status": "completed", "request": request, "rows": rows,
              "files": [path.name for path in (csv_path, png_path, pdf_path)]}
    atomic_json(output / "result.json", result)
    return result
