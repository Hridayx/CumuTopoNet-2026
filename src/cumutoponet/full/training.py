from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import signal
import tempfile
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import IntegrityError, atomic_json, read_json, workspace
from .dataset import FixedSampler, ShardedDataset, balanced_order, make_inputs
from .models import build_model, parameter_count, supcon_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def _restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda"]])


@torch.no_grad()
def _ema_update(model, ema, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.mul_(decay).add_(source, alpha=1 - decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def _loader(dataset, sampler, batch_size, workers, drop_last=False):
    generator = torch.Generator()
    generator.manual_seed(0)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=workers, pin_memory=True, drop_last=drop_last,
                      persistent_workers=False, generator=generator)


def _optimizer_step(optimizer, scaler, parameters, max_norm):
    parameters = [value for value in parameters if value.requires_grad]
    scaler.unscale_(optimizer)
    gradients = [value.grad for value in parameters if value.grad is not None]
    if not gradients:
        raise FloatingPointError("No gradients were produced.")
    finite = bool(torch.stack([torch.isfinite(value).all() for value in gradients]).all())
    if not finite:
        if not scaler.is_enabled():
            raise FloatingPointError("Nonfinite gradients with AMP disabled.")
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)  # GradScaler skips this update after unscale_ found Inf/NaN.
        scaler.update()
        if float(scaler.get_scale()) >= old_scale:
            raise FloatingPointError("AMP overflow did not reduce its loss scale.")
        return False
    torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    scaler.step(optimizer); scaler.update()
    return True


@torch.inference_mode()
def predict(model, dataset, device, model_name, batch_size, workers, heartbeat=None):
    model.eval()
    probabilities, labels = [], []
    loader = _loader(dataset, FixedSampler(np.arange(len(dataset), dtype=np.uint32)),
                     batch_size, workers)
    for index, batch in enumerate(loader):
        logits, _ = model(make_inputs(batch, device, model_name))
        value = torch.softmax(logits.float(), dim=1).cpu().numpy()
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite evaluation probabilities.")
        probabilities.append(value)
        labels.append(batch["label"].numpy())
        if heartbeat and index % 100 == 0:
            heartbeat()
    classes = model.classifier.out_features if hasattr(model, "classifier") else model.output.out_features
    return (np.concatenate(probabilities) if probabilities else np.empty((0, classes), np.float32),
            np.concatenate(labels) if labels else np.empty(0, np.uint8))


def _metrics(labels, probabilities, loss=None):
    predicted = probabilities.argmax(axis=1)
    classes = probabilities.shape[1]
    return {"loss": None if loss is None else float(loss),
            "accuracy": float(accuracy_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, average="macro")),
            "confusion_matrix": confusion_matrix(labels, predicted, labels=np.arange(classes)).tolist()}


def _validation_loss(probabilities, labels, weights=None, smoothing=.05):
    logits = torch.from_numpy(np.log(np.maximum(probabilities, 1e-30)))
    target = torch.from_numpy(labels.astype(np.int64))
    weight = None if weights is None else torch.tensor(weights, dtype=torch.float32)
    return float(F.cross_entropy(logits, target, weight=weight, label_smoothing=smoothing))


def _save_observations(output, name, dataset, labels):
    """Save a compact, auditable mapping from predictions to source recordings."""
    np.savez(
        output / f"{name}_observations.npz",
        entry=dataset.entry_index.astype(np.uint16, copy=False),
        local=dataset.local_index.astype(np.uint32, copy=False),
        labels=np.asarray(labels, dtype=np.uint8),
    )
    atomic_json(
        output / f"{name}_entries.json",
        [
            {
                "index": index, "id": entry["id"], "label": int(entry["label"]),
                "label_name": entry["label_name"], "mode": entry["mode"],
                "condition": entry["condition"], "count": int(entry["count"]),
            }
            for index, entry in enumerate(dataset.entries)
        ],
    )


def train_run(cfg, spec, run_id, device="cuda:0", heartbeat=None):
    root = workspace(cfg)
    output = root / "runs" / run_id
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result.get("spec") != spec:
            raise IntegrityError("Completed run does not match the queued specification.")
        if not (output / "best.pt").is_file():
            raise IntegrityError("Completed run is missing its checkpoint.")
        torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        return result
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise IntegrityError("Production runs require a CUDA GPU.")

    profile, task = spec["profile"], spec["task"]
    model_name = spec["model"]
    sensitivity = spec.get("sensitivity")
    heldout_condition = spec.get("heldout_condition")
    classes = 7 if task == "drone7" else 9
    train = ShardedDataset(cfg, profile, task, "train", model_name, sensitivity,
                           heldout_condition)
    validation = ShardedDataset(cfg, profile, task, "validation", model_name, sensitivity,
                                heldout_condition)
    test = ShardedDataset(cfg, profile, task, "test", model_name, sensitivity,
                          heldout_condition) if spec.get("evaluate_test", True) else None
    for name, data in [("train", train), ("validation", validation)]:
        if set(np.unique(data.labels)) != set(range(classes)):
            raise IntegrityError(f"{name} split lacks complete {classes}-class support.")

    training = {**cfg["training"], **spec.get("training", {})}
    seed_everything(spec["seed"])
    torch_device = torch.device(device)
    model = build_model(model_name, classes).to(torch_device)
    ema = copy.deepcopy(model).to(torch_device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"],
                                  weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["max_epochs"], eta_min=training["lr"] / 20)
    amp = bool(training["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    manual_weights = None
    if spec.get("manual_class_weights"):
        manual_weights = np.asarray(spec["manual_class_weights"], dtype=np.float32)
        if len(manual_weights) != classes:
            raise IntegrityError("Manual class-weight vector does not match task classes.")
    class_weights = None if manual_weights is None else torch.tensor(manual_weights, device=torch_device)

    epoch = position = global_step = stale = 0
    best_loss, order, history = float("inf"), None, []
    train_loss_sum = train_examples = 0
    started = time.monotonic()
    stop = {"requested": False, "signal": None}
    previous = {}
    def stop_handler(signum, _frame):
        stop.update(requested=True, signal=signum)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        previous[sig] = signal.signal(sig, stop_handler)

    def snapshot():
        return {"spec": spec, "model": model.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "rng": _rng_state(), "epoch": epoch,
                "position": position, "global_step": global_step, "stale": stale,
                "best_loss": best_loss, "order": order, "history": history,
                "train_loss_sum": train_loss_sum, "train_examples": train_examples,
                "elapsed_seconds": time.monotonic() - started}

    last_path = output / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=torch_device, weights_only=False)
        if state.get("spec") != spec:
            raise IntegrityError("Checkpoint specification changed; resume refused.")
        model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"]); _restore_rng(state["rng"])
        epoch, position, global_step, stale = state["epoch"], state["position"], state["global_step"], state["stale"]
        best_loss, order, history = state["best_loss"], state["order"], state["history"]
        train_loss_sum, train_examples = state["train_loss_sum"], state["train_examples"]

    try:
        while epoch < training["max_epochs"] and stale < training["patience"]:
            if order is None:
                order = balanced_order(train.labels, spec["seed"], epoch, spec.get("sampler_power", 1.0))
            model.train()
            loader = _loader(train, FixedSampler(order[position:]), training["batch_size"],
                             training["num_workers"], drop_last=True)
            for batch in loader:
                if stop["requested"]:
                    _atomic_torch(last_path, snapshot())
                    return {"status": "interrupted", "signal": stop["signal"]}
                labels = batch["label"].to(torch_device, non_blocking=True)
                retries = 0
                while True:
                    before_batch = _rng_state()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                        logits, embedding = model(make_inputs(batch, torch_device, model_name))
                        ce = F.cross_entropy(logits, labels, weight=class_weights,
                                             label_smoothing=training["label_smoothing"])
                        weight = float(spec.get("supcon_weight", 0))
                        contrast = supcon_loss(embedding, labels, training["temperature"]) if weight else ce.new_zeros(())
                        loss = ce + weight * contrast
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss.")
                    scaler.scale(loss).backward()
                    if _optimizer_step(optimizer, scaler, model.parameters(), training["gradient_clip"]):
                        break
                    retries += 1
                    if retries > 8:
                        raise FloatingPointError("AMP overflow persisted across eight identical-batch retries.")
                    _restore_rng(before_batch)
                _ema_update(model, ema, training["ema_decay"])
                batch_count = len(labels)
                train_loss_sum += float(loss.detach()) * batch_count
                train_examples += batch_count
                position += batch_count
                global_step += 1
                if heartbeat and global_step % 100 == 0:
                    heartbeat()
                if global_step % training["checkpoint_steps"] == 0:
                    _atomic_torch(last_path, snapshot())

            probabilities, val_labels = predict(
                ema, validation, torch_device, model_name,
                training["evaluation_batch_size"], training["num_workers"], heartbeat)
            val_loss = _validation_loss(probabilities, val_labels, manual_weights,
                                        training["label_smoothing"])
            val_metrics = _metrics(val_labels, probabilities, val_loss)
            history.append({"epoch": epoch + 1, "train_loss": train_loss_sum / max(train_examples, 1),
                            "validation": val_metrics, "lr": optimizer.param_groups[0]["lr"]})
            improved = val_loss < best_loss
            best_loss, stale = (val_loss, 0) if improved else (best_loss, stale + 1)
            scheduler.step(); epoch += 1
            position, order, train_loss_sum, train_examples = 0, None, 0, 0
            if improved:
                _atomic_torch(output / "best.pt", snapshot())
            _atomic_torch(last_path, snapshot())
            atomic_json(output / "progress.json", {"epoch": epoch, "global_step": global_step,
                        "best_validation_loss": best_loss, "stale_epochs": stale,
                        "latest_validation": val_metrics, "updated": time.time()})
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    best = torch.load(output / "best.pt", map_location=torch_device, weights_only=False)
    if best.get("spec") != spec:
        raise IntegrityError("Best checkpoint does not match the queued specification.")
    ema.load_state_dict(best["ema"])
    val_probability, val_labels = predict(ema, validation, torch_device, model_name,
                                           training["evaluation_batch_size"], training["num_workers"], heartbeat)
    validation_metrics = _metrics(val_labels, val_probability,
                                  _validation_loss(val_probability, val_labels, manual_weights,
                                                   training["label_smoothing"]))
    np.save(output / "validation_probabilities.npy", val_probability)
    np.save(output / "validation_labels.npy", val_labels.astype(np.uint8))
    _save_observations(output, "validation", validation, val_labels)
    test_metrics = None
    if test is not None:
        test_probability, test_labels = predict(ema, test, torch_device, model_name,
                                                training["evaluation_batch_size"], training["num_workers"], heartbeat)
        test_metrics = _metrics(test_labels, test_probability)
        np.save(output / "test_probabilities.npy", test_probability)
        np.save(output / "test_labels.npy", test_labels.astype(np.uint8))
        _save_observations(output, "test", test, test_labels)
    result = {"status": "completed", "spec": spec,
              "parameters": parameter_count(model),
              "data_plan_id": read_json(root / "data_plan.json")["plan_id"],
              "partition_indices": {
                  "train": str(train.index_path), "validation": str(validation.index_path),
                  "test": None if test is None else str(test.index_path),
              },
              "best_validation_loss": best_loss, "validation": validation_metrics,
              "test": test_metrics, "history": history, "epochs": epoch,
              "global_steps": global_step, "elapsed_seconds": time.monotonic() - started,
              "files": ["best.pt", "validation_probabilities.npy", "validation_labels.npy"]}
    atomic_json(result_path, result)
    return result
