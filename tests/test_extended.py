import json

import numpy as np
import torch

from cumutoponet.full.common import DEFAULTS
from cumutoponet.full.dataset import build_index, make_inputs
from cumutoponet.full.models import build_model
from cumutoponet.full.extended import HELDOUT_CONDITIONS, extended_specs


def test_extended_suite_is_complete_and_local():
    specs = extended_specs(DEFAULTS)
    assert len(specs) == 25
    pilots = [s for s in specs if s["phase"] == "extended-lstm-pilot"]
    assert {s["training"]["lr"] for s in pilots} == {.0003, .001}
    assert all(not s["training"]["amp"] and not s["evaluate_test"] for s in pilots)
    assert all(s["task"] == "drone7" for s in specs)
    assert all("assignment" not in key and "lane" not in key
               for spec in specs for key in spec)
    heldout = [s for s in specs if s["phase"] == "extended-heldout"]
    assert {s["heldout_condition"] for s in heldout} == set(HELDOUT_CONDITIONS)
    assert len(heldout) == 12


def test_spectrogram_transform_and_model_are_finite():
    torch.manual_seed(18)
    batch = {"raw": torch.randn(4, 1024, dtype=torch.complex64)}
    inputs = make_inputs(batch, torch.device("cpu"), "spectrogram_cnn")
    assert inputs["spectrogram"].shape == (4, 1, 128, 29)
    assert torch.isfinite(inputs["spectrogram"]).all()
    model = build_model("spectrogram_cnn", 7)
    logits, embedding = model(inputs)
    assert logits.shape == (4, 7) and embedding.shape == (4, 128)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_heldout_index_uses_no_heldout_training_or_validation_records(tmp_path):
    labels = ["Air2S", "Inspire2", "MavicMini", "MavicPro", "MavicPro2", "Phantom4", "ParrotDisco"]
    records = []
    split = {}
    for label, name in enumerate(labels):
        for condition in ("BT", "Clean", "WiFi"):
            for partition in ("train", "validation"):
                record_id = f"{label}-{condition}-{partition}"
                records.append({"id": record_id, "label": label, "label_name": name,
                                "mode": "Flying", "condition": condition})
                split[record_id] = partition
    plan = {
        "inventory": {"records": records}, "caps": {name: 2 for name in labels},
        "corrected_record_splits": split, "legacy_held_record_ids": [], "plan_id": "fixture",
    }
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "data_plan.json").write_text(json.dumps(plan))
    cfg = {**DEFAULTS, "workspace": str(tmp_path),
           "dataset": {**DEFAULTS["dataset"], "caps": plan["caps"]}}
    values = {}
    for partition in ("train", "validation", "test"):
        path = build_index(cfg, "corrected", "drone7", partition,
                           heldout_condition="BT")
        with np.load(path, allow_pickle=False) as index:
            values[partition] = set(index["entry"].tolist())
            assert set(index["labels"].tolist()) == set(range(7))
    assert not values["train"] & values["validation"]
    assert not values["train"] & values["test"]
    assert not values["validation"] & values["test"]
    for entry_index in values["train"] | values["validation"]:
        assert records[entry_index]["condition"] != "BT"
    assert all(records[index]["condition"] == "BT" for index in values["test"])
