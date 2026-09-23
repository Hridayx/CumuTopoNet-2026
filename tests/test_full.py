import numpy as np
import torch

from cumutoponet.full.dataset import balanced_order, make_inputs
from cumutoponet.full.features import corrected_hoc, legacy_hoc
from cumutoponet import SUPCON_WEIGHT
from cumutoponet.full.models import build_model, model_summary, supcon_loss
from cumutoponet.full.protocol import experiment_specs
from cumutoponet.full.common import DEFAULTS


def test_full_protocol_has_all_predeclared_fits():
    specs = experiment_specs(DEFAULTS)
    assert len(specs) == 35
    assert sum(s["phase"] == "core" for s in specs) == 22
    assert sum(s["phase"] == "legacy" for s in specs) == 2
    assert sum(not s["evaluate_test"] for s in specs) == 11
    assert {s["seed"] for s in specs if s["phase"] == "core"} == {42}


def test_full_models_are_finite_and_parameter_summary_is_computed():
    summary = model_summary()
    assert summary["default"]["trainable"] > 0
    assert summary["compact"]["trainable"] > 0
    batch = {"hoc": torch.randn(4, 6), "temporal": torch.randn(4, 3, 128),
             "tda": torch.rand(4, 2, 20, 20)}
    for name in ("hoc", "temporal", "tda", "hoc_temporal", "hoc_tda",
                 "temporal_tda", "full", "full_ce", "wide_temporal"):
        logits, embedding = build_model(name, 7)(batch)
        assert logits.shape == (4, 7)
        assert torch.isfinite(logits).all() and torch.isfinite(embedding).all()


def test_full_features_and_sampler_are_deterministic():
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(8, 1024)) + 1j * rng.normal(size=(8, 1024))
    for extractor in (corrected_hoc, legacy_hoc):
        value = extractor(raw)
        assert value.shape == (8, 6) and np.isfinite(value).all()
    labels = np.array([0] * 100 + [1] * 10 + [2])
    np.testing.assert_array_equal(balanced_order(labels, 42, 0), balanced_order(labels, 42, 0))


def test_gpu_batch_transform_and_losses_have_gradients():
    batch = {"raw": torch.randn(4, 128, dtype=torch.complex64),
             "hoc": torch.randn(4, 6), "tda": torch.rand(4, 2, 20, 20)}
    inputs = make_inputs(batch, torch.device("cpu"), "full")
    assert inputs["temporal"].shape == (4, 3, 128)
    model = build_model("full", 7)
    logits, embedding = model(inputs)
    labels = torch.tensor([0, 0, 1, 1])
    loss = torch.nn.functional.cross_entropy(logits, labels) + SUPCON_WEIGHT * supcon_loss(embedding, labels)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
