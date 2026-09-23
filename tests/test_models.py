import numpy as np
import pytest
import torch
from cumutoponet.matched.models import build_model, parameter_count, capacity_width, supcon_loss, BRANCHES

torch.set_num_threads(1)


@pytest.mark.parametrize('length', [128, 1024])
@pytest.mark.parametrize('name', list(BRANCHES)+['raw_iq', 'lstm'])
def test_models_forward_backward_at_both_lengths(name, length):
    torch.manual_seed(17)
    m = build_model(name)
    inputs = {'iq': torch.randn(2, 2, length), 'temporal': torch.randn(2, 3, length),
              'hoc': torch.randn(2, 6), 'tda': torch.randn(2, 2, 20, 20)}
    logits, embedding = m(inputs)
    assert logits.shape == (2, 7)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 6])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    m.eval()
    assert torch.equal(m(inputs)[0], m(inputs)[0])


def test_lstm_single_bias_and_capacity_control():
    m = build_model('lstm')
    counts = parameter_count(m)
    assert counts['trainable'] == sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert counts['total'] == sum(p.numel() for p in m.parameters())
    assert 0 < counts['trainable'] < counts['total']
    assert not m.lstm.bias_hh_l0.requires_grad
    assert not m.lstm.bias_hh_l0.any()
    width, audit = capacity_width()
    assert width > 1
    assert audit['relative_error'] < .04


def test_supcon_no_positives_and_extreme_inputs_have_finite_gradients():
    for labels in [torch.arange(4), torch.tensor([1, 1, 2, 2]), torch.zeros(4, dtype=torch.long)]:
        z = (torch.randn(4, 182)*1e5).requires_grad_()
        loss = supcon_loss(z, labels)
        assert torch.isfinite(loss)
        if len(torch.unique(labels)) == 4:
            assert loss.item() == 0
        loss.backward()
        assert torch.isfinite(z.grad).all()
