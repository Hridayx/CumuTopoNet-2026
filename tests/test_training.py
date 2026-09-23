import numpy as np
import pytest
import torch
from cumutoponet.matched.common import DEFAULTS
from cumutoponet.matched.training import train_run, balanced_order, scaled_optimizer_step

torch.set_num_threads(1)


def easy_arrays():
    rng = np.random.default_rng(21)
    labels = np.tile(np.arange(7), 12)
    hoc = np.eye(7)[labels, :6]*8 + rng.normal(0, .03, (len(labels), 6))
    # Class 6 is all zero, still linearly distinguishable with a bias.
    n = len(labels)
    return {'raw': (rng.normal(size=(n, 128)) + 1j*rng.normal(size=(n, 128))).astype('complex64'),
        'hoc': hoc.astype('float32'), 'pi': np.zeros((n, 2, 20, 20), 'float32'), 'labels': labels,
        'ids': np.array([f'window-{i}' for i in range(n)]),
        'splits': np.array(['train']*56 + ['validation']*28),
        'scaler': {'mean': [0.]*6, 'scale': [1.]*6},
        'feature_spec': {'version': 'synthetic'}, 'feature_key': 'synthetic',
        'data_plan_id': 'synthetic'}


def test_balanced_sampling_reproducible_with_rare_classes():
    labels = np.array([0]*1000 + [1]*10 + [2])
    a = balanced_order(labels, seed=42, epoch=0)
    np.testing.assert_array_equal(a, balanced_order(labels, 42, 0))
    assert not np.array_equal(a, balanced_order(labels, 42, 1))
    assert np.min(np.bincount(labels[a])) > 200


def test_tiny_overfit_and_exact_cpu_checkpoint_resume(tmp_path):
    arrays = easy_arrays()
    spec = {'model': 'hoc', 'seed': 17, 'lr': .01, 'width': 1., 'supcon_weight': 0.,
            'training': {**DEFAULTS['training'], 'batch_size': 28, 'max_epochs': 12,
                         'min_epochs': 12, 'checkpoint_steps': 1, 'amp': False},
            'feature_key': 'synthetic', 'data_plan_id': 'synthetic'}
    uninterrupted = train_run(spec, arrays, tmp_path/'full', device='cpu')
    interrupted = train_run(spec, arrays, tmp_path/'resumed', device='cpu', stop_after_steps=3)
    assert interrupted['status'] == 'interrupted'
    resumed = train_run(spec, arrays, tmp_path/'resumed', device='cpu')
    assert resumed['best_validation_macro_f1'] > .97
    assert uninterrupted['history'] == resumed['history']
    a = torch.load(tmp_path/'full'/'last.pt', map_location='cpu', weights_only=False)
    b = torch.load(tmp_path/'resumed'/'last.pt', map_location='cpu', weights_only=False)
    for k in a['model']:
        assert torch.equal(a['model'][k], b['model'][k])


def test_nonfinite_gradients_with_amp_disabled_remain_a_hard_failure():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    parameter.grad = torch.tensor([float('inf')])
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    with pytest.raises(FloatingPointError, match='AMP disabled'):
        scaled_optimizer_step(optimizer, scaler, [parameter])


def test_amp_overflow_requests_a_retry_without_mutating_parameters():
    class RecoveringScaler:
        def __init__(self):
            self.scale = 65536.
            self.stepped = False

        def unscale_(self, _optimizer):
            return None

        def is_enabled(self):
            return True

        def get_scale(self):
            return self.scale

        def step(self, _optimizer):
            self.stepped = True

        def update(self):
            self.scale /= 2

    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    parameter.grad = torch.tensor([float('inf')])
    scaler = RecoveringScaler()
    assert scaled_optimizer_step(optimizer, scaler, [parameter])
    assert scaler.stepped
    assert parameter.item() == 1.
