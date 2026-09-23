"""Model definitions for matched-budget experiments."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

BRANCHES = {'hoc': ('hoc',), 'temporal': ('temporal',), 'tda': ('tda',),
            'hoc_temporal': ('hoc', 'temporal'), 'hoc_tda': ('hoc', 'tda'),
            'temporal_tda': ('temporal', 'tda'), 'full': ('hoc', 'temporal', 'tda')}


class SeparableResidual(nn.Module):
    def __init__(self, n_in, n_out, dilation):
        super().__init__()
        padding = 3*dilation
        groups = math.gcd(8, n_out)
        self.path = nn.Sequential(
            nn.Conv1d(n_in, n_in, 7, padding=padding, dilation=dilation, groups=n_in, bias=False),
            nn.Conv1d(n_in, n_out, 1, bias=False), nn.GroupNorm(groups, n_out), nn.SiLU(),
            nn.Conv1d(n_out, n_out, 7, padding=padding, dilation=dilation, groups=n_out, bias=False),
            nn.Conv1d(n_out, n_out, 1, bias=False), nn.GroupNorm(groups, n_out))
        self.skip = nn.Identity() if n_in == n_out else nn.Conv1d(n_in, n_out, 1, bias=False)

    def forward(self, x):
        return F.silu(self.path(x) + self.skip(x))


class TemporalEncoder(nn.Module):
    def __init__(self, n_in=3, width=1.):
        super().__init__()
        channels = [max(8, int(round(c*width/8))*8) for c in [32, 48, 64, 96, 144]]
        layers = []
        for c, dilation in zip(channels, [1, 2, 4, 8, 16]):
            layers.append(SeparableResidual(n_in, c, dilation))
            n_in = c
        self.blocks = nn.Sequential(*layers)
        self.project = nn.Identity() if n_in == 144 else nn.Linear(n_in, 144)

    def forward(self, x):
        return self.project(self.blocks(x).mean(dim=-1))


class TopologyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 88, 3, padding=1, bias=False), nn.GroupNorm(8, 88), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(88, 176, 3, padding=1, bias=False), nn.GroupNorm(8, 176), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(176, 176, 3, padding=1, groups=176, bias=False),
            nn.Conv2d(176, 176, 1, bias=False), nn.GroupNorm(8, 176), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class Fusion(nn.Module):
    def __init__(self, branches, width=1., raw_iq=False):
        super().__init__()
        self.branches, self.raw_iq = tuple(branches), raw_iq
        modules, sizes = {}, []
        for name in branches:
            if name == 'hoc':
                modules[name] = nn.Sequential(nn.Linear(6, 88), nn.LayerNorm(88), nn.SiLU(),
                                              nn.Linear(88, 44), nn.LayerNorm(44), nn.SiLU())
                sizes.append(44)
            elif name == 'temporal':
                modules[name] = TemporalEncoder(2 if raw_iq else 3, width)
                sizes.append(144)
            elif name == 'tda':
                modules[name] = TopologyEncoder()
                sizes.append(176)
        self.encoders = nn.ModuleDict(modules)
        self.embedding = nn.Sequential(nn.Linear(sum(sizes), 182), nn.LayerNorm(182), nn.SiLU())
        self.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(182, 7))

    def forward(self, inputs):
        values = []
        for name in self.branches:
            key = 'iq' if name == 'temporal' and self.raw_iq else name
            values.append(self.encoders[name](inputs[key]))
        embedding = self.embedding(torch.cat(values, dim=-1))
        return self.classifier(embedding), embedding


class LSTM256(nn.Module):
    """One 256-unit LSTM, Dense128/ReLU and 7 logits. Single trainable bias."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 256, num_layers=1, batch_first=True)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        self.lstm.bias_hh_l0.requires_grad_(False)
        self.dense = nn.Linear(256, 128)
        self.output = nn.Linear(128, 7)

    def forward(self, inputs):
        _, (h, _) = self.lstm(inputs['iq'].transpose(1, 2).contiguous())
        embedding = F.relu(self.dense(h[-1]))
        return self.output(embedding), embedding


def build_model(name, width=1.):
    if name == 'lstm':
        return LSTM256()
    if name == 'raw_iq':
        return Fusion(('temporal',), raw_iq=True)
    if name == 'wide_temporal':
        return Fusion(('temporal',), width=width)
    if name == 'full_supcon':
        name = 'full'
    if name not in BRANCHES:
        raise ValueError(f'Unknown model {name}')
    return Fusion(BRANCHES[name])


def parameter_count(model):
    return {'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total': sum(p.numel() for p in model.parameters())}


def capacity_width():
    """Nearest width on a predeclared 0.01 grid, before seeing any performance."""
    # fork_rng prevents architecture accounting from consuming the training RNG.
    with torch.random.fork_rng():
        target = parameter_count(build_model('full'))['trainable']
        low, high = 100, 1000
        while low < high:
            mid = (low+high)//2
            count = parameter_count(build_model('wide_temporal', mid/100))['trainable']
            if count < target:
                low = mid+1
            else:
                high = mid
        candidates = [(abs(parameter_count(build_model('wide_temporal', i/100))['trainable']-target), i)
                      for i in range(max(100, low-3), low+4)]
        _, selected = min(candidates)
        actual = parameter_count(build_model('wide_temporal', selected/100))['trainable']
    return selected/100, {'target_parameters': target, 'actual_parameters': actual,
                         'relative_error': abs(actual-target)/target,
                         'method': 'nearest 0.01 width; channels rounded to multiples of 8'}


def supcon_loss(embedding, labels, temperature=.07):
    """Single-view supervised contrastive loss, excluding self comparisons.

    Anchors without a second example of their class contribute no term.
    Reductions are float32 even inside mixed-precision training.
    """
    if temperature <= 0:
        raise ValueError('Temperature must be positive.')
    z = F.normalize(embedding.float(), dim=-1)
    n = len(z)
    if n < 2:
        return z.sum()*0.
    self_mask = torch.eye(n, device=z.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~self_mask
    valid = positives.any(dim=1)
    if not valid.any():
        return z.sum()*0.
    logits = (z @ z.T)/temperature
    denominator = torch.logsumexp(logits.masked_fill(self_mask, -torch.inf), dim=1)
    log_prob = logits - denominator[:, None]
    sums = log_prob.masked_fill(~positives, 0.).sum(dim=1)
    return -(sums[valid]/positives.sum(dim=1)[valid]).mean()
