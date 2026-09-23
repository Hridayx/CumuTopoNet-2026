from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


BRANCHES = {
    "hoc": ("hoc",), "temporal": ("temporal",), "tda": ("tda",),
    "hoc_temporal": ("hoc", "temporal"), "hoc_tda": ("hoc", "tda"),
    "temporal_tda": ("temporal", "tda"), "full": ("hoc", "temporal", "tda"),
    "full_ce": ("hoc", "temporal", "tda"),
}


class SE1d(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(),
                                 nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x.mean(dim=-1)).unsqueeze(-1)


class SE2d(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(),
                                 nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)


class DSBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=.1):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation,
                      groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
            SE1d(channels, 4),
        )

    def forward(self, x):
        return F.gelu(x + self.path(x))


class TemporalEncoder(nn.Module):
    def __init__(self, inputs=3, channels=144, dropout=.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(inputs, channels, 1, bias=False),
                                  nn.BatchNorm1d(channels), nn.GELU())
        self.blocks = nn.Sequential(*[DSBlock(channels, d, dropout) for d in [1, 2, 4, 8, 16]])

    def forward(self, x):
        return self.blocks(self.stem(x)).amax(dim=-1)


class HOCEncoder(nn.Module):
    def __init__(self, dropout=.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 88), nn.BatchNorm1d(88), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(88, 44), nn.BatchNorm1d(44), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TDAEncoder(nn.Module):
    def __init__(self, channels=88):
        super().__init__()
        out = 2 * channels
        self.block1 = nn.Sequential(nn.Conv2d(2, channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(channels), nn.GELU())
        self.block2 = nn.Sequential(nn.Conv2d(channels, out, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(out), nn.GELU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(out, out, 3, padding=1, groups=out, bias=False),
            nn.Conv2d(out, out, 1, bias=False), nn.BatchNorm2d(out), nn.GELU(),
            nn.MaxPool2d(2), SE2d(out, 4),
        )
        self.project = nn.Sequential(nn.Linear(2 * out, out), nn.BatchNorm1d(out), nn.GELU())

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        return self.project(torch.cat([x.mean(dim=(-2, -1)), x.amax(dim=(-2, -1))], dim=1))


class CumuTopoNet(nn.Module):
    def __init__(self, branches=("hoc", "temporal", "tda"), classes=9,
                 temporal_channels=144, tda_channels=88, fusion_dropout=.45,
                 temporal_inputs=3):
        super().__init__()
        self.branches = tuple(branches)
        encoders, sizes = {}, []
        for branch in self.branches:
            if branch == "hoc":
                encoders[branch], size = HOCEncoder(), 44
            elif branch == "temporal":
                encoders[branch], size = TemporalEncoder(temporal_inputs, temporal_channels), temporal_channels
            elif branch == "tda":
                encoders[branch], size = TDAEncoder(tda_channels), 2 * tda_channels
            else:
                raise ValueError(branch)
            sizes.append(size)
        self.encoders = nn.ModuleDict(encoders)
        joined = sum(sizes)
        embedding = joined // 2 if self.branches == ("hoc", "temporal", "tda") else 182
        self.embedding = nn.Sequential(nn.LayerNorm(joined), nn.Dropout(fusion_dropout),
                                       nn.Linear(joined, embedding), nn.GELU())
        self.classifier = nn.Linear(embedding, classes)

    def forward(self, inputs):
        values = [self.encoders[b](inputs[b]) for b in self.branches]
        embedding = self.embedding(torch.cat(values, dim=1))
        return self.classifier(embedding), embedding


class LSTM256(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.lstm = nn.LSTM(2, 256, batch_first=True)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        self.lstm.bias_hh_l0.requires_grad_(False)
        self.dense = nn.Linear(256, 128)
        self.output = nn.Linear(128, classes)

    def forward(self, inputs):
        _, (hidden, _) = self.lstm(inputs["raw"].transpose(1, 2).contiguous())
        embedding = F.relu(self.dense(hidden[-1]))
        return self.output(embedding), embedding


class SpectrogramCNN(nn.Module):
    """Conventional log-power STFT image baseline."""

    def __init__(self, classes):
        super().__init__()
        def block(inputs, outputs):
            return nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                nn.BatchNorm2d(outputs), nn.GELU(), nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Dropout(.3),
        )
        self.classifier = nn.Linear(128, classes)

    def forward(self, inputs):
        value = self.pool(self.features(inputs["spectrogram"])).flatten(1)
        embedding = self.embedding(value)
        return self.classifier(embedding), embedding


def parameter_count(model):
    return {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters())}


def _wide_channels(classes):
    target = parameter_count(CumuTopoNet(classes=classes))["trainable"]
    candidates = []
    for channels in range(144, 512, 8):
        model = CumuTopoNet(("temporal",), classes, temporal_channels=channels)
        count = parameter_count(model)["trainable"]
        candidates.append((abs(count - target), channels, count))
    _, channels, actual = min(candidates)
    return channels, target, actual


def build_model(name, classes):
    if name == "lstm":
        return LSTM256(classes)
    if name == "spectrogram_cnn":
        return SpectrogramCNN(classes)
    if name == "raw_iq":
        return CumuTopoNet(("temporal",), classes, temporal_inputs=2)
    if name == "wide_temporal":
        channels, _, _ = _wide_channels(classes)
        return CumuTopoNet(("temporal",), classes, temporal_channels=channels)
    key = "full_ce" if name == "full_ce" else name
    if key not in BRANCHES:
        raise ValueError(f"Unknown model: {name}")
    return CumuTopoNet(BRANCHES[key], classes)


def model_summary():
    default = CumuTopoNet(classes=9)
    compact = CumuTopoNet(classes=9, temporal_channels=88, tda_channels=64)
    return {
        "default": parameter_count(default),
        "compact": parameter_count(compact),
    }


def supcon_loss(embedding, labels, temperature=.07):
    z = F.normalize(embedding.float(), dim=1)
    n = len(z)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    valid = positive.any(dim=1)
    if not valid.any():
        return z.sum() * 0
    logits = z @ z.T / temperature
    denominator = torch.logsumexp(logits.masked_fill(eye, -torch.inf), dim=1)
    log_probability = logits - denominator[:, None]
    sums = log_probability.masked_fill(~positive, 0).sum(dim=1)
    return -(sums[valid] / positive.sum(dim=1)[valid]).mean()
