from __future__ import annotations

import numpy as np

from .common import IntegrityError


HOC_NAMES = ["abs_C20", "abs_C40", "abs_C41", "C42", "abs_C60", "C63"]


def normalize(x):
    z = np.asarray(x, dtype=np.complex128)
    z = z - z.mean(axis=-1, keepdims=True)
    power = np.mean(np.abs(z) ** 2, axis=-1, keepdims=True)
    if not np.isfinite(power).all():
        raise IntegrityError("Nonfinite waveform power.")
    return np.divide(z, np.sqrt(np.maximum(power, 1e-24)), out=np.zeros_like(z), where=power > 1e-24)


def corrected_hoc(x):
    z = normalize(x)
    m = lambda p, q: np.mean(z ** (p - q) * z.conj() ** q, axis=-1)
    m20, m21 = m(2, 0), m(2, 1).real
    m30, m31 = m(3, 0), m(3, 1)
    m40, m41, m42 = m(4, 0), m(4, 1), m(4, 2).real
    c20 = m20
    c40 = m40 - 3 * m20 ** 2
    c41 = m41 - 3 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2 * m21 ** 2
    c60 = m(6, 0) - 15 * m40 * m20 - 10 * m30 ** 2 + 30 * m20 ** 3
    c63 = (m(6, 3).real - 9 * m42 * m21 - 6 * np.real(m20.conj() * m41)
           - np.abs(m30) ** 2 - 9 * np.abs(m31) ** 2
           + 18 * np.abs(m20) ** 2 * m21 + 12 * m21 ** 3)
    return np.stack([abs(c20), abs(c40), abs(c41), c42, abs(c60), c63], axis=-1).astype(np.float32)


def _legacy_moments(z):
    m20 = np.mean(z ** 2, axis=-1)
    m21 = np.mean(np.abs(z) ** 2, axis=-1)
    notebook_m40 = np.mean(np.abs(z) ** 4, axis=-1)
    m41 = np.mean(z ** 3 * z.conj(), axis=-1)
    notebook_m42 = np.mean(z ** 4, axis=-1)
    m60 = np.mean(z ** 6, axis=-1)
    m63 = np.mean(np.abs(z) ** 6, axis=-1)
    c40 = notebook_m42 - 3 * m20 ** 2
    c41 = m41 - 3 * m20 * m21
    c42 = notebook_m40 - np.abs(m20) ** 2 - 2 * m21 ** 2
    c60 = m60 - 15 * notebook_m42 * m20 + 30 * m20 ** 3
    c63 = m63 - 9 * notebook_m40 * m21 + 12 * m21 ** 3
    return np.stack([abs(m20), abs(c40), abs(c41), c42.real, abs(c60), c63.real], axis=-1).astype(np.float32)


def legacy_hoc(x):
    """The six retained features computed in the v2.5 training preprocessing."""
    return _legacy_moments(normalize(x))


def legacy_scaler_hoc(x):
    """NB03 fitted its saved scaler after RMS scaling but before mean removal."""
    z = np.asarray(x, dtype=np.complex128)
    power = np.mean(np.abs(z) ** 2, axis=-1, keepdims=True)
    z = np.divide(z, np.sqrt(np.maximum(power, 1e-24)), out=np.zeros_like(z), where=power > 1e-24)
    return _legacy_moments(z)


def temporal(x):
    z = normalize(x)
    phase = np.angle(z)
    unwrapped = np.unwrap(phase, axis=-1)
    increment = np.diff(unwrapped, axis=-1, prepend=unwrapped[..., :1])
    return np.stack([np.abs(z), phase / np.pi, increment / np.pi], axis=-2).astype(np.float32)


def raw_iq(x):
    z = normalize(x)
    return np.stack([z.real, z.imag], axis=-2).astype(np.float32)


def takens(amplitude, dim, delay, points):
    count = len(amplitude) - (dim - 1) * delay
    if count < points:
        raise ValueError("Not enough Takens points.")
    cloud = np.stack([amplitude[i * delay:i * delay + count] for i in range(dim)], axis=1)
    low, high = cloud.min(), cloud.max()
    if high <= low:
        return np.zeros((points, dim), dtype=np.float32)
    cloud = (cloud - low) / (high - low)
    return cloud[np.linspace(0, count - 1, points, dtype=int)].astype(np.float32)


def persistence_image(diagram, cfg):
    size, sigma = int(cfg["pi_size"]), float(cfg["pi_sigma"])
    result = np.zeros((size, size), dtype=np.float32)
    if diagram.size == 0:
        return result
    finite = diagram[np.isfinite(diagram[:, 1])]
    if not len(finite):
        return result
    birth = finite[:, 0]
    persistence = finite[:, 1] - finite[:, 0]
    keep = persistence > 0
    birth, persistence = birth[keep], persistence[keep]
    if not len(persistence):
        return result
    weights = persistence / persistence.max()
    gx = np.linspace(0, float(cfg["pi_birth_max"]), size)
    gy = np.linspace(0, float(cfg["pi_persistence_max"]), size)
    x_grid, y_grid = np.meshgrid(gx, gy)
    distance = ((x_grid[..., None] - birth) ** 2 +
                (y_grid[..., None] - persistence) ** 2)
    return (np.exp(-distance / (2 * sigma ** 2)) * weights).sum(axis=-1).astype(np.float32)


def topology(window, cfg, legacy=False):
    from ripser import ripser
    z = np.asarray(window) if legacy else normalize(window)
    cloud = takens(np.abs(z), int(cfg["tda_dim"]), int(cfg["tda_delay"]), int(cfg["tda_points"]))
    if not np.any(cloud):
        return np.zeros((2, int(cfg["pi_size"]), int(cfg["pi_size"])), dtype=np.float32)
    diagrams = ripser(cloud, maxdim=1)["dgms"]
    return np.stack([persistence_image(diagrams[0], cfg), persistence_image(diagrams[1], cfg)])


def scale_hoc(values, scaler, clip):
    center = np.asarray(scaler["center"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    return np.clip((values - center) / scale, -clip, clip).astype(np.float32)
