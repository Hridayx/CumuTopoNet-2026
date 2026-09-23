"""Central-moment plug-in estimators. Magnitudes have finite-sample bias.

M_pq = E[z**(p-q) conj(z)**q], with centered, unit-power z.
The sixth-order expressions retain noncircular and odd-moment terms.
"""
from __future__ import annotations
import numpy as np
from .common import IntegrityError

FEATURE_VERSION = 'central-full-cumulants-pi-persistence-v1'
HOC_NAMES = ['abs_C20', 'abs_C40', 'abs_C41', 'C42', 'abs_C60', 'C63']


def normalize(x):
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim < 1 or x.shape[-1] < 2 or not np.isfinite(x).all():
        raise IntegrityError('Waveforms must be finite and have at least two samples.')
    z = x - x.mean(axis=-1, keepdims=True)
    power = np.mean(np.abs(z)**2, axis=-1, keepdims=True)
    return np.divide(z, np.sqrt(np.maximum(power, 1e-24)),
                     out=np.zeros_like(z), where=power > 1e-24)


def cumulants(x):
    z = normalize(x)
    m = lambda p, q: np.mean(z**(p-q) * z.conj()**q, axis=-1)
    m20, m21 = m(2, 0), m(2, 1).real
    m30, m31 = m(3, 0), m(3, 1)
    m40, m41, m42 = m(4, 0), m(4, 1), m(4, 2).real
    return {
        'C20': m20,
        'C40': m40 - 3*m20**2,
        'C41': m41 - 3*m20*m21,
        'C42': m42 - np.abs(m20)**2 - 2*m21**2,
        'C60': m(6, 0) - 15*m40*m20 - 10*m30**2 + 30*m20**3,
        'C63': (m(6, 3).real - 9*m42*m21 - 6*np.real(m20.conj()*m41)
                - np.abs(m30)**2 - 9*np.abs(m31)**2
                + 18*np.abs(m20)**2*m21 + 12*m21**3),
    }


def hoc_features(x):
    c = cumulants(x)
    return np.stack([abs(c['C20']), abs(c['C40']), abs(c['C41']),
                     c['C42'], abs(c['C60']), c['C63']], axis=-1)


def temporal_features(x):
    """Channels: amplitude, principal phase/pi, unwrapped phase increment/pi.

    First increment is exactly zero. This is not a Hilbert transform.
    """
    z = normalize(x)
    phase = np.angle(z)
    unwrapped = np.unwrap(phase, axis=-1)
    increments = np.diff(unwrapped, axis=-1, prepend=unwrapped[..., :1])
    return np.stack([np.abs(z), phase/np.pi, increments/np.pi], axis=-2).astype(np.float32)


def feature_spec(cfg, length=None, points=None, channels=None):
    spec = dict(cfg['features'])
    if length is not None:
        spec['length'] = int(length)
    if points is not None:
        spec['tda_points'] = int(points)
    if channels is not None:
        spec['tda_channels'] = list(channels)
    spec.update(version=FEATURE_VERSION,
                normalization='center-unit-power-complex128', hoc_names=HOC_NAMES,
                pi_bounds=[0., float(np.sqrt(3.))], pi_weight='persistence',
                pi_sampling='density-at-bin-centers', amplitude_scaling='per-window-minmax')
    return spec


def feature_key(spec):
    channels = '-'.join(str(value) for value in spec['tda_channels'])
    return (
        f"{spec['version']}-length-{spec['length']}-dim-{spec['tda_dim']}-"
        f"delay-{spec['tda_delay']}-points-{spec['tda_points']}-"
        f"pi-{spec['pi_size']}-sigma-{float(spec['pi_sigma']):g}-channels-{channels}"
    )


def persistence_images(x, spec):
    """Fixed persistence weighting, no per-diagram maximum normalization.

    Embedding coordinates lie in [0,1]; both birth and persistence axes cover
    [0,sqrt(3)]. We sample normalized Gaussian densities at fixed grid centers.
    """
    from ripser import ripser
    z = normalize(x)
    if z.ndim != 1:
        raise ValueError('persistence_images expects one waveform.')
    amp = np.abs(z)
    n, d, tau = len(amp), spec['tda_dim'], spec['tda_delay']
    count = n - (d - 1)*tau
    if d != 3 or count < spec['tda_points'] or tau < 1:
        raise ValueError('TDA requires dimension 3 and enough unique embedded points.')
    size, sigma = spec['pi_size'], spec['pi_sigma']
    if size < 2 or sigma <= 0:
        raise ValueError('Invalid persistence image grid.')
    result = np.zeros((2, size, size), dtype=np.float64)
    span = np.ptp(amp)
    if span < 1e-12:
        return result.astype(np.float32)
    amp = (amp - amp.min()) / span
    cloud = np.stack([amp[i*tau:i*tau+count] for i in range(d)], axis=1)
    cloud = cloud[np.linspace(0, count-1, spec['tda_points'], dtype=int)]
    # Direct squared differences avoid cancellation in ||a||^2+||b||^2-2*a.b,
    # and avoid platform BLAS floating-point warnings for these tiny clouds.
    distances = np.sqrt(np.sum((cloud[:, None, :]-cloud[None, :, :])**2, axis=-1))
    if not np.isfinite(distances).all() or distances.max() > np.sqrt(3.)+1e-12:
        raise IntegrityError('Invalid normalized Takens-cloud distance matrix.')
    diagrams = ripser(distances, distance_matrix=True, maxdim=1)['dgms']
    grid = (np.arange(size) + 0.5)*np.sqrt(3.)/size
    birth_grid, persistence_grid = np.meshgrid(grid, grid, indexing='xy')
    for k in spec['tda_channels']:
        if k not in (0, 1):
            raise ValueError('Only H0 and H1 are supported.')
        for birth, death in diagrams[k]:
            persistence = death - birth
            if not np.isfinite(death) or persistence <= 0:
                continue
            distance = (birth_grid-birth)**2 + (persistence_grid-persistence)**2
            result[k] += persistence*np.exp(-distance/(2*sigma**2))/(2*np.pi*sigma**2)
    if not np.isfinite(result).all():
        raise IntegrityError('Nonfinite persistence image.')
    return result.astype(np.float32)


def extract_features(windows, spec):
    windows = np.asarray(windows)
    return hoc_features(windows).astype(np.float32), np.stack(
        [persistence_images(x, spec) for x in windows])


def fit_scaler(hoc, labels, splits, window_ids):
    mask = np.asarray(splits) == 'train'
    if not mask.any():
        raise IntegrityError('Cannot fit scaler without training windows.')
    x = np.asarray(hoc[mask], dtype=np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    return {'mean': mean.tolist(), 'scale': np.where(std < 1e-8, 1., std).tolist(),
            'constant_columns': np.flatnonzero(std < 1e-8).tolist(),
            'n_train': int(mask.sum()), 'feature_names': HOC_NAMES}


def add_noise(x, snr_db, seed, colored=False):
    """Optional deterministic diagnostic; fresh Gaussian real AND imaginary parts.

    Colored noise filters a complex Gaussian time series in the frequency domain.
    Filtering alone does not make it non-Gaussian. DC is explicitly removed.
    """
    z = normalize(x)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=z.shape) + 1j*rng.normal(size=z.shape)
    if colored:
        f = np.abs(np.fft.fftfreq(z.shape[-1]))
        weight = np.zeros_like(f)
        weight[f > 0] = 1/np.sqrt(f[f > 0])
        noise = np.fft.ifft(np.fft.fft(noise, axis=-1)*weight, axis=-1)
    noise = normalize(noise)
    return normalize(z + noise*10**(-float(snr_db)/20))
