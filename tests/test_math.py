import math
import numpy as np
from cumutoponet.matched.features import (normalize, cumulants, hoc_features, temporal_features,
                               persistence_images, feature_spec, add_noise)
from cumutoponet.matched.common import DEFAULTS


def partitions(items):
    if not items:
        yield []
        return
    first, *rest = items
    for part in partitions(rest):
        yield [[first]] + part
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]


def joint_cumulant(x, p, q):
    variables = [x] * (p - q) + [x.conj()] * q
    answer = 0j
    for part in partitions(list(range(p))):
        term = complex(math.factorial(len(part) - 1) * (-1) ** (len(part) - 1))
        for block in part:
            term *= np.mean(np.prod([variables[i] for i in block], axis=0))
        answer += term
    return answer


def test_bpsk_and_qpsk_known_population_values():
    bpsk = np.tile([-1., 1.], 64).astype(complex)
    qpsk = np.tile([1., 1j, -1., -1j], 32)
    np.testing.assert_allclose(hoc_features(bpsk), [1, 2, 2, -2, 16, 16], atol=1e-12)
    np.testing.assert_allclose(hoc_features(qpsk), [0, 1, 0, -1, 0, 4], atol=1e-12)


def test_full_cumulants_agree_with_independent_partition_definition():
    x = normalize(np.array([-2-1j, -1+.5j, -1+.5j, 0, 0, 0, 3+2j]))
    cs = cumulants(x)
    for p, q in [(2, 0), (4, 0), (4, 1), (4, 2), (6, 0), (6, 3)]:
        np.testing.assert_allclose(cs[f'C{p}{q}'], joint_cumulant(x, p, q), atol=1e-10)


def test_normalization_phase_invariance_and_degenerate_inputs():
    rng = np.random.default_rng(4)
    x = rng.normal(size=128) + 1j*rng.normal(size=128)
    np.testing.assert_allclose(normalize(x+3-7j), normalize(x), atol=1e-14)
    np.testing.assert_allclose(hoc_features(x*np.exp(.721j)), hoc_features(x), atol=1e-12)
    spec = feature_spec(DEFAULTS)
    np.testing.assert_allclose(persistence_images(x*np.exp(.721j), spec),
                               persistence_images(x, spec), atol=2e-5)
    np.testing.assert_allclose(normalize(np.ones(128)*(3+4j)), 0.)
    assert not persistence_images(np.ones(128), spec).any()
    assert not hoc_features(np.zeros(128)).any()
    assert temporal_features(x)[2, 0] == 0.
    np.testing.assert_allclose(np.mean(normalize(x)), 0., atol=1e-15)
    np.testing.assert_allclose(np.mean(abs(normalize(x))**2), 1.)


def test_gaussian_complex_mean_and_magnitude_are_different():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(3000, 1024)) + 1j*rng.normal(size=(3000, 1024))
    c = cumulants(x)
    assert abs(c['C40'].mean()) < .015
    assert abs(c['C60'].mean()) < .04
    assert np.abs(c['C40']).mean() > .05  # a magnitude estimator is biased
    assert abs(c['C42'].mean()) < .02
    assert abs(c['C63'].mean()) < .06


def test_colored_noise_is_not_an_even_time_signal():
    x = np.tile([1, 1j, -1, -1j], 32)
    y = add_noise(x, -10, 91, colored=True)
    assert np.max(abs(y[1:] - y[:0:-1])) > .1
    assert abs(y.mean()) < 1e-14


def test_larger_tda_cloud_has_finite_images_without_runtime_warnings():
    import warnings
    rng = np.random.default_rng(91)
    spec = feature_spec(DEFAULTS, points=96)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        for _ in range(8):
            wave = rng.normal(size=128) + 1j*rng.normal(size=128)
            assert np.isfinite(persistence_images(wave, spec)).all()
