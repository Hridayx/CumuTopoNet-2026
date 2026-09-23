from cumutoponet.matched.reporting import summarize_runs


def test_incomplete_seed_sets_are_never_presented_as_completed_results():
    rows = [{'spec': {'model': 'full', 'seed': 17},
             'metrics': {'accuracy': .9, 'balanced_accuracy': .8, 'macro_f1': .7}}]
    report = summarize_runs(rows, ['full', 'lstm'], [17, 42, 91])
    assert report['full']['complete'] is False
    assert report['full']['accuracy_mean'] is None
    assert report['full']['completed_seeds'] == [17]
    assert report['lstm']['completed_seeds'] == []


def test_seed_std_is_distinct_from_a_group_confidence_interval():
    rows = [{'spec': {'model': 'full', 'seed': s},
             'metrics': {'accuracy': v, 'balanced_accuracy': v, 'macro_f1': v}}
            for s, v in [(17, .7), (42, .8), (91, .9)]]
    report = summarize_runs(rows, ['full'], [17, 42, 91])['full']
    assert report['complete']
    assert abs(report['accuracy_mean']-.8) < 1e-10
    assert abs(report['accuracy_seed_std']-.1) < 1e-10
    assert 'ci95' not in report


def test_uncached_latency_features_agree_with_training_views_and_are_finite():
    import numpy as np
    import torch
    from cumutoponet.matched.common import DEFAULTS
    from cumutoponet.matched.features import feature_spec, extract_features
    from cumutoponet.matched.training import make_inputs
    from cumutoponet.matched.evaluation import waveform_inputs, latency
    from cumutoponet.matched.models import build_model
    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(8, 128)) + 1j*rng.normal(size=(8, 128))
    spec = feature_spec(DEFAULTS)
    hoc, pi = extract_features(raw, spec)
    arrays = {'raw': raw, 'hoc': hoc, 'pi': pi, 'splits': np.array(['train']*8),
              'scaler': {'mean': [0.]*6, 'scale': [1.]*6}, 'feature_spec': spec}
    cached = make_inputs(arrays, [0], 'cpu')
    fresh = waveform_inputs(raw[0], 'full', arrays['scaler'], spec, 'cpu')
    for key in ['hoc', 'temporal', 'tda']:
        np.testing.assert_allclose(cached[key], fresh[key], atol=2e-5)
    result = latency(build_model('full'), arrays, 'full', 'cpu', batch_size=4, repeats=5)
    assert result['end_to_end_batch1']['p95_ms'] >= result['end_to_end_batch1']['p50_ms'] > 0
    assert result['network_batch_throughput_windows_s'] > 0
    assert result['peak_cuda_allocated_bytes'] is None
