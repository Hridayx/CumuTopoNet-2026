import numpy as np
from cumutoponet.matched.metrics import classification_metrics, paired_group_interval


def test_metrics_match_worked_confusion_matrix():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 0, 2])
    result = classification_metrics(y, p, classes=3)
    assert result['accuracy'] == 4/6
    np.testing.assert_allclose(result['balanced_accuracy'], 2/3)
    np.testing.assert_allclose(result['macro_f1'], (.5 + .8 + 2/3)/3)
    assert result['confusion_matrix'] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    absent = classification_metrics(np.array([0]), np.array([0]), classes=3)
    assert absent['per_class'][1]['recall'] is None


def test_paired_resampling_keeps_identical_models_identical():
    y = np.tile(np.arange(7), 4)
    groups = np.array([f'{label}-{rep}' for rep in range(4) for label in range(7)])
    pred = np.tile(y, (3, 1))
    result = paired_group_interval(y, pred, pred, groups, replicates=100, seed=17)
    for metric in ['accuracy', 'balanced_accuracy', 'macro_f1']:
        assert result[metric]['difference'] == 0.
        assert result[metric]['ci95'] == [0., 0.]


def test_perfect_minus_always_wrong_has_unit_paired_effect():
    y = np.tile(np.arange(7), 6)
    groups = np.array([f'{label}-{rep//2}' for rep in range(6) for label in range(7)])
    perfect = np.tile(y, (3, 1))
    wrong = (perfect+1) % 7
    result = paired_group_interval(y, perfect, wrong, groups, replicates=100)
    for metric in ['accuracy', 'balanced_accuracy', 'macro_f1']:
        np.testing.assert_allclose(result[metric]['difference'], 1.)
        np.testing.assert_allclose(result[metric]['ci95'], [1., 1.])
