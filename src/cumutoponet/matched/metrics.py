"""Metrics with explicit support and paired, stratified recording-group intervals."""
import numpy as np


def metric_values(cm):
    cm = np.asarray(cm, dtype=np.float64)
    actual, predicted = cm.sum(axis=-1), cm.sum(axis=-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    f1 = np.divide(2*tp, actual+predicted, out=np.zeros_like(tp), where=actual+predicted > 0)
    total = actual.sum(axis=-1)
    return {'accuracy': np.divide(tp.sum(axis=-1), total, out=np.zeros_like(total), where=total > 0),
            'balanced_accuracy': recall.sum(axis=-1)/np.maximum((actual > 0).sum(axis=-1), 1),
            'macro_f1': f1.mean(axis=-1)}


def confusion(y, pred, classes=7):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    if y.shape != pred.shape or y.ndim != 1 or np.any(y < 0) or np.any(y >= classes) or np.any(pred < 0) or np.any(pred >= classes):
        raise ValueError('Invalid class predictions.')
    return np.bincount(y*classes+pred, minlength=classes*classes).reshape(classes, classes)


def classification_metrics(y, pred, classes=7):
    cm = confusion(y, pred, classes)
    result = {k: float(v) if len(y) else None for k, v in metric_values(cm).items()}
    result.update(support=len(y), confusion_matrix=cm.tolist(), per_class=[],
                  macro_f1_definition='mean over all declared classes; zero F1 when both supports are zero',
                  balanced_accuracy_definition='mean recall over classes with actual support')
    for i in range(classes):
        actual, predicted, tp = int(cm[i].sum()), int(cm[:, i].sum()), int(cm[i, i])
        result['per_class'].append({'class': i, 'support': actual, 'predicted': predicted,
            'precision': tp/predicted if predicted else None,
            'recall': tp/actual if actual else None,
            'f1': 2*tp/(actual+predicted) if actual+predicted else None})
    return result


def paired_group_interval(y, pred_a, pred_b, groups, replicates=2000, seed=42, classes=7):
    """Resample groups, preserving model pairing and the fixed list of training seeds.

    Homogeneous-label groups are stratified by label to preserve class coverage.
    Coarsened sessions spanning labels use an unstratified cluster bootstrap.
    This does not estimate variance across alternative outer dataset partitions.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    a, b = np.atleast_2d(pred_a), np.atleast_2d(pred_b)
    if a.shape != b.shape or a.shape[1] != len(y) or len(groups) != len(y) or replicates < 20:
        raise ValueError('Paired predictions must share seeds, labels and observation IDs.')
    unique = np.unique(groups)
    strata, ca, cb = [], [], []
    for g in unique:
        mask = groups == g
        labels = np.unique(y[mask])
        strata.append(int(labels[0]) if len(labels) == 1 else -1)
        ca.append([confusion(y[mask], p[mask], classes) for p in a])
        cb.append([confusion(y[mask], p[mask], classes) for p in b])
    ca, cb, strata = np.asarray(ca), np.asarray(cb), np.asarray(strata)
    buckets = [np.arange(len(unique))] if -1 in strata else [np.flatnonzero(strata == c) for c in np.unique(strata)]
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(unique)), dtype=int)
    for bucket in buckets:
        draws = rng.choice(bucket, (replicates, len(bucket)), replace=True)
        for i in range(replicates):
            weights[i] += np.bincount(draws[i], minlength=len(unique))
    ma = metric_values(np.einsum('bg,gsij->bsij', weights, ca))
    mb = metric_values(np.einsum('bg,gsij->bsij', weights, cb))
    point_a, point_b = metric_values(ca.sum(axis=0)), metric_values(cb.sum(axis=0))
    result = {'groups': len(unique), 'seeds': len(a), 'replicates': replicates,
              'bootstrap_seed': seed, 'stratified_by_class': -1 not in strata,
              'interpretation': 'paired conditional interval for this fixed split and fixed training seeds',
              'groups_per_stratum': [len(x) for x in buckets]}
    for metric in ma:
        difference = (ma[metric]-mb[metric]).mean(axis=-1)
        result[metric] = {'difference': float(np.mean(point_a[metric]-point_b[metric])),
                          'ci95': np.quantile(difference, [.025, .975]).tolist()}
    return result
