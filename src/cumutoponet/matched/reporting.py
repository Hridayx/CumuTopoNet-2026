"""Tables and figures derived from saved predictions, never hand-entered scores."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .common import IntegrityError, read_json, write_json, atomic_target, workspace
from .metrics import classification_metrics, paired_group_interval
from .benchmark import PRINCIPAL
from .data import LABELS
from .queue import JobQueue

NAMES = {'hoc': 'HOC', 'temporal': 'Temporal', 'tda': 'TDA', 'hoc_temporal': 'HOC + temporal',
         'hoc_tda': 'HOC + TDA', 'temporal_tda': 'Temporal + TDA', 'full': 'Full fusion (CE)',
         'raw_iq': 'Raw-IQ TCN', 'lstm': 'LSTM-256', 'wide_temporal': 'Wider temporal',
         'full_supcon': 'Full fusion (CE + SupCon)'}


def summarize_runs(rows, models, seeds):
    result = {}
    for model in models:
        selected = [r for r in rows if r['spec']['model'] == model]
        observed = sorted(r['spec']['seed'] for r in selected)
        if len(set(observed)) != len(observed):
            raise IntegrityError('Duplicate seed/configuration in principal summary.')
        complete = observed == sorted(seeds)
        item = {'complete': complete, 'completed_seeds': observed, 'expected_seeds': seeds,
                'seed_results': {str(r['spec']['seed']): r['metrics'] for r in selected}}
        for metric in ['accuracy', 'balanced_accuracy', 'macro_f1']:
            values = [r['metrics'][metric] for r in selected]
            item[f'{metric}_mean'] = float(np.mean(values)) if complete else None
            item[f'{metric}_seed_std'] = float(np.std(values, ddof=1)) if complete and len(values) > 1 else None
        result[model] = item
    return result


def csv_file(path, rows, fields):
    with atomic_target(path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: 'NA' if row.get(k) is None else row.get(k) for k in fields})


def load_evidence(root, protocol):
    rows = []
    for marker in sorted((root/'evaluation').glob('*/complete.json')):
        complete = read_json(marker)
        jid = complete['job_id']
        spec = read_json(root/'queue'/jid/'spec.json')
        context = complete['context']
        if context['protocol_id'] != protocol['protocol_id'] or context['spec'] != spec:
            raise IntegrityError('Report inputs do not match the active protocol.')
        for name in complete['files']:
            if not (marker.parent/name).is_file():
                raise IntegrityError(f'Report input is missing: {name}')
        obs = dict(np.load(marker.parent/'observations.npz', allow_pickle=False))
        probs = np.load(marker.parent/'probabilities.npy', allow_pickle=False)
        if probs.shape != (len(obs['ids']), 7) or not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1., atol=1e-5):
            raise IntegrityError('Invalid saved class probabilities.')
        pred = probs.argmax(axis=1)
        metrics = classification_metrics(obs['labels'], pred)
        saved = read_json(marker.parent/'metrics.json')
        if metrics != saved['overall']:
            raise IntegrityError('Saved metrics disagree with independently reopened predictions.')
        result = read_json(root/'runs'/jid/'result.json')
        rows.append({'job': jid, 'spec': spec, 'metrics': metrics, 'saved_metrics': saved,
                     'obs': obs, 'pred': pred, 'latency': read_json(marker.parent/'latency.json'),
                     'result': result})
    return rows


def report(cfg, output_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = workspace(cfg)
    protocol = read_json(root/'protocol.json')
    rows = load_evidence(root, protocol)
    main_rows = [r for r in rows if r['spec']['phase'] == 'core']
    expected_seeds = sorted({s['seed'] for s in protocol['core_specs']})
    summary = summarize_runs(main_rows, PRINCIPAL, expected_seeds)
    directory = Path(output_dir).expanduser().resolve() if output_dir else root/'reports'
    directory.mkdir(parents=True, exist_ok=True)
    data = read_json(root/'data_plan.json')
    context = {'protocol_id': protocol['protocol_id'], 'data_plan_id': data['plan_id'],
               'fixture': protocol['fixture']}
    complete = all(r['complete'] for r in summary.values())
    latency_complete = all('end_to_end_batch1' in r['latency'] for r in main_rows) and complete
    write_json(directory/'summary.json', {'context': context, 'primary': summary,
               'primary_complete': complete, 'latency_complete': latency_complete})
    table = [{'configuration': NAMES[m], 'model': m, **summary[m]} for m in PRINCIPAL]
    csv_file(directory/'primary_results.csv', table, ['configuration', 'complete', 'completed_seeds',
        'accuracy_mean', 'accuracy_seed_std', 'balanced_accuracy_mean', 'balanced_accuracy_seed_std',
        'macro_f1_mean', 'macro_f1_seed_std'])
    # Export all optional and partial results individually, never disguise a partial set as a full comparison.
    csv_file(directory/'all_runs.csv', [{'job': r['job'], **{k: r['spec'][k] for k in ['phase', 'model', 'seed', 'length', 'points', 'channels']},
        **{k: r['metrics'][k] for k in ['accuracy', 'balanced_accuracy', 'macro_f1']},
        'parameters': r['result']['parameters']['trainable']} for r in rows],
        ['job', 'phase', 'model', 'seed', 'length', 'points', 'channels', 'parameters', 'accuracy', 'balanced_accuracy', 'macro_f1'])
    per_class, subgroup, cells, processing = [], [], [], []
    for r in rows:
        for item in r['metrics']['per_class']:
            per_class.append({'job': r['job'], 'model': r['spec']['model'], 'seed': r['spec']['seed'],
                              'label': LABELS[item['class']], **item})
        for field, values in r['saved_metrics']['subgroups'].items():
            for value, scores in values.items():
                subgroup.append({'job': r['job'], 'field': field, 'value': value, **scores})
        cells.extend({'job': r['job'], **cell} for cell in r['saved_metrics']['condition_cells'])
        processing.append({'job': r['job'], 'model': r['spec']['model'], 'seed': r['spec']['seed'],
            'network_p50_ms': r['latency'].get('network_batch1', {}).get('p50_ms'),
            'network_p95_ms': r['latency'].get('network_batch1', {}).get('p95_ms'),
            'features_p50_ms': r['latency'].get('feature_batch1', {}).get('p50_ms'),
            'features_p95_ms': r['latency'].get('feature_batch1', {}).get('p95_ms'),
            'total_p50_ms': r['latency'].get('end_to_end_batch1', {}).get('p50_ms'),
            'total_p95_ms': r['latency'].get('end_to_end_batch1', {}).get('p95_ms'),
            'network_batch_windows_s': r['latency'].get('network_batch_throughput_windows_s'),
            'serial_end_to_end_windows_s': r['latency'].get('end_to_end_serial_windows_s'),
            'peak_cuda_bytes': r['latency'].get('peak_cuda_allocated_bytes'),
            'process_rss_bytes': r['latency'].get('process_rss_bytes')})
    csv_file(directory/'per_class.csv', per_class, ['job', 'model', 'seed', 'label', 'support', 'predicted', 'precision', 'recall', 'f1'])
    csv_file(directory/'subgroups.csv', subgroup, ['job', 'field', 'value', 'support', 'accuracy', 'balanced_accuracy', 'macro_f1'])
    csv_file(directory/'condition_cells.csv', cells, ['job', 'label', 'mode', 'condition', 'support', 'recall'])
    csv_file(directory/'processing_cost.csv', processing, list(processing[0]) if processing else ['job', 'status'])
    intervals = {}
    def ordered(model):
        return sorted([r for r in main_rows if r['spec']['model'] == model], key=lambda r: r['spec']['seed'])
    if summary['full']['complete']:
        reference = ordered('full')
        obs = reference[0]['obs']
        for model in protocol['evaluation']['primary_contrasts'] + ['full_supcon']:
            if not summary[model]['complete']:
                continue
            comparison = ordered(model)
            for r in reference+comparison:
                for key in ['ids', 'labels', 'groups']:
                    if not np.array_equal(obs[key], r['obs'][key]):
                        raise IntegrityError('Paired evaluation observation/label/group alignment mismatch.')
            intervals[f'full-minus-{model}'] = paired_group_interval(obs['labels'],
                np.stack([r['pred'] for r in reference]), np.stack([r['pred'] for r in comparison]), obs['groups'],
                replicates=protocol['evaluation']['bootstrap_replicates'], seed=protocol['evaluation']['bootstrap_seed'])
    write_json(directory/'paired_intervals.json', {'context': context, 'comparisons': intervals,
        'multiplicity': 'Prespecified descriptive marginal 95% intervals; no family-wise significance claim.'})
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    if complete:
        fig, ax = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
        values = [100*summary[m]['macro_f1_mean'] for m in PRINCIPAL]
        std = [100*(summary[m]['macro_f1_seed_std'] or 0) for m in PRINCIPAL]
        ax.barh([NAMES[m] for m in PRINCIPAL], values, xerr=std, color='#256b8a', capsize=3)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel('Macro-F1 (%) - mean and sample SD across training seeds')
        if protocol['fixture']:
            ax.set_title('SYNTHETIC TEST FIXTURE - NOT RESEARCH RESULTS')
        fig.savefig(directory/'ablations.pdf')
        fig.savefig(directory/'ablations.png', dpi=200)
        plt.close(fig)
        full_rows = ordered('full')
        cm = np.mean([r['metrics']['confusion_matrix'] for r in full_rows], axis=0)
        cm = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(1, keepdims=True) > 0)
        fig, ax = plt.subplots(figsize=(3.2, 3.35), constrained_layout=True)
        image = ax.imshow(cm, vmin=0, vmax=1, cmap='Blues')
        ax.set_xticks(range(7), LABELS, rotation=45, ha='right')
        ax.set_yticks(range(7), LABELS)
        ax.set_xlabel('Predicted model')
        ax.set_ylabel('True model')
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center', fontsize=9,
                        color='white' if cm[i,j] > .5 else '#152939')
        fig.savefig(directory/'confusion.pdf')
        fig.savefig(directory/'confusion.png', dpi=200)
        plt.close(fig)
    support_rows = []
    for record in data['inventory']['records']:
        fold = data['partitions'][record['id']]
        support_rows.append({**record, 'partition': fold, 'split': {0: 'test', 1: 'validation'}.get(fold, 'train'),
                             'windows': data['windows_per_recording']})
    csv_file(directory/'recording_support.csv', support_rows, ['id', 'label_name', 'family', 'group', 'mode', 'condition', 'partition', 'split', 'windows'])
    write_json(directory/'missing_combinations.json', data['inventory']['missing_combinations'])
    status = 'Complete primary comparison' if complete else 'Incomplete primary comparison; missing entries remain pending'
    text = f'# Generated experiment report\n\n{status}.\n\n'
    if protocol['fixture']:
        text += '**SYNTHETIC FIXTURE ONLY. These values are software-test output.**\n\n'
    text += ('Seed standard deviations and paired family-bootstrap intervals answer different questions. '
             'The intervals are conditional on this one fixed partition and the fixed training seeds. '
             'They do not establish performance on unseen datasets or a physical classification ceiling.\n')
    with atomic_target(directory/'report.md', 'w') as f:
        f.write(text)
    return {'directory': str(directory), 'primary_complete': complete, 'latency_complete': latency_complete,
            'evaluated_runs': len(rows)}
