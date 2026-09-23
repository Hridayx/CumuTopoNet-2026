#!/usr/bin/env python3
"""Exercise every optional path on a completed SOFTWARE FIXTURE only."""
import argparse
from cumutoponet.matched.common import load_config, write_json, workspace
from cumutoponet.matched.protocol import optional_batch
from cumutoponet.matched.data import prepare_dataset
from cumutoponet.matched.runner import worker
from cumutoponet.matched.queue import JobQueue
from cumutoponet.matched.evaluation import evaluate, robustness
from cumutoponet.matched.reporting import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.get('fixture', False):
        raise ValueError('This diagnostic is restricted to synthetic fixtures.')
    root = workspace(cfg)
    for name in ['long', 'topology']:
        decision = optional_batch(cfg, name)
        for f in decision.get('feature_requests', []):
            prepare_dataset(cfg, f['length'], f['points'], f['channels'])
        decision = optional_batch(cfg, name)
        if decision['state'] != 'queued':
            raise RuntimeError(decision)
        while any(JobQueue(root).status()[j]['state'] != 'completed' for j in decision['job_ids']):
            result = worker(cfg, 'optional-fixture', device='cpu', once=True)
            if result.get('state') == 'blocked':
                raise RuntimeError(result)
    decision = optional_batch(cfg, 'robustness')
    if decision['state'] != 'ready-for-evaluation':
        raise RuntimeError(decision)
    robustness(cfg, device='cpu', features_only=True)
    result = robustness(cfg, device='cpu')
    if result['robustness_outputs'] < 1:
        raise RuntimeError(result)
    evaluate(cfg, device='cpu', measure_latency=False)
    report_result = report(cfg)
    record = {'scope': 'SYNTHETIC SOFTWARE TEST ONLY',
              'robustness_outputs': result['robustness_outputs'],
              'evaluated_runs': report_result['evaluated_runs']}
    write_json(root/'optional_smoke_result.json', record)
    print(record)


if __name__ == '__main__':
    main()
