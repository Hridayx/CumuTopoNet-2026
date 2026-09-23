#!/usr/bin/env python3
"""Run a small synthetic workflow for local verification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import signal
from cumutoponet.matched.fixtures import create_fixture
from cumutoponet.matched.common import load_config, workspace, write_json
from cumutoponet.matched.benchmark import cpu_benchmark, gpu_benchmark
from cumutoponet.matched.protocol import budget_plan, start_pilots, advance
from cumutoponet.matched.data import prepare_dataset
from cumutoponet.matched.runner import worker
from cumutoponet.matched.queue import JobQueue
from cumutoponet.matched.evaluation import evaluate
from cumutoponet.matched.reporting import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New scratch directory; never a real experiment workspace')
    parser.add_argument('--full', action='store_true', help='Execute the complete fixture workflow and report')
    parser.add_argument('--workers', type=int, choices=[1, 2], default=1, help='Use two actual CLI processes to check shared queue execution')
    args = parser.parse_args()
    config_path = create_fixture(args.output)
    cfg = load_config(config_path)
    start = time.monotonic()
    cpu_benchmark(cfg)
    gpu_benchmark(cfg, device='cpu', steps=2)
    budget_plan(cfg)
    prepare_dataset(cfg)
    start_pilots(cfg)
    fits, processes, logs = 0, [], []
    try:
        if args.workers == 2:
            if not args.full:
                raise ValueError('--workers 2 requires --full')
            for i in range(2):
                log = (workspace(cfg)/f'fixture-worker-{i}.log').open('w')
                logs.append(log)
                processes.append(subprocess.Popen([sys.executable, '-m', 'cumutoponet', 'matched-128', 'worker', '-c', str(config_path),
                    '--name', f'fixture-{i}', '--device', 'cpu'], stdout=log, stderr=subprocess.STDOUT))
        while True:
            if args.workers == 1:
                worker(cfg, 'fixture', device='cpu', once=True)
            elif any(p.poll() is not None for p in processes):
                raise RuntimeError('A fixture worker exited early; inspect its log.')
            status = advance(cfg)
            states = JobQueue(cfg['workspace']).status()
            fits = sum(s['state'] == 'completed' for s in states.values())
            if not args.full or status['stage'] == 'core-completed':
                break
            if any(s['state'] in ['blocked', 'budget-skipped'] for s in states.values()):
                raise RuntimeError('Synthetic pipeline blocked; inspect its diagnostic files.')
            if time.monotonic()-start > 600:
                raise RuntimeError('Fixture exceeded ten-minute diagnostic limit.')
            if processes:
                time.sleep(.2)
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    if args.full:
        evaluate(cfg, device='cpu', measure_latency=False)
        report(cfg)
    result = {'scope': 'SYNTHETIC SOFTWARE TEST ONLY', 'full': args.full, 'fits': fits, 'workers': args.workers,
              'stage': status['stage'], 'elapsed_seconds': time.monotonic()-start}
    write_json(workspace(cfg)/'smoke_result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
