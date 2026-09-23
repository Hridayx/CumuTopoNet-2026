"""One public command for all CumuTopoNet experiment suites."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def _config(parser):
    parser.add_argument("-c", "--config", required=True, help="Local YAML configuration")


def parser():
    cli = argparse.ArgumentParser(prog="cumutopo", description="CumuTopoNet drone-RF experiments")
    suites = cli.add_subparsers(dest="suite", required=True)

    matched = suites.add_parser("matched-128", help="Matched 128-sample study: 10 pilots and 33 core fits")
    m = matched.add_subparsers(dest="action", required=True)
    for name in ("doctor", "inventory", "prepare", "plan", "worker", "status", "evaluate", "report"):
        cmd = m.add_parser(name)
        _config(cmd)
        if name == "doctor":
            cmd.add_argument("--cuda", action="store_true")
            cmd.add_argument("--device", default="cuda:0")
            cmd.add_argument("--verify-workers", action="store_true")
            cmd.add_argument("--worker-name", default="local")
            cmd.add_argument("--tests", choices=("all", "math", "data", "models", "training", "orchestration", "reporting"))
        elif name == "prepare":
            cmd.add_argument("--benchmark", action="store_true")
            cmd.add_argument("--robustness", action="store_true")
            cmd.add_argument("--length", type=int, choices=(128, 1024), default=128)
            cmd.add_argument("--points", type=int)
            cmd.add_argument("--channels", nargs="+", type=int, choices=(0, 1))
        elif name == "plan":
            modes = cmd.add_mutually_exclusive_group(required=True)
            modes.add_argument("--budget", action="store_true")
            modes.add_argument("--pilots", action="store_true")
            modes.add_argument("--advance", action="store_true")
            modes.add_argument("--optional", choices=("long", "topology", "robustness"))
        elif name == "worker":
            cmd.add_argument("--name", required=True)
            cmd.add_argument("--device", default="cuda:0")
            cmd.add_argument("--once", action="store_true")
            cmd.add_argument("--benchmark", action="store_true")
            cmd.add_argument("--smoke-steps", type=int)
            cmd.add_argument("--smoke-label", default="primary")
            cmd.add_argument("--model", default="full")
            cmd.add_argument("--lr", type=float, default=.0003)
            cmd.add_argument("--length", type=int, choices=(128, 1024), default=128)
            cmd.add_argument("--points", type=int)
            cmd.add_argument("--channels", nargs="+", type=int, choices=(0, 1))
        elif name == "status":
            cmd.add_argument("--recover", action="store_true")
            cmd.add_argument("--advance", action="store_true")
            cmd.add_argument("--watch", action="store_true")
        elif name == "evaluate":
            cmd.add_argument("--device", default="cuda:0")
            cmd.add_argument("--partial", action="store_true")
            cmd.add_argument("--worker-index", type=int, default=0)
            cmd.add_argument("--worker-count", type=int, default=1)
            cmd.add_argument("--skip-latency", action="store_true")
            cmd.add_argument("--robustness", action="store_true")
        elif name == "report":
            cmd.add_argument("--output-dir")

    full = suites.add_parser("full-1024", help="Full-window standard and extended studies")
    f = full.add_subparsers(dest="action", required=True)
    for name in ("inventory", "prepare", "prepare-sensitivity", "plan", "worker", "status", "report"):
        cmd = f.add_parser(name)
        _config(cmd)
        if name == "plan":
            cmd.add_argument("--suite", dest="experiment_suite",
                             choices=("standard-1024", "extended-1024"), required=True)
        elif name == "worker":
            cmd.add_argument("--name", required=True)
            cmd.add_argument("--device", default="cuda:0")
            cmd.add_argument("--once", action="store_true")
        elif name == "status":
            cmd.add_argument("--recover", action="store_true")
            cmd.add_argument("--advance-lstm", action="store_true")
            cmd.add_argument("--watch", action="store_true")

    analysis = suites.add_parser("analyze", help="Interference, subgroup, and cost analyses")
    a = analysis.add_subparsers(dest="action", required=True)
    for name in ("interference", "subgroups", "cost"):
        cmd = a.add_parser(name)
        _config(cmd)
        if name == "interference":
            cmd.add_argument("--per-class", type=int, default=128)
        elif name == "subgroups":
            cmd.add_argument("--replicates", type=int, default=2000)
        elif name == "cost":
            cmd.add_argument("--device", default="cuda:0")
            cmd.add_argument("--batch-size", type=int, default=256)
            cmd.add_argument("--repeats", type=int, default=20)
    return cli


def _tests(cfg, gate):
    from .matched.common import IntegrityError, write_json, workspace
    mapping = {"math": ("test_math.py",), "data": ("test_data.py",),
               "models": ("test_models.py",), "training": ("test_training.py",),
               "orchestration": ("test_queue.py", "test_protocol.py"),
               "reporting": ("test_metrics.py", "test_reporting.py")}
    if gate == "all":
        return [_tests(cfg, name) for name in mapping]
    tests = Path(__file__).resolve().parents[2] / "tests"
    if not tests.is_dir():
        raise IntegrityError("Source checkout with tests/ is required for software gates.")
    command = [sys.executable, "-m", "pytest", "-q", *(str(tests / item) for item in mapping[gate])]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True)
    root = workspace(cfg)
    log = root / "gates" / f"{gate}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + result.stderr)
    record = {"gate": gate, "status": "passed" if result.returncode == 0 else "failed",
              "command": command, "elapsed_seconds": time.monotonic() - started,
              "log": str(log)}
    write_json(root / "gates" / f"{gate}.json", record)
    if result.returncode:
        raise IntegrityError(f"Software gate {gate} failed; see {log}")
    return record


def _matched(args, cfg):
    from .matched.common import workspace, write_json
    action = args.action
    if action == "doctor":
        if args.tests:
            return _tests(cfg, args.tests)
        from .matched.environment import doctor
        return doctor(cfg, args.cuda, args.worker_name, args.verify_workers, args.device)
    if action == "inventory":
        from .matched.data import inventory
        result = inventory(cfg)
        write_json(workspace(cfg) / "inventory.json", result)
        return {key: result[key] for key in ("counts", "total_bytes", "missing_combinations")}
    if action == "prepare":
        if args.benchmark:
            from .matched.benchmark import cpu_benchmark
            return cpu_benchmark(cfg)
        if args.robustness:
            from .matched.evaluation import robustness
            return robustness(cfg, device="cpu", features_only=True)
        from .matched.data import prepare_dataset
        return prepare_dataset(cfg, args.length, args.points, args.channels)
    if action == "plan":
        from .matched.protocol import budget_plan, start_pilots, advance, optional_batch
        if args.budget:
            return budget_plan(cfg)
        if args.pilots:
            return start_pilots(cfg)
        if args.advance:
            return advance(cfg)
        return optional_batch(cfg, args.optional)
    if action == "worker":
        if args.benchmark:
            from .matched.benchmark import gpu_benchmark
            return gpu_benchmark(cfg, args.device)
        if args.smoke_steps is not None:
            from .matched.training import numerical_smoke
            return numerical_smoke(cfg, args.device, args.smoke_steps, args.smoke_label,
                                   args.model, args.lr, args.length, args.points, args.channels)
        from .matched.runner import worker
        return worker(cfg, args.name, args.device, args.once)
    if action == "status":
        from .matched.runner import monitor
        while True:
            result = monitor(cfg, args.recover, args.advance)
            summary = {"counts": result["counts"], "protocol": result["protocol"],
                       "status_file": str(workspace(cfg) / "status.json"),
                       "recovery_findings": result["recovery_findings"]}
            if not args.watch:
                return summary
            print(json.dumps(summary, indent=2), flush=True)
            time.sleep(cfg["runtime"]["monitor_seconds"])
    if action == "evaluate":
        from .matched.evaluation import evaluate, robustness
        if args.robustness:
            return robustness(cfg, args.device)
        return evaluate(cfg, args.device, args.partial, args.worker_index,
                        args.worker_count, not args.skip_latency)
    if action == "report":
        from .matched.reporting import report
        return report(cfg, args.output_dir)
    raise ValueError(action)


def _full(args, cfg):
    action = args.action
    if action == "inventory":
        from .full.data import experiment_inventory
        return experiment_inventory(cfg)
    if action == "prepare":
        from .full.data import prepare
        return prepare(cfg)
    if action == "prepare-sensitivity":
        from .full.data import prepare_sensitivity
        return prepare_sensitivity(cfg)
    if action == "plan":
        if args.experiment_suite == "standard-1024":
            from .full.protocol import commit_plan
            return commit_plan(cfg)
        from .full.extended import commit_extended
        return commit_extended(cfg)
    if action == "worker":
        from .full.runner import worker
        return worker(cfg, args.name, args.device, args.once)
    if action == "status":
        from .full.runner import status
        from .full.extended import advance_lstm
        while True:
            result = status(cfg, args.recover)
            if args.advance_lstm:
                result["lstm_selection"] = advance_lstm(cfg)
            if not args.watch:
                return result
            print(json.dumps(result, indent=2), flush=True)
            time.sleep(cfg["runtime"]["heartbeat_seconds"])
    if action == "report":
        from .full.reporting import report
        return report(cfg)
    raise ValueError(action)


def dispatch(args):
    if args.suite == "matched-128":
        from .matched.common import load_config
        return _matched(args, load_config(args.config))
    from .full.common import load_config
    cfg = load_config(args.config)
    if args.suite == "full-1024":
        return _full(args, cfg)
    if args.action == "interference":
        from .analysis.interference import run
        return run(cfg, args.per_class)
    if args.action == "subgroups":
        from .analysis.subgroups import run
        return run(cfg, args.replicates)
    if args.action == "cost":
        from .analysis.cost import run
        return run(cfg, args.device, args.batch_size, args.repeats)
    raise ValueError(args.action)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if isinstance(result, dict) and (result.get("state") == "blocked" or result.get("feasible") is False) else 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(json.dumps({"status": "blocked", "type": type(error).__name__, "message": str(error)}, indent=2), file=sys.stderr)
        return 2
