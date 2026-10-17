"""Validate three chat backends, then supervise a balanced repeated ablation."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from examples.qtrade_fin_quant.ablation_report import write_ablation_report
from examples.qtrade_fin_quant.run import write_json


DEFAULT_MODELS = ["gpt-6-luna", "gpt-6-astra", "gpt-6.1-sol"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_jobs(study_id: str, models: list[str], seeds: list[int]) -> list[dict]:
    jobs = [{"kind": "preflight", "model": model, "seed": seeds[0], "state": "queued",
             "run_id": f"{study_id}-{model}-preflight"} for model in models]
    for index, seed in enumerate(seeds):
        order = models[index % len(models):] + models[:index % len(models)]
        jobs.extend({"kind": "research", "model": model, "seed": seed, "state": "queued",
                     "run_id": f"{study_id}-{model}-s{seed}"} for model in order)
    return jobs


def source_commit(root: Path) -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise RuntimeError("The ablation requires committed source; use another checkout for edits while it runs")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def stop_child(child: subprocess.Popen) -> None:
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--hours-per-run", type=int, default=4)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--baseline-run-id", default="fin-baselines-01")
    parser.add_argument("--min-free-gib", type=float, default=50)
    args = parser.parse_args()
    for value in (args.study_id, args.baseline_run_id):
        if not value or Path(value).name != value or value in {".", ".."}:
            parser.error("Study and baseline IDs must be single directory names")
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Models and seeds must be unique")
    if args.hours_per_run < 1 or min(args.seeds) < 0:
        parser.error("Hours must be positive and seeds nonnegative")
    root = Path(__file__).resolve().parents[2]
    commit = source_commit(root)
    baseline_dir = root / "git_ignore_folder/reproduction/runs" / args.baseline_run_id
    if not (baseline_dir / "median-metrics.json").exists():
        parser.error("Complete the reference baselines before starting the ablation")
    output = root / "git_ignore_folder/reproduction/ablations" / args.study_id
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "state": "running", "started_at": now(), "supervisor_pid": os.getpid(), "git_commit": commit,
        "models": args.models, "seeds": args.seeds, "hours_per_run": args.hours_per_run,
        "reasoning_effort": args.reasoning_effort, "gpu": args.gpu, "baseline_dir": str(baseline_dir),
        "nominal_research_hours": len(args.models) * len(args.seeds) * args.hours_per_run,
        "jobs": build_jobs(args.study_id, args.models, args.seeds),
    }
    write_json(output / "status.json", state)
    write_ablation_report(output)
    child = None
    active = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Ablation supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for job in state["jobs"]:
            if source_commit(root) != commit:
                raise RuntimeError("Source revision changed during the ablation")
            active = job
            preflight = job["kind"] == "preflight"
            duration = "1h" if preflight else f"{args.hours_per_run}h"
            hard_limit = "65m" if preflight else f"{args.hours_per_run + 1}h"
            command = ["timeout", "--signal=TERM", "--kill-after=30s", hard_limit,
                       sys.executable, "-u", "-m", "examples.qtrade_fin_quant.run", "--mode", "quant",
                       "--duration", duration, "--run-id", job["run_id"], "--gpu", args.gpu,
                       "--model", job["model"], "--scheduler-seed", str(job["seed"]),
                       "--reasoning-effort", args.reasoning_effort]
            if preflight:
                command.extend(["--loops", "1"])
            job.update({"state": "running", "started_at": now(), "command": command})
            started = time.monotonic()
            with (output / f"{job['run_id']}.console.log").open("x") as log:
                child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
                job["pid"] = child.pid
                write_json(output / "status.json", state)
                while True:
                    if shutil.disk_usage(root).free < args.min_free_gib * 1024**3:
                        raise RuntimeError(f"Free disk space fell below {args.min_free_gib} GiB")
                    if source_commit(root) != commit:
                        raise RuntimeError("Source revision changed during the ablation")
                    try:
                        job["exit_code"] = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        job["last_checked_at"] = now()
                        job["elapsed_seconds"] = round(time.monotonic() - started, 2)
                        write_json(output / "status.json", state)
            child = None
            job.update({"finished_at": now(), "elapsed_seconds": round(time.monotonic() - started, 2)})
            result_path = root / "git_ignore_folder/reproduction/runs" / job["run_id"] / "status.json"
            if result_path.exists():
                job["result"] = json.loads(result_path.read_text())
            success = job["exit_code"] == 0 and job.get("result", {}).get("state") == "finished"
            job["state"] = "finished" if success else "failed"
            write_json(output / "status.json", state)
            write_ablation_report(output)
            if preflight and not success:
                raise RuntimeError(f"{job['model']} preflight failed; research runs were not started")
            active = None
        state["state"] = "finished_with_failures" if any(j["state"] == "failed" for j in state["jobs"]) else "finished"
    except BaseException as error:
        state["state"] = "failed"
        state["error"] = str(error)
        if active and active["state"] == "running":
            active["state"] = "failed"
            active["error"] = str(error)
        raise
    finally:
        if child is not None:
            stop_child(child)
            if active is not None:
                active["exit_code"] = child.returncode
                active["finished_at"] = now()
        state["finished_at"] = now()
        write_json(output / "status.json", state)
        write_ablation_report(output)


if __name__ == "__main__":
    main()
