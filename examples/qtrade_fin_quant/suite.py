"""Supervise sequential paper-budget research runs and record their exit status."""

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from examples.qtrade_fin_quant.run import write_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model")
    parser.add_argument("--min-free-gib", type=float, default=50)
    parser.add_argument("--modes", nargs="+", choices=("quant", "factor", "model"),
                        default=["quant", "factor", "model"])
    args = parser.parse_args()
    if Path(args.suite_id).name != args.suite_id or args.suite_id in {".", ".."}:
        parser.error("suite-id must be a single directory name")
    if len(set(args.modes)) != len(args.modes):
        parser.error("modes must be unique")
    root = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        parser.error("Commit the reproduction source before starting a long suite")
    output = root / "git_ignore_folder/reproduction/suites" / args.suite_id
    output.mkdir(parents=True, exist_ok=False)
    state = {"state": "running", "started_at": now(), "supervisor_pid": os.getpid(),
             "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
             "phases": []}
    child = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for mode in args.modes:
            run_id = f"{args.suite_id}-{mode}"
            duration = "12h" if mode == "quant" else "6h"
            hard_limit = "13h" if mode == "quant" else "7h"
            command = ["timeout", "--signal=TERM", "--kill-after=30s", hard_limit,
                       sys.executable, "-u", "-m", "examples.qtrade_fin_quant.run",
                       "--mode", mode, "--duration", duration, "--run-id", run_id, "--gpu", args.gpu]
            if args.model:
                command.extend(["--model", args.model])
            phase = {"mode": mode, "run_id": run_id, "duration": duration,
                     "hard_limit": hard_limit, "started_at": now(), "command": command}
            state["phases"].append(phase)
            with (output / f"{mode}.console.log").open("x") as log:
                child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
                phase["pid"] = child.pid
                write_json(output / "status.json", state)
                while True:
                    if shutil.disk_usage(root).free < args.min_free_gib * 1024**3:
                        raise RuntimeError(f"Free disk space fell below {args.min_free_gib} GiB")
                    try:
                        phase["exit_code"] = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            child = None
            phase["finished_at"] = now()
            run_dir = root / "git_ignore_folder/reproduction/runs" / run_id
            if (run_dir / "status.json").exists():
                phase["result"] = json.loads((run_dir / "status.json").read_text())
            write_json(output / "status.json", state)
            if phase["exit_code"] != 0:
                raise RuntimeError(f"{mode} exited with code {phase['exit_code']}; subsequent phases not started")
        state["state"] = "finished"
    except BaseException as error:
        state["state"] = "failed"
        state["error"] = str(error)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state["finished_at"] = now()
        write_json(output / "status.json", state)


if __name__ == "__main__":
    main()
