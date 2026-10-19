"""Wait for the serial Sol run, benchmark concurrency, then run fresh replicas."""

import argparse
import csv
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from examples.qtrade_fin_quant.ablation import DEFAULT_MODELS, source_commit
from examples.qtrade_fin_quant.concurrency_benchmark import (
    choose_concurrency,
    freeze_fixtures,
    rank_configurations,
    write_benchmark_report,
)
from examples.qtrade_fin_quant.parallel_common import (
    group_members,
    identity,
    isolated_env,
    now,
    read_json,
    run_pool,
    same_process,
    verify_ipc_environment,
    write_json,
)
from examples.qtrade_fin_quant.parallel_report import write_protocol_reports


def remaining_jobs(study_id: str, runs_dir: Path) -> list[dict]:
    jobs = []
    for seed in (1, 2):
        order = DEFAULT_MODELS[seed:] + DEFAULT_MODELS[:seed]
        for model in order:
            run_id = f"{study_id}-{model}-s{seed}"
            jobs.append(
                {
                    "kind": "research",
                    "model": model,
                    "seed": seed,
                    "state": "queued",
                    "run_id": run_id,
                    "run_dir": str(runs_dir / run_id),
                }
            )
    return jobs


def handoff_gate(handoff: dict, supervisor: dict | None, sol: dict | None, status: dict) -> str:
    """Pure gate: a finished status alone does not authorize overlap with Sol."""
    if not same_process(handoff["serial_supervisor"], supervisor) or supervisor["state"] != "T":
        raise RuntimeError("The legacy supervisor is missing, resumed or has changed identity")
    if not same_process(handoff["sol_timeout"], sol):
        raise RuntimeError("The serial Sol process identity is unavailable or has changed")
    if sol["ppid"] != supervisor["pid"]:
        raise RuntimeError("Sol is no longer the frozen supervisor's child")
    if sol["state"] == "T":
        raise RuntimeError("The serial Sol worker is unexpectedly paused")
    if sol["state"] != "Z":
        return "waiting"
    if sol["wait_status"] != 0 or status.get("state") != "finished" or not status.get("successful_backtests"):
        raise RuntimeError("Serial Sol did not finish successfully; benchmark and new research were not started")
    return "ready"


def finalize_handoff(legacy_study: Path, handoff: dict, state: dict) -> None:
    sol = identity(handoff["sol_timeout"]["pid"])
    supervisor = identity(handoff["serial_supervisor"]["pid"])
    sol_dir = legacy_study.parent.parent / "runs" / handoff["sol_run_id"]
    if handoff_gate(handoff, supervisor, sol, read_json(sol_dir / "status.json", {})) != "ready":
        raise RuntimeError("Sol has not completed")
    if group_members(sol["pgid"]):
        raise RuntimeError("Serial Sol left running descendants; refusing overlapping experiments")
    previous = read_json(legacy_study / "status.json")
    if any(j["state"] != "queued" for j in previous["jobs"] if j["kind"] == "research" and j["seed"] in (1, 2)):
        raise RuntimeError("A remaining serial replica already started")
    # A frozen parent cannot execute its cleanup handler or start another run.
    # Signal only this verified parent PID, after its worker has fully exited.
    os.kill(supervisor["pid"], signal.SIGKILL)
    for _ in range(100):
        process = identity(supervisor["pid"])
        if process is None or process["state"] == "Z":
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("Legacy supervisor did not stop")
    write_json(legacy_study / "status-before-parallel-handoff.json", previous)
    target_jobs = {(job["model"], job["seed"]): job for job in state["jobs"]}
    for job in previous["jobs"]:
        if job["run_id"] == handoff["sol_run_id"]:
            result = read_json(sol_dir / "status.json")
            job.update(
                state="finished",
                exit_code=0,
                result=result,
                finished_at=result["finished_at"],
                elapsed_seconds=(
                    datetime.fromisoformat(result["finished_at"]) - datetime.fromisoformat(job["started_at"])
                ).total_seconds(),
            )
        elif job["kind"] == "research" and job["seed"] in (1, 2):
            if job["state"] != "queued":
                raise RuntimeError("A remaining serial replica already started")
            target = target_jobs[(job["model"], job["seed"])]
            job.update(
                state="transferred_to_parallel",
                replacement_run_id=target["run_id"],
                replacement_run_dir=target["run_dir"],
            )
    previous.update(
        state="handed_off",
        finished_at=now(),
        parallel_study=state["output"],
        handoff_note="Seed 0 is the serial reference; seeds 1/2 moved to a separate protocol and reports",
    )
    write_json(legacy_study / "status.json", previous)
    # Preserve the original job ledger, including its transferred rows.
    from examples.qtrade_fin_quant.ablation_report import write_ablation_report

    write_ablation_report(legacy_study)
    for serial in state["serial_runs"]:
        old = next(job for job in previous["jobs"] if job["run_id"] == serial["run_id"])
        serial.update({key: old[key] for key in ("state", "exit_code", "elapsed_seconds", "finished_at")})
    handoff.update(state="handed_off", completed_at=now(), parallel_study=state["output"])
    write_json(legacy_study / "parallel-handoff.json", handoff)


class Controller:
    def __init__(self, root: Path, output: Path, state: dict, min_free_gib: float):
        self.root, self.output, self.state = root, output, state
        self.min_free_gib = min_free_gib
        self.last_guard = self.last_sample = self.last_report = 0.0
        self.benchmark = None
        self.telemetry = output / "resources.csv"

    def check_benchmark_infrastructure(self):
        if not self.benchmark or self.benchmark["state"] != "running":
            return
        for batch in self.benchmark["rounds"]:
            for job in batch["jobs"]:
                if job["state"] != "running":
                    continue
                log = Path(job["workspace"]) / "qrun.log"
                if not log.is_file():
                    continue
                with log.open("rb") as file:
                    file.seek(max(0, log.stat().st_size - 32768))
                    tail = file.read().decode(errors="replace")
                for error in ("AF_UNIX path too long", "No space left on device", "Too many open files"):
                    if error in tail:
                        job["infrastructure_error"] = error
                        raise RuntimeError(f"Benchmark infrastructure failure in {job['fixture_id']}: {error}")

    def tick(self, force=False):
        current = time.monotonic()
        if current - self.last_guard >= 15 or force:
            self.check_benchmark_infrastructure()
            if source_commit(self.root) != self.state["git_commit"]:
                raise RuntimeError("Parallel source revision changed")
            if shutil.disk_usage(self.root).free < self.min_free_gib * 1024**3:
                raise RuntimeError(f"Free space fell below {self.min_free_gib} GiB")
            self.last_guard = current
        if current - self.last_sample >= 10 or force:
            self.sample_resources()
            self.last_sample = current
        if current - self.last_report >= 30 or force:
            self.state["updated_at"] = now()
            write_json(self.output / "status.json", self.state)
            if self.benchmark is not None:
                write_benchmark_report(self.output / "benchmark", self.benchmark)
            write_protocol_reports(self.output, self.state)
            self.last_report = current

    def sample_resources(self):
        active_jobs = self.state["jobs"]
        if self.benchmark:
            active_jobs = active_jobs + [j for r in self.benchmark["rounds"] for j in r["jobs"]]
        row = {
            "timestamp": now(),
            "phase": self.state["state"],
            "active_workers": sum(j["state"] == "running" for j in active_jobs),
            "cpu_load1": os.getloadavg()[0],
            "gpu_util_percent": None,
            "gpu_memory_used_mib": None,
            "gpu_memory_total_mib": None,
            "gpu_power_w": None,
        }
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.state['gpu']}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=4,
            )
            values = [float(value.strip()) for value in result.strip().split(",")]
            row.update(
                dict(zip(["gpu_util_percent", "gpu_memory_used_mib", "gpu_memory_total_mib", "gpu_power_w"], values))
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass  # Telemetry absence is recorded; it does not corrupt a run.
        exists = self.telemetry.exists()
        with self.telemetry.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def wait_for_serial(self, legacy_study: Path, handoff: dict):
        serial_root = legacy_study.parents[3]
        print(f"{now()} Waiting for serial Sol to exit successfully", flush=True)
        while True:
            if source_commit(serial_root) != self.state["serial_source_commit"]:
                raise RuntimeError("Serial source changed before handoff")
            for job in self.state["serial_runs"]:
                status = read_json(Path(job["run_dir"]) / "status.json", {})
                job["live_result"] = status
            sol_status = read_json(legacy_study.parent.parent / "runs" / handoff["sol_run_id"] / "status.json", {})
            result = handoff_gate(
                handoff,
                identity(handoff["serial_supervisor"]["pid"]),
                identity(handoff["sol_timeout"]["pid"]),
                sol_status,
            )
            self.tick()
            if result == "ready":
                break
            time.sleep(10)
        finalize_handoff(legacy_study, handoff, self.state)
        self.state["handoff_completed_at"] = now()
        print(f"{now()} Serial handoff complete; queued replicas transferred", flush=True)
        self.tick(force=True)

    def popen(self, command: list[str], runtime_id: str, log_path: Path) -> subprocess.Popen:
        env = isolated_env(self.root, self.output / "runtime" / runtime_id, self.state["gpu"])
        verify_ipc_environment(env)
        # configure() sets the per-run research paths; this wrapper additionally
        # isolates temporary caches and removes inherited memory settings.
        with log_path.open("x") as log:
            return subprocess.Popen(
                command,
                cwd=self.root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

    def run_benchmark(self) -> int:
        self.state["state"] = "benchmarking"
        directory = self.output / "benchmark"
        directory.mkdir()
        fixtures = freeze_fixtures(self.state["serial_runs"], directory / "fixtures")
        self.benchmark = {
            "state": "running",
            "started_at": now(),
            "fixtures": fixtures,
            "rounds": [],
            "order": [1, 3, 6, 6, 3, 1],
            "epoch_cap": 5,
            "scope": "Fresh full-data Qlib training, prediction and backtest; excludes remote LLM",
            "selection": "Highest pooled valid evaluations/hour among levels passing both rounds",
        }
        for index, concurrency in enumerate(self.benchmark["order"], 1):
            print(f"{now()} Benchmark batch {index}/6, concurrency {concurrency}", flush=True)
            round_dir = directory / f"round-{index}-c{concurrency}"
            round_dir.mkdir()
            jobs = [
                {
                    "state": "queued",
                    "fixture_id": f["fixture_id"],
                    "fixture": f,
                    "workspace": str(round_dir / f["fixture_id"]),
                }
                for f in fixtures
            ]
            if index > 3:
                jobs.reverse()
            entry = {"round": index, "concurrency": concurrency, "started_at": now(), "state": "running", "jobs": jobs}
            self.benchmark["rounds"].append(entry)
            started = time.monotonic()

            def launch(job):
                command = [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=30s",
                    "45m",
                    sys.executable,
                    "-u",
                    "-m",
                    "examples.qtrade_fin_quant.concurrency_benchmark",
                    "--fixture",
                    str(Path(job["fixture"]["directory"]) / "fixture.json"),
                    "--workspace",
                    job["workspace"],
                ]
                job["command"] = command
                return self.popen(
                    command, f"bench-r{index}-{job['fixture_id']}", round_dir / f"{job['fixture_id']}.console.log"
                )

            def validate(job):
                job["result"] = read_json(Path(job["workspace"]) / "evaluation.json", {})
                return job["exit_code"] == 0 and job["result"].get("state") == "finished"

            run_pool(jobs, concurrency, launch, validate, self.tick)
            entry.update(state="finished", finished_at=now(), elapsed_seconds=time.monotonic() - started)
            self.tick(force=True)
            if concurrency == 1 and not any(job["state"] == "finished" for job in jobs):
                raise RuntimeError(
                    "No serial benchmark task succeeded; refusing to repeat a broken workload at higher concurrency"
                )
        selected = choose_concurrency(self.benchmark["rounds"], fixtures)
        self.benchmark.update(
            state="finished",
            finished_at=now(),
            selected_concurrency=selected,
            ranking=rank_configurations(self.benchmark["rounds"], fixtures),
        )
        self.state["selected_concurrency"] = selected
        print(f"{now()} Selected concurrency {selected} from completed benchmark", flush=True)
        self.tick(force=True)
        return selected

    def run_research(self, concurrency: int):
        self.state["state"] = "research_running"
        # Cohort barriers retain one backend per wave at c=3. At c=6 all six
        # start together; at c=1 the balanced seed order is preserved.
        for offset in range(0, len(self.state["jobs"]), concurrency):
            jobs = self.state["jobs"][offset : offset + concurrency]
            print(
                f"{now()} Starting research wave {offset // concurrency + 1}: "
                + ", ".join(job["run_id"] for job in jobs),
                flush=True,
            )
            for job in jobs:
                job["wave"] = offset // concurrency + 1

            def launch(job):
                if Path(job["run_dir"]).exists():
                    raise RuntimeError("A research run directory already exists; refusing memory reuse")
                command = [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=30s",
                    "5h",
                    sys.executable,
                    "-u",
                    "-m",
                    "examples.qtrade_fin_quant.run",
                    "--mode",
                    "quant",
                    "--duration",
                    "4h",
                    "--run-id",
                    job["run_id"],
                    "--gpu",
                    self.state["gpu"],
                    "--model",
                    job["model"],
                    "--scheduler-seed",
                    str(job["seed"]),
                    "--reasoning-effort",
                    "medium",
                ]
                job["command"] = command
                return self.popen(command, job["run_id"], self.output / f"{job['run_id']}.console.log")

            def validate(job):
                job["result"] = read_json(Path(job["run_dir"]) / "status.json", {})
                return job["exit_code"] == 0 and job["result"].get("state") == "finished"

            run_pool(jobs, concurrency, launch, validate, self.tick)
        self.state["state"] = (
            "finished" if all(j["state"] == "finished" for j in self.state["jobs"]) else "finished_with_failures"
        )


def validate_legacy(legacy_study: Path, handoff: dict) -> dict:
    old = read_json(legacy_study / "status.json")
    if old["git_commit"] != handoff["serial_source_commit"] or old["hours_per_run"] != 4:
        raise RuntimeError("Unexpected serial protocol")
    if old["models"] != DEFAULT_MODELS or old["seeds"] != [0, 1, 2] or old["reasoning_effort"] != "medium":
        raise RuntimeError("Expected the three-backend, three-seed medium-reasoning study")
    for model in DEFAULT_MODELS:
        preflight = next(j for j in old["jobs"] if j["kind"] == "preflight" and j["model"] == model)
        verification = read_json(
            legacy_study.parent.parent / "runs" / preflight["run_id"] / "preflight-verification.json", {}
        )
        if preflight["state"] != "finished" or verification.get("passed") is not True:
            raise RuntimeError(f"{model} has no successful serial preflight")
        job = next(j for j in old["jobs"] if j["kind"] == "research" and j["model"] == model and j["seed"] == 0)
        if job["run_id"] != handoff["sol_run_id"] and job["state"] != "finished":
            raise RuntimeError("The other serial seed-0 runs must already be complete")
    if any(j["state"] != "queued" for j in old["jobs"] if j["kind"] == "research" and j["seed"] in (1, 2)):
        raise RuntimeError("A remaining serial replica already started")
    return old


def validate_benchmark_restart(previous_study: Path, legacy_study: Path, handoff: dict) -> dict:
    """Restart a failed benchmark only; never reuse partially run research."""
    previous = read_json(previous_study / "status.json")
    if not previous or previous["state"] != "failed":
        raise RuntimeError("Only a failed benchmark study can be restarted")
    if handoff["state"] != "handed_off" or Path(previous["legacy_study"]) != legacy_study:
        raise RuntimeError("The serial handoff is not complete for this predecessor")
    if previous["serial_source_commit"] != handoff["serial_source_commit"]:
        raise RuntimeError("Serial source provenance does not match")
    process = identity(previous["controller"]["pid"])
    if same_process(previous["controller"], process) and process["state"] != "Z":
        raise RuntimeError("The previous controller is still running")
    for job in previous["jobs"]:
        if job["state"] != "queued" or Path(job["run_dir"]).exists():
            raise RuntimeError("A previous formal research run already started; do not restart its budget")
    for batch in read_json(previous_study / "benchmark/benchmark.json", {}).get("rounds", []):
        for job in batch["jobs"]:
            expected = job.get("process")
            if not expected:
                continue
            leader = identity(expected["pid"])
            if leader and not same_process(expected, leader):
                continue  # PID/group ID has since been reused by another process.
            if group_members(expected["pgid"]):
                raise RuntimeError("A previous benchmark worker group is still running")
    if {job["model"] for job in previous["serial_runs"]} != set(DEFAULT_MODELS):
        raise RuntimeError("The complete three-backend serial reference is required")
    for job in previous["serial_runs"]:
        status = read_json(Path(job["run_dir"]) / "status.json", {})
        if job["seed"] != 0 or job["state"] != "finished" or status.get("state") != "finished":
            raise RuntimeError("A serial reference is incomplete")
    return previous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--legacy-study", type=Path, required=True)
    parser.add_argument(
        "--restart-benchmark-from",
        type=Path,
        help="Failed parallel study with completed handoff and no formal research runs started",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--min-free-gib", type=float, default=50)
    args = parser.parse_args()
    if not args.study_id or Path(args.study_id).name != args.study_id or args.study_id in {".", ".."}:
        parser.error("study-id must be a single directory name")
    if not args.gpu.isdigit():
        parser.error("Select one physical GPU index")
    root = Path(__file__).resolve().parents[2]
    legacy_study = args.legacy_study.resolve()
    commit = source_commit(root)
    with (legacy_study / "parallel-handoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handoff = read_json(legacy_study / "parallel-handoff.json")
        previous = None
        if args.restart_benchmark_from:
            args.restart_benchmark_from = args.restart_benchmark_from.resolve()
            previous = validate_benchmark_restart(args.restart_benchmark_from, legacy_study, handoff)
            old = read_json(legacy_study / "status.json")
        else:
            if handoff["state"] != "legacy_supervisor_frozen_sol_continues":
                raise RuntimeError("No pending frozen-supervisor handoff")
            old = validate_legacy(legacy_study, handoff)
        if args.gpu != old["gpu"]:
            raise RuntimeError("The continuation must use the same physical GPU")
        output = root / "git_ignore_folder/reproduction/ablations" / args.study_id
        output.mkdir(parents=True, exist_ok=False)
        runs_dir = output.parent.parent / "runs"
        serial = [
            {**j, "run_dir": str(legacy_study.parent.parent / "runs" / j["run_id"])}
            for j in old["jobs"]
            if j["kind"] == "research" and j["seed"] == 0
        ]
        state = {
            "state": "benchmark_pending" if previous else "waiting_for_serial_sol",
            "started_at": now(),
            "output": str(output),
            "controller": identity(os.getpid()),
            "git_commit": commit,
            "serial_source_commit": old["git_commit"],
            "legacy_study": str(legacy_study),
            "gpu": args.gpu,
            "hours_per_run": 4,
            "baseline_dir": old["baseline_dir"],
            "serial_runs": serial,
            "jobs": remaining_jobs(args.study_id, runs_dir),
            "nominal_research_hours": 24,
            "protocol_version": 3,
            "resource_note": "GPU telemetry includes unrelated processes; no global GPU configuration changes",
        }
        if previous:
            state.update(
                predecessor_study=str(args.restart_benchmark_from),
                predecessor_source_commit=previous["git_commit"],
                handoff_completed_at=previous["handoff_completed_at"],
                restart_policy="Fresh benchmark and research directories; predecessor measurements excluded",
            )
        for job in state["jobs"]:
            if Path(job["run_dir"]).exists():
                raise RuntimeError("A planned research run directory already exists")
        controller = Controller(root, output, state, args.min_free_gib)
        handoff.update(parallel_controller=state["controller"], parallel_study=str(output))
        write_json(legacy_study / "parallel-handoff.json", handoff)

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"Parallel controller received signal {signum}")

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        try:
            controller.tick(force=True)
            if previous:
                previous["replacement_study"] = str(output)
                write_json(args.restart_benchmark_from / "status.json", previous)
                print(
                    f"{now()} Restarting benchmark after completed serial handoff; predecessor {args.restart_benchmark_from}",
                    flush=True,
                )
            else:
                controller.wait_for_serial(legacy_study, handoff)
            concurrency = controller.run_benchmark()
            controller.run_research(concurrency)
        except BaseException as error:
            state.update(state="failed", error=str(error))
            if controller.benchmark and controller.benchmark["state"] == "running":
                controller.benchmark.update(state="failed", error=str(error))
            raise
        finally:
            state["finished_at"] = now()
            write_json(output / "status.json", state)
            if controller.benchmark:
                write_benchmark_report(output / "benchmark", controller.benchmark)
            write_protocol_reports(output, state)


if __name__ == "__main__":
    main()
