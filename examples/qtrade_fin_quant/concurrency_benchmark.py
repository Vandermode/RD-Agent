"""Compare fresh, fixed Qlib evaluation workloads at 1/3/6 processes."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from examples.qtrade_fin_quant.parallel_common import now, read_json, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(directory: Path, names: list[str]) -> dict:
    return {name: sha256(directory / name) for name in names}


def freeze_fixtures(serial_runs: list[dict], destination: Path, epochs: int = 5) -> list[dict]:
    """Copy inputs only. Benchmarks never import an agent's research memory."""
    import pandas as pd
    import yaml
    from jinja2 import Template

    destination.mkdir(parents=True, exist_ok=False)
    fixtures = []
    for run in serial_runs:
        trials = read_json(Path(run["run_dir"]) / "results.json", [])
        accepted = [t for t in trials if t.get("accepted") and t.get("metrics")]
        candidates = accepted or [t for t in trials if t.get("metrics")]
        if not candidates:
            raise RuntimeError(f"No successful evaluation for {run['model']}")
        trial = candidates[-1]
        receipt = trial.get("evaluation") or {}
        if receipt.get("state") != "finished":
            raise RuntimeError("Benchmark inputs require an audited successful evaluation")
        source = Path(trial["workspace"])
        source.relative_to(Path(run["run_dir"]) / "workspaces")
        config_name = receipt["config"]
        if Path(config_name).name != config_name:
            raise RuntimeError("Unexpected Qlib configuration path")
        input_names = [config_name]
        for name in ("model.py", "combined_factors_df.parquet"):
            if name in receipt["inputs_before"]:
                input_names.append(name)
        hashes = snapshot(source, input_names)
        expected = {name: receipt["inputs_before"][name]["sha256"] for name in input_names}
        if hashes != expected or receipt["inputs_before"] != receipt["inputs_after"]:
            raise RuntimeError("Serial candidate inputs no longer match the successful evaluation")
        config = yaml.safe_load(Template((source / config_name).read_text()).render(receipt["run_env"]))
        model = config["task"]["model"]
        if model["class"] not in {"GeneralPTNN", "LGBModel"}:
            raise RuntimeError(f"Unsupported benchmark model: {model['class']}")
        source_returns = pd.read_parquet(source / "ret.parquet")
        expected_test = {
            "rows": len(source_returns),
            "start": str(source_returns.index.min()),
            "end": str(source_returns.index.max()),
        }
        # Same full training/validation/test data; the short benchmark measures
        # candidate evaluation throughput, not remote LLM or four-hour research.
        for replicate in range(2):
            fixture_id = f"{run['model']}-rep{replicate}"
            directory = destination / fixture_id
            directory.mkdir()
            for name in input_names:
                if name != config_name:
                    shutil.copyfile(source / name, directory / name)
            shutil.copyfile(source / "read_exp_res.py", directory / "read_exp_res.py")
            config_copy = json.loads(json.dumps(config, default=str))
            kwargs = config_copy["task"]["model"]["kwargs"]
            kwargs["seed"] = 202600 + replicate
            if model["class"] == "GeneralPTNN":
                kwargs["n_epochs"] = min(epochs, int(kwargs["n_epochs"]))
                kwargs["early_stop"] = kwargs["n_epochs"]
            (directory / "benchmark.yaml").write_text(yaml.safe_dump(config_copy, sort_keys=False))
            names = ["benchmark.yaml", "read_exp_res.py", *[n for n in input_names if n != config_name]]
            fixture = {
                "fixture_id": fixture_id,
                "directory": str(directory),
                "model": run["model"],
                "source_run": run["run_id"],
                "source_iteration": trial["iteration"],
                "source_workspace": str(source),
                "source_hashes": hashes,
                "source_recorder_id": receipt["recorder_id"],
                "source_selection": "last accepted" if accepted else "last successful (none accepted)",
                "training_seed": kwargs["seed"],
                "model_class": model["class"],
                "epoch_cap": epochs if model["class"] == "GeneralPTNN" else None,
                "input_hashes": snapshot(directory, names),
                "expected_test": expected_test,
            }
            write_json(directory / "fixture.json", fixture)
            fixtures.append(fixture)
    write_json(destination / "manifest.json", fixtures)
    return fixtures


def prepare_workspace(fixture: dict, workspace: Path) -> dict:
    workspace.mkdir(parents=True, exist_ok=False)
    source = Path(fixture["directory"])
    if snapshot(source, list(fixture["input_hashes"])) != fixture["input_hashes"]:
        raise RuntimeError("Frozen benchmark inputs changed")
    for name in fixture["input_hashes"]:
        if Path(name).name != name:
            raise RuntimeError("Unexpected benchmark input path")
        shutil.copyfile(source / name, workspace / name)
    # In particular: never copy mlruns, model weights, results, or checkpoints.
    return snapshot(workspace, list(fixture["input_hashes"]))


def evaluate(fixture: dict, workspace: Path) -> dict:
    """Execute a fresh qrun and verify the recorder, inputs and return series."""
    import pandas as pd

    before = prepare_workspace(fixture, workspace)
    started = time.monotonic()
    result = {
        "state": "running",
        "started_at": now(),
        "inputs_before": before,
        "fixture_id": fixture["fixture_id"],
        "workspace": str(workspace),
    }
    write_json(workspace / "evaluation.json", result)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workspace) + os.pathsep + env.get("PYTHONPATH", "")
    # qrun uses a local file store by default, independent of other processes.
    try:
        qrun = Path(sys.executable).parent / "qrun"
        for label, command in [
            ("qrun", [str(qrun), "benchmark.yaml"]),
            ("extraction", [sys.executable, "read_exp_res.py"]),
        ]:
            with (workspace / f"{label}.log").open("x") as log:
                completed = subprocess.run(
                    command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            result[label + "_exit_code"] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(f"{label} exited with code {completed.returncode}")
        after = snapshot(workspace, list(before))
        result["inputs_after"] = after
        if before != after:
            raise RuntimeError("Benchmark inputs changed during execution")
        recorder_id = read_json(workspace / "recorder.json", {}).get("recorder_id")
        recorders = list(workspace.glob(f"mlruns/*/{recorder_id}/meta.yaml"))
        if not recorder_id or recorder_id == fixture["source_recorder_id"] or len(recorders) != 1:
            raise RuntimeError("No unique fresh MLflow recorder")
        import yaml

        metadata = yaml.safe_load(recorders[0].read_text())
        if metadata.get("status") != 3:  # MLflow FINISHED
            raise RuntimeError("The MLflow recorder did not finish")
        metrics = pd.read_csv(workspace / "qlib_res.csv", index_col=0).iloc[:, 0].to_dict()
        from examples.qtrade_fin_quant.report import METRICS

        if any(not math.isfinite(float(metrics.get(key, float("nan")))) for key in METRICS.values()):
            raise RuntimeError("Missing or nonfinite backtest metrics")
        returns = pd.read_parquet(workspace / "ret.parquet")
        values = returns[["return", "cost", "bench"]]
        if len(values) < 100 or not values.map(math.isfinite).all().all() or not values.index.is_unique:
            raise RuntimeError("Missing, nonfinite or duplicate daily backtest returns")
        observed_test = {"rows": len(returns), "start": str(returns.index.min()), "end": str(returns.index.max())}
        if observed_test != fixture["expected_test"]:
            raise RuntimeError("Backtest did not cover the fixture's complete test interval")
        result.update(
            state="finished",
            recorder_id=recorder_id,
            metrics=metrics,
            return_rows=len(returns),
            test_start=str(returns.index.min()),
            test_end=str(returns.index.max()),
        )
    except Exception as error:
        result.update(state="failed", error=str(error))
    result.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
    write_json(workspace / "evaluation.json", result)
    return result


def rank_configurations(rounds: list[dict], fixtures: list[dict]) -> list[dict]:
    """Only complete, error-free repeated workloads are eligible."""
    expected_ids = sorted(f["fixture_id"] for f in fixtures)
    summaries = []
    all_recorders = [
        j.get("result", {}).get("recorder_id")
        for r in rounds
        for j in r["jobs"]
        if j.get("result", {}).get("recorder_id")
    ]
    duplicate_recorders = {r for r in all_recorders if all_recorders.count(r) != 1}
    for concurrency in (1, 3, 6):
        group = [r for r in rounds if r["concurrency"] == concurrency]
        jobs = [j for r in group for j in r["jobs"]]
        elapsed = sum(r.get("elapsed_seconds", 0) for r in group)
        valid = [
            j
            for j in jobs
            if j["state"] == "finished"
            and j.get("exit_code") == 0
            and j.get("result", {}).get("state") == "finished"
            and j["result"].get("recorder_id")
            and j["result"]["recorder_id"] not in duplicate_recorders
            and j["result"].get("inputs_before")
            == j["result"].get("inputs_after")
            == next(f["input_hashes"] for f in fixtures if f["fixture_id"] == j["fixture_id"])
        ]
        eligible = (
            len(group) == 2
            and all(r.get("state") == "finished" for r in group)
            and all(sorted(j["fixture_id"] for j in r["jobs"]) == expected_ids for r in group)
            and len(valid) == 2 * len(fixtures)
            and elapsed > 0
        )
        latencies = sorted(j.get("elapsed_seconds", 0) for j in jobs)
        summaries.append(
            {
                "concurrency": concurrency,
                "eligible": eligible,
                "completed_rounds": len(group),
                "valid_evaluations": len(valid),
                "planned_evaluations": 2 * len(fixtures),
                "failed_evaluations": sum(j["state"] == "failed" for j in jobs),
                "elapsed_seconds": elapsed,
                "valid_evaluations_per_hour": len(valid) * 3600 / elapsed if elapsed else 0,
                "mean_job_seconds": sum(latencies) / len(latencies) if latencies else None,
                "max_job_seconds": max(latencies) if latencies else None,
            }
        )
    return sorted(summaries, key=lambda s: (not s["eligible"], -s["valid_evaluations_per_hour"], s["concurrency"]))


def choose_concurrency(rounds: list[dict], fixtures: list[dict]) -> int:
    ranking = rank_configurations(rounds, fixtures)
    if not ranking[0]["eligible"]:
        raise RuntimeError("No concurrency level completed both benchmark rounds successfully")
    return ranking[0]["concurrency"]


def write_benchmark_report(directory: Path, state: dict) -> None:
    write_json(directory / "benchmark.json", state)
    if state.get("fixtures"):
        ranking = rank_configurations(state["rounds"], state["fixtures"])
        with (directory / "throughput.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(ranking[0]))
            writer.writeheader()
            writer.writerows(ranking)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(read_json(args.fixture), args.workspace)
    print(json.dumps({key: result.get(key) for key in ("state", "recorder_id", "elapsed_seconds", "error")}))
    if result["state"] != "finished":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
