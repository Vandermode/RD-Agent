"""Keep serial reference trajectories and throughput-selected runs separate."""

import csv
import math
from pathlib import Path
from statistics import mean, median, stdev

from examples.qtrade_fin_quant.ablation_report import selected_outcome
from examples.qtrade_fin_quant.parallel_common import read_json, write_json
from examples.qtrade_fin_quant.report import METRICS


def sharpe_from_selected(trials: list[dict]) -> dict:
    accepted = [t for t in trials if t.get("accepted") and t.get("metrics")]
    if not accepted:
        return {"portfolio_sharpe_after_costs_rf0": None, "sharpe_status": "no selected candidate workspace"}
    selected = accepted[-1]
    receipt = selected.get("evaluation") or {}
    try:
        import pandas as pd

        workspace = Path(selected["workspace"])
        recorder = read_json(workspace / "recorder.json", {}).get("recorder_id")
        if receipt.get("state") != "finished" or recorder != receipt.get("recorder_id"):
            raise ValueError("Selected workspace is not the verified recorder")
        returns = pd.read_parquet(workspace / "ret.parquet")
        values = returns[["return", "cost", "bench"]]
        if not values.map(math.isfinite).all().all():
            raise ValueError("Nonfinite daily returns")
        net = values["return"] - values["cost"]
        excess = net - values["bench"]
        information_ratio = excess.mean() / excess.std(ddof=1) * math.sqrt(238)
        reported_ir = selected["metrics"][METRICS["excess_return_information_ratio_after_costs"]]
        if not math.isclose(information_ratio, reported_ir, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Return-series information ratio differs from the selected result")
        if net.std(ddof=1) <= 0:
            raise ValueError("Daily net return variance is zero")
        return {
            "portfolio_sharpe_after_costs_rf0": float(net.mean() / net.std(ddof=1) * math.sqrt(238)),
            "sharpe_status": "verified selected recorder",
            "selected_recorder_id": recorder,
            "daily_return_observations": len(net),
        }
    except Exception as error:
        return {"portfolio_sharpe_after_costs_rf0": None, "sharpe_status": str(error)}


def run_row(job: dict, cohort: str, hours: int, concurrency: int) -> dict:
    directory = Path(job["run_dir"])
    trials = read_json(directory / "results.json", [])
    status = read_json(directory / "status.json", {})
    metrics, selection = selected_outcome(trials)
    return {
        "protocol": cohort,
        "model": job["model"],
        "scheduler_seed": job["seed"],
        "run_id": job["run_id"],
        "run_dir": str(directory),
        "state": job["state"],
        "run_state": status.get("state", "not_started"),
        "budget_hours": hours,
        "scheduled_concurrency": concurrency,
        "wave": job.get("wave"),
        "elapsed_seconds": job.get("elapsed_seconds"),
        "exit_code": job.get("exit_code"),
        "recorded_iterations": len(trials),
        "successful_backtests": sum(bool(t.get("metrics")) for t in trials),
        "accepted_iterations": sum(bool(t.get("accepted") and t.get("metrics")) for t in trials),
        "verified_backtests": sum((t.get("evaluation") or {}).get("state") == "finished" for t in trials),
        "schema_rejected_iterations": sum(
            bool(t.get("feedback_schema_error") or t.get("proposal_schema_error")) for t in trials
        ),
        "selection": selection if trials else "no completed result yet",
        **{name: metrics.get(key) for name, key in METRICS.items()},
        **sharpe_from_selected(trials),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for model in sorted({row["model"] for row in rows}):
        group = [row for row in rows if row["model"] == model]
        finished = [row for row in group if row["state"] == "finished"]
        for metric in [
            *METRICS,
            "portfolio_sharpe_after_costs_rf0",
            "recorded_iterations",
            "successful_backtests",
            "accepted_iterations",
            "verified_backtests",
        ]:
            values = [
                float(row[metric])
                for row in finished
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            output.append(
                {
                    "model": model,
                    "metric": metric,
                    "planned_runs": len(group),
                    "completed_runs": len(finished),
                    "failed_runs": sum(row["state"] in {"failed", "interrupted"} for row in group),
                    "valid_values": len(values),
                    "mean": mean(values) if values else None,
                    "median": median(values) if values else None,
                    "sample_std": stdev(values) if len(values) > 1 else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_protocol_reports(directory: Path, state: dict) -> dict:
    serial = [run_row(job, "serial_seed0", state["hours_per_run"], 1) for job in state["serial_runs"]]
    parallel = [
        run_row(
            job,
            state.get("parallel_cohort", "throughput_selected_seeds1_2"),
            state["hours_per_run"],
            state.get("selected_concurrency"),
        )
        for job in state["jobs"]
    ]
    report = {
        "state": state["state"],
        "serial_reference": {"runs": serial, "aggregate": aggregate(serial)},
        "parallel_study": {"runs": parallel, "aggregate": aggregate(parallel)},
        "baseline_medians": read_json(Path(state["baseline_dir"]) / "median-metrics.json", {}),
        "protocol": {
            "version": 3,
            "serial_source_commit": state["serial_source_commit"],
            "parallel_source_commit": state["git_commit"],
            "hours_per_run": state["hours_per_run"],
            "reasoning_effort": "medium",
            "temperature": 1.0,
            "max_output_tokens": 10000,
            "selected_concurrency": state.get("selected_concurrency"),
            "aggregation": "Serial seed 0 and fresh seeds 1/2 are never pooled",
            "selection": "Last accepted candidate; original reference metrics if none accepted",
            "budget_policy": "Four-hour wall-clock budget includes API retries; finish current stage only; five-hour hard limit",
            "memory": "Fresh interpreter, knowledge stores, prompt/pickle caches, workspace, temporary and local caches per run",
            "isolation_boundary": "Application paths; not a Unix filesystem access-control sandbox",
            "benchmark": state.get(
                "benchmark_protocol",
                "Two fixed-workload rounds per level, order 1/3/6/6/3/1; full data; neural epoch cap 5",
            ),
            "concurrency_selection": state.get("concurrency_selection", "Highest valid pooled benchmark throughput"),
            "throughput": "Valid completed Qlib evaluations per elapsed hour, including launches and extraction",
            "sharpe": "Net portfolio return = return - cost; risk-free rate 0; sqrt(238); sample standard deviation (ddof=1)",
            "interpretation": [
                "Benchmark measurements select evaluation concurrency; they do not measure remote LLM throughput",
                "Parallel resource sharing can change research latency and the path explored within four hours",
                "One serial and two new trajectories per backend do not establish a definitive ranking",
                "Scheduler seeds do not seed training subprocesses or make remote responses deterministic",
                "Backtest statistics participate in research; these are not sealed-holdout results",
            ],
        },
    }
    write_json(directory / "comparison.json", report)
    write_csv(directory / "serial-runs.csv", serial)
    write_csv(directory / "serial-summary.csv", report["serial_reference"]["aggregate"])
    write_csv(directory / "parallel-runs.csv", parallel)
    write_csv(directory / "parallel-summary.csv", report["parallel_study"]["aggregate"])
    return report
