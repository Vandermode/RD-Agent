"""Report every backend-ablation replicate without selecting favorable trials."""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median, stdev

from examples.qtrade_fin_quant.report import METRICS
from examples.qtrade_fin_quant.run import write_json


def selected_outcome(trials: list[dict]) -> tuple[dict, str]:
    accepted = [trial for trial in trials if trial.get("accepted") and trial.get("metrics")]
    if accepted:
        selected = accepted[-1]
        return selected["metrics"], f"last accepted iteration {selected['iteration']}"
    reference = next((trial.get("reference_metrics") for trial in trials if trial.get("reference_metrics")), {})
    return reference, "baseline retained; no accepted candidate"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_ablation_report(study_dir: Path) -> dict:
    state = json.loads((study_dir / "status.json").read_text())
    runs_dir = study_dir.parent.parent / "runs"
    rows = []
    for job in state["jobs"]:
        if job["kind"] != "research":
            continue
        run = runs_dir / job["run_id"]
        trials = json.loads((run / "results.json").read_text()) if (run / "results.json").exists() else []
        metrics, selection = selected_outcome(trials)
        log = study_dir / f"{job['run_id']}.console.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        rows.append({
            "model": job["model"], "scheduler_seed": job["seed"], "run_id": job["run_id"],
            "state": job["state"], "budget_hours": state["hours_per_run"],
            "elapsed_seconds": job.get("elapsed_seconds"),
            "recorded_iterations": len(trials),
            "successful_backtests": sum(bool(trial.get("metrics")) for trial in trials),
            "accepted_iterations": sum(bool(trial.get("accepted")) for trial in trials),
            "logged_chat_completions": text.count("Using chat model"),
            "selection": selection if trials else "no completed result yet",
            **{name: metrics.get(key) for name, key in METRICS.items()},
        })

    aggregate = []
    for model in state["models"]:
        model_rows = [row for row in rows if row["model"] == model]
        completed = [row for row in model_rows if row["state"] == "finished"]
        for metric in [*METRICS, "recorded_iterations", "successful_backtests", "accepted_iterations",
                       "logged_chat_completions"]:
            values = [float(row[metric]) for row in completed
                      if row.get(metric) is not None and math.isfinite(float(row[metric]))]
            aggregate.append({
                "model": model, "metric": metric, "completed_runs": len(completed),
                "failed_runs": sum(row["state"] == "failed" for row in model_rows),
                "planned_runs": len(model_rows), "valid_values": len(values),
                "mean": mean(values) if values else None,
                "median": median(values) if values else None,
                "sample_std": stdev(values) if len(values) > 1 else None,
                "min": min(values) if values else None, "max": max(values) if values else None,
            })
    baseline_path = Path(state["baseline_dir"]) / "median-metrics.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    report = {
        "state": state["state"], "models": state["models"], "runs": rows, "aggregate": aggregate,
        "baseline_medians": baseline,
        "protocol": {
            "mode": "joint factor/model research", "hours_per_run": state["hours_per_run"],
            "scheduler_seeds": state["seeds"], "reasoning_effort": state["reasoning_effort"],
            "training_seeds": "Upstream defaults; controller seeds do not seed model subprocesses",
            "selection": "Last accepted candidate in each run; never the largest observed test return",
            "aggregation": "All finished replicates; failures and missing runs reported explicitly",
            "interpretation": [
                "Three trajectories per backend describe variability; they do not establish a definitive model ranking",
                "Research consumes backtest feedback; these are not sealed-holdout results",
                "Portfolio metrics are Qlib excess returns over CSI300 after costs",
                "Equal wall-clock budgets include model API and training latency",
                "Reported chat completions are log counts, not token usage or billed API cost",
            ],
        },
    }
    write_json(study_dir / "comparison.json", report)
    write_csv(study_dir / "runs.csv", rows, ["model", "scheduler_seed", "run_id", "state", "budget_hours",
              "elapsed_seconds", "recorded_iterations", "successful_backtests", "accepted_iterations",
              "logged_chat_completions", "selection", *METRICS])
    write_csv(study_dir / "summary.csv", aggregate, ["model", "metric", "completed_runs", "failed_runs",
              "planned_runs", "valid_values", "mean", "median", "sample_std", "min", "max"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    args = parser.parse_args()
    report = write_ablation_report(args.study_dir.resolve())
    print(json.dumps({"state": report["state"], "research_runs": len(report["runs"])}, indent=2))


if __name__ == "__main__":
    main()
