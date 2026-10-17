"""Compare the final agent-selected outcomes with fixed-parameter baselines."""

import argparse
import csv
import json
from pathlib import Path

from examples.qtrade_fin_quant.run import write_json


METRICS = {
    "IC": "IC",
    "Rank_IC": "Rank IC",
    "annualized_excess_return_after_costs": "1day.excess_return_with_cost.annualized_return",
    "excess_return_information_ratio_after_costs": "1day.excess_return_with_cost.information_ratio",
    "excess_return_max_drawdown_after_costs": "1day.excess_return_with_cost.max_drawdown",
}


def write_report(suite_dir: Path, baseline_dir: Path | None = None) -> dict:
    state = json.loads((suite_dir / "status.json").read_text())
    runs_dir = suite_dir.parent.parent / "runs"
    rows = []
    if baseline_dir and (baseline_dir / "median-metrics.json").exists():
        for name, metrics in json.loads((baseline_dir / "median-metrics.json").read_text()).items():
            rows.append({"method": f"LightGBM/{name}", "selection": "per-metric median over configured seeds",
                         **{name: metrics.get(key) for name, key in METRICS.items()}})
    for phase in state["phases"]:
        results_file = runs_dir / phase["run_id"] / "results.json"
        if not results_file.exists():
            continue
        trials = json.loads(results_file.read_text())
        accepted = [trial for trial in trials if trial["accepted"] and trial["metrics"]]
        if accepted:
            metrics = accepted[-1]["metrics"]
            selection = f"last accepted iteration {accepted[-1]['iteration']} of {len(trials)}"
        else:
            metrics = next((trial.get("reference_metrics") for trial in trials if trial.get("reference_metrics")), {})
            selection = "baseline retained; no accepted candidate"
        rows.append({"method": f"RD-Agent/{phase['mode']}", "selection": selection,
                     **{name: metrics.get(key) for name, key in METRICS.items()}})
    report = {
        "state": state["state"],
        "baseline_run": str(baseline_dir) if baseline_dir else None,
        "rows": rows,
        "interpretation": [
            "Public-data method reproduction with substituted chat/embedding models",
            "One research trajectory per mode, versus five fixed-parameter baseline seeds",
            "All portfolio metrics use the released Qlib costed excess-return convention",
            "Backtest statistics were exposed during research; not a sealed holdout",
            "These rows do not reproduce the complete paper baseline table or its HPO",
        ],
    }
    write_json(suite_dir / "comparison.json", report)
    with (suite_dir / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "selection", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_report(args.suite_dir.resolve(), args.baseline_dir), indent=2))


if __name__ == "__main__":
    main()
