import json

from examples.qtrade_fin_quant.report import write_report


def write_run(tmp_path, trials):
    suite_dir = tmp_path / "suites" / "study"
    run_dir = tmp_path / "runs" / "study-quant"
    suite_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (suite_dir / "status.json").write_text(json.dumps({
        "state": "finished", "phases": [{"mode": "quant", "run_id": "study-quant"}],
    }))
    (run_dir / "results.json").write_text(json.dumps(trials))
    return suite_dir


def test_report_does_not_select_rejected_high_return_trial(tmp_path):
    key = "1day.excess_return_with_cost.annualized_return"
    suite_dir = write_run(tmp_path, [
        {"iteration": 0, "accepted": True, "metrics": {key: 0.04}},
        {"iteration": 1, "accepted": False, "metrics": {key: 0.90}},
        {"iteration": 2, "accepted": True, "metrics": {key: 0.03}},
    ])
    row = write_report(suite_dir)["rows"][0]
    assert row["annualized_excess_return_after_costs"] == 0.03
    assert row["selection"] == "last accepted iteration 2 of 3"


def test_report_retains_reference_when_all_candidates_are_rejected(tmp_path):
    key = "1day.excess_return_with_cost.annualized_return"
    suite_dir = write_run(tmp_path, [
        {"iteration": 0, "accepted": False, "metrics": {key: -0.10}, "reference_metrics": {key: 0.02}},
    ])
    row = write_report(suite_dir)["rows"][0]
    assert row["annualized_excess_return_after_costs"] == 0.02
    assert row["selection"] == "baseline retained; no accepted candidate"
