import json

from examples.qtrade_fin_quant.ablation import DEFAULT_MODELS, build_jobs
from examples.qtrade_fin_quant.ablation_report import selected_outcome, write_ablation_report


def test_ablation_balances_models_seeds_and_execution_position():
    jobs = build_jobs("study", DEFAULT_MODELS, [0, 1, 2])
    assert len(jobs) == 12
    assert all(job["kind"] == "preflight" for job in jobs[:3])
    research = jobs[3:]
    assert {(job["model"], job["seed"]) for job in research} == {
        (model, seed) for model in DEFAULT_MODELS for seed in [0, 1, 2]
    }
    for position in range(3):
        assert {research[block * 3 + position]["model"] for block in range(3)} == set(DEFAULT_MODELS)
    assert len({job["run_id"] for job in jobs}) == len(jobs)


def test_ablation_selects_last_accepted_instead_of_highest_test_result():
    trials = [
        {"iteration": 0, "accepted": True, "metrics": {"IC": 0.1}},
        {"iteration": 1, "accepted": False, "metrics": {"IC": 0.9}},
        {"iteration": 2, "accepted": True, "metrics": {"IC": 0.05}},
    ]
    assert selected_outcome(trials) == ({"IC": 0.05}, "last accepted iteration 2")


def test_ablation_reports_failed_replicates_without_counting_them_as_completed(tmp_path):
    study = tmp_path / "ablations/study"
    study.mkdir(parents=True)
    baseline = tmp_path / "runs/baseline"
    baseline.mkdir(parents=True)
    (baseline / "median-metrics.json").write_text("{}")
    jobs = []
    for seed, (state, value) in enumerate([("finished", 0.02), ("finished", 0.04), ("failed", 0.99)]):
        run_id = f"study-luna-s{seed}"
        run = tmp_path / "runs" / run_id
        run.mkdir()
        (run / "results.json").write_text(json.dumps([
            {"iteration": 0, "accepted": True, "metrics": {"IC": value}}
        ]))
        jobs.append({"kind": "research", "model": "gpt-6-luna", "seed": seed,
                     "state": state, "run_id": run_id})
    (study / "status.json").write_text(json.dumps({
        "state": "finished_with_failures", "jobs": jobs, "models": ["gpt-6-luna"],
        "seeds": [0, 1, 2], "hours_per_run": 4, "reasoning_effort": "medium",
        "baseline_dir": str(baseline),
    }))
    report = write_ablation_report(study)
    ic = next(row for row in report["aggregate"] if row["metric"] == "IC")
    assert ic["completed_runs"] == 2
    assert ic["failed_runs"] == 1
    assert ic["planned_runs"] == 3
    assert ic["mean"] == 0.03
    assert ic["median"] == 0.03
    assert len(report["runs"]) == 3
