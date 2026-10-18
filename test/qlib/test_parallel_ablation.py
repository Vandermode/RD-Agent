import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from examples.qtrade_fin_quant import parallel_ablation as orchestration
from examples.qtrade_fin_quant import concurrency_benchmark as benchmark
from examples.qtrade_fin_quant.parallel_common import isolated_env, read_json, run_pool, write_json
from examples.qtrade_fin_quant.parallel_report import write_protocol_reports
from examples.qtrade_fin_quant.report import METRICS


def process(pid, state="S", ppid=1):
    return {
        "pid": pid,
        "state": state,
        "ppid": ppid,
        "pgid": pid,
        "session": pid,
        "start_ticks": pid * 100,
        "uid": os.getuid(),
        "wait_status": 0,
    }


def test_handoff_waits_for_process_exit_even_after_finished_status():
    parent, sol = process(100, "T"), process(101, "S", 100)
    handoff = {"serial_supervisor": parent.copy(), "sol_timeout": sol.copy()}
    status = {"state": "finished", "successful_backtests": 3}
    assert orchestration.handoff_gate(handoff, parent, sol, status) == "waiting"
    sol["state"] = "Z"
    assert orchestration.handoff_gate(handoff, parent, sol, status) == "ready"
    for invalid in [dict(sol, start_ticks=1000), dict(sol, wait_status=124 * 256), dict(sol, ppid=1)]:
        with pytest.raises(RuntimeError):
            orchestration.handoff_gate(handoff, parent, invalid, status)
    with pytest.raises(RuntimeError):
        orchestration.handoff_gate(handoff, dict(parent, state="S"), sol, status)
    with pytest.raises(RuntimeError):
        orchestration.handoff_gate(handoff, parent, sol, {"state": "failed"})


def test_handoff_signals_only_finished_workers_frozen_parent_and_transfers_queue(tmp_path, monkeypatch):
    import signal

    legacy = tmp_path / "old/git_ignore_folder/reproduction/ablations/serial"
    legacy.mkdir(parents=True)
    jobs = []
    for seed in (0, 1, 2):
        for model in orchestration.DEFAULT_MODELS:
            jobs.append(
                {
                    "kind": "research",
                    "model": model,
                    "seed": seed,
                    "run_id": f"serial-{model}-s{seed}",
                    "state": "finished" if seed == 0 else "queued",
                    "started_at": "2026-10-18T00:00:00+00:00",
                    "finished_at": "2026-10-18T04:00:00+00:00",
                    "exit_code": 0,
                    "elapsed_seconds": 14400,
                }
            )
    sol_job = jobs[2]
    sol_job["state"] = "running"
    sol_run = legacy.parent.parent / "runs" / sol_job["run_id"]
    sol_run.mkdir(parents=True)
    write_json(
        sol_run / "status.json",
        {"state": "finished", "successful_backtests": 3, "finished_at": "2026-10-18T04:00:01+00:00"},
    )
    old = {
        "state": "running",
        "jobs": jobs,
        "models": orchestration.DEFAULT_MODELS,
        "hours_per_run": 4,
        "seeds": [0, 1, 2],
        "reasoning_effort": "medium",
        "baseline_dir": str(tmp_path / "baseline"),
    }
    write_json(legacy / "status.json", old)
    parent, sol = process(100, "T"), process(101, "Z", 100)
    handoff = {"serial_supervisor": parent.copy(), "sol_timeout": sol.copy(), "sol_run_id": sol_job["run_id"]}
    state = {
        "serial_runs": [j.copy() for j in jobs if j["seed"] == 0],
        "jobs": orchestration.remaining_jobs("new", tmp_path / "new-runs"),
        "output": str(tmp_path / "new-study"),
    }
    signals = []

    def fake_kill(pid, signum):
        signals.append((pid, signum))
        parent["state"] = "Z"

    monkeypatch.setattr(orchestration, "identity", lambda pid: parent if pid == 100 else sol)
    monkeypatch.setattr(orchestration, "group_members", lambda pgid: [])
    monkeypatch.setattr(orchestration.os, "kill", fake_kill)
    orchestration.finalize_handoff(legacy, handoff, state)
    assert signals == [(100, signal.SIGKILL)]
    transferred = read_json(legacy / "status.json")
    assert transferred["state"] == "handed_off"
    assert all(j["state"] == "transferred_to_parallel" for j in transferred["jobs"] if j["seed"] > 0)
    assert all(j["state"] == "finished" for j in state["serial_runs"])
    assert read_json(legacy / "status-before-parallel-handoff.json")["state"] == "running"


def test_isolation_removes_inherited_memory_and_mlflow_targets(tmp_path):
    inherited = {
        "PATH": os.environ["PATH"],
        "FACTOR_COSTEER_KNOWLEDGE_BASE_PATH": "/other/kb.pkl",
        "MODEL_COSTEER_NEW_KNOWLEDGE_BASE_PATH": "/other/new.pkl",
        "CoSTEER_knowledge_base_path": "/old",
        "LITELLM_PROMPT_CACHE_PATH": "/shared/cache.db",
        "WORKSPACE_PATH": "/shared/workspace",
        "MLFLOW_TRACKING_URI": "file:///shared/mlruns",
        "ARTIFACT_SIGNING_KEY": "old-key",
    }
    first = isolated_env(tmp_path, tmp_path / "first", "0", inherited)
    second = isolated_env(tmp_path, tmp_path / "second", "0", inherited)
    for name in inherited:
        if name != "PATH":
            assert name not in first
    for name in (
        "TMPDIR",
        "RDAGENT_LOCAL_CACHE_ROOT",
        "XDG_CACHE_HOME",
        "CUDA_CACHE_PATH",
        "ARTIFACT_SIGNING_KEY_PATH",
    ):
        assert first[name] != second[name]
    assert first["CUDA_VISIBLE_DEVICES"] == "0"
    assert first["PYTHONPATH"] == str(tmp_path)
    with pytest.raises(FileExistsError):
        isolated_env(tmp_path, tmp_path / "first", "0", inherited)


def make_serial_inputs(tmp_path):
    runs = []
    for model in orchestration.DEFAULT_MODELS:
        run = tmp_path / model
        workspace = run / "workspaces" / "candidate"
        workspace.mkdir(parents=True)
        (workspace / "model.py").write_text("model_cls = object\n")
        (workspace / "conf.yaml").write_text(
            "qlib_init: {provider_uri: '~/.qlib/qlib_data/cn_data'}\n"
            "task:\n  model:\n    class: GeneralPTNN\n    kwargs: {n_epochs: '{{ n_epochs }}', early_stop: 10}\n"
        )
        (workspace / "read_exp_res.py").write_text("# extraction\n")
        pd.DataFrame({"return": [0.001] * 120}, index=pd.date_range("2017-01-01", periods=120)).to_parquet(
            workspace / "ret.parquet"
        )
        (workspace / "model_weights.pt").write_text("must never be copied")
        (workspace / "mlruns").mkdir()
        hashes = {n: {"sha256": benchmark.sha256(workspace / n)} for n in ("model.py", "conf.yaml")}
        evaluation = {
            "state": "finished",
            "config": "conf.yaml",
            "run_env": {"n_epochs": "100"},
            "inputs_before": hashes,
            "inputs_after": hashes,
            "recorder_id": model + "-old",
        }
        write_json(
            run / "results.json",
            [
                {
                    "iteration": 0,
                    "accepted": True,
                    "metrics": {"IC": 0.1},
                    "workspace": str(workspace),
                    "evaluation": evaluation,
                }
            ],
        )
        runs.append({"model": model, "run_id": model, "run_dir": str(run), "seed": 0, "state": "finished"})
    return runs


def test_fixtures_copy_only_verified_inputs_and_never_results(tmp_path):
    import yaml

    runs = make_serial_inputs(tmp_path)
    fixtures = benchmark.freeze_fixtures(runs, tmp_path / "fixtures")
    assert len(fixtures) == 6
    for fixture in fixtures:
        directory = Path(fixture["directory"])
        assert not (directory / "mlruns").exists()
        assert not (directory / "model_weights.pt").exists()
        kwargs = yaml.safe_load((directory / "benchmark.yaml").read_text())["task"]["model"]["kwargs"]
        assert kwargs["n_epochs"] == 5
        assert kwargs["early_stop"] == 5
        assert kwargs["seed"] in (202600, 202601)
        destination = tmp_path / ("fresh-" + fixture["fixture_id"])
        assert benchmark.prepare_workspace(fixture, destination) == fixture["input_hashes"]
        assert not (destination / "mlruns").exists()
    (Path(fixtures[0]["directory"]) / "model.py").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="changed"):
        benchmark.prepare_workspace(fixtures[0], tmp_path / "tampered")


def make_rounds():
    fixtures = [{"fixture_id": str(i), "input_hashes": {"input": str(i)}} for i in range(6)]
    rounds = []
    for repeat in range(2):
        for concurrency, elapsed in [(1, 120), (3, 50), (6, 40)]:
            jobs = []
            for f in fixtures:
                jobs.append(
                    {
                        "fixture_id": f["fixture_id"],
                        "state": "finished",
                        "exit_code": 0,
                        "result": {
                            "state": "finished",
                            "recorder_id": f"r{repeat}-c{concurrency}-{f['fixture_id']}",
                            "inputs_before": f["input_hashes"],
                            "inputs_after": f["input_hashes"],
                        },
                    }
                )
            rounds.append({"concurrency": concurrency, "state": "finished", "elapsed_seconds": elapsed, "jobs": jobs})
    return fixtures, rounds


def test_selection_excludes_fast_failures_stale_results_and_incomplete_rounds():
    fixtures, rounds = make_rounds()
    assert benchmark.choose_concurrency(rounds, fixtures) == 6
    rounds[2]["jobs"][0]["state"] = "failed"
    assert benchmark.choose_concurrency(rounds, fixtures) == 3
    rounds[1]["jobs"][0]["result"]["recorder_id"] = rounds[4]["jobs"][0]["result"]["recorder_id"]
    assert benchmark.choose_concurrency(rounds, fixtures) == 1
    rounds[0]["jobs"][0]["result"]["inputs_after"] = {"input": "changed"}
    with pytest.raises(RuntimeError, match="No concurrency"):
        benchmark.choose_concurrency(rounds, fixtures)
    fixtures, rounds = make_rounds()
    with pytest.raises(RuntimeError):
        benchmark.choose_concurrency(rounds[:3], fixtures)


def test_evaluation_checks_fresh_recorder_and_unchanged_inputs(tmp_path, monkeypatch):
    runs = make_serial_inputs(tmp_path)
    fixture = benchmark.freeze_fixtures(runs, tmp_path / "fixtures")[0]
    workspace = tmp_path / "worker"

    def fake_run(command, cwd, **kwargs):
        cwd = Path(cwd)
        recorder = cwd / "mlruns/1/fresh/meta.yaml"
        recorder.parent.mkdir(parents=True, exist_ok=True)
        recorder.write_text("status: 3\n")
        write_json(cwd / "recorder.json", {"recorder_id": "fresh"})
        pd.Series({key: 0.1 for key in METRICS.values()}).to_csv(cwd / "qlib_res.csv")
        returns = pd.DataFrame(
            {"return": [0.001] * 120, "cost": [0.0001] * 120, "bench": [0.0005] * 120},
            index=pd.date_range("2017-01-01", periods=120),
        )
        if cwd.name == "incomplete":
            returns = returns.iloc[:-1]
        returns.to_parquet(cwd / "ret.parquet")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    assert benchmark.evaluate(fixture, workspace)["state"] == "finished"
    fixture["source_recorder_id"] = "fresh"
    assert benchmark.evaluate(fixture, tmp_path / "stale")["state"] == "failed"
    fixture["source_recorder_id"] = "old"
    assert benchmark.evaluate(fixture, tmp_path / "incomplete")["state"] == "failed"


def test_pool_caps_live_workers_and_isolates_their_environments(tmp_path):
    jobs = [{"state": "queued", "index": i} for i in range(7)]
    seen = []

    def launch(job):
        env = isolated_env(tmp_path, tmp_path / f"runtime-{job['index']}", "0")
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.08)"], env=env, start_new_session=True)

    def tick(force=False):
        seen.append(sum(j["state"] == "running" for j in jobs))

    run_pool(jobs, 3, launch, lambda j: j["exit_code"] == 0, tick, poll_seconds=0.01)
    assert max(seen) == 3
    assert all(j["state"] == "finished" for j in jobs)
    assert len({j["process"]["pid"] for j in jobs}) == 7


def test_pool_cleans_owned_children_after_guard_failure(tmp_path):
    jobs = [{"state": "queued"}]
    children = []

    def launch(job):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
        children.append(child)
        return child

    def tick(force=False):
        if children:
            raise RuntimeError("guard failed")

    with pytest.raises(RuntimeError, match="guard failed"):
        run_pool(jobs, 1, launch, lambda j: True, tick, poll_seconds=0.01)
    assert children[0].poll() is not None
    assert jobs[0]["state"] == "interrupted"


def test_reports_do_not_pool_serial_and_parallel_results(tmp_path):
    serial = make_serial_inputs(tmp_path)
    jobs = orchestration.remaining_jobs("new", tmp_path / "new-runs")
    for job in jobs:
        directory = Path(job["run_dir"])
        directory.mkdir(parents=True)
        write_json(
            directory / "results.json",
            [
                {"iteration": 0, "accepted": True, "metrics": {"IC": 0.02}},
                {"iteration": 1, "accepted": False, "metrics": {"IC": 0.99}},
            ],
        )
        job["state"] = "finished" if job["seed"] == 1 else "failed"
    state = {
        "state": "finished_with_failures",
        "serial_runs": serial,
        "jobs": jobs,
        "hours_per_run": 4,
        "selected_concurrency": 3,
        "baseline_dir": str(tmp_path),
        "serial_source_commit": "old",
        "git_commit": "new",
    }
    report = write_protocol_reports(tmp_path, state)
    for cohort, value, planned in [("serial_reference", 0.1, 1), ("parallel_study", 0.02, 2)]:
        for row in report[cohort]["aggregate"]:
            if row["metric"] == "IC":
                assert row["mean"] == value
                assert row["planned_runs"] == planned
                assert row["completed_runs"] == 1
    assert all(row["selection"] == "last accepted iteration 0" for row in report["parallel_study"]["runs"])


def test_sharpe_uses_selected_net_returns_and_matches_reported_ir(tmp_path):
    from examples.qtrade_fin_quant.parallel_report import sharpe_from_selected

    returns = pd.DataFrame(
        {"return": [0.01, -0.005, 0.003, 0.007], "cost": [0.001] * 4, "bench": [0.002, -0.003, 0.004, 0.001]}
    )
    returns.to_parquet(tmp_path / "ret.parquet")
    write_json(tmp_path / "recorder.json", {"recorder_id": "selected"})
    net = returns["return"] - returns["cost"]
    excess = net - returns["bench"]
    trial = {
        "accepted": True,
        "workspace": str(tmp_path),
        "evaluation": {"state": "finished", "recorder_id": "selected"},
        "metrics": {
            METRICS["excess_return_information_ratio_after_costs"]: excess.mean() / excess.std() * math.sqrt(238)
        },
    }
    result = sharpe_from_selected([trial])
    assert result["portfolio_sharpe_after_costs_rf0"] == pytest.approx(net.mean() / net.std() * math.sqrt(238))
    trial["evaluation"]["recorder_id"] = "different"
    assert sharpe_from_selected([trial])["portfolio_sharpe_after_costs_rf0"] is None


def test_controller_completes_all_benchmarks_then_six_fresh_runs(tmp_path, monkeypatch):
    serial = make_serial_inputs(tmp_path / "serial")
    output = tmp_path / "study"
    output.mkdir()
    state = {
        "state": "waiting_for_serial_sol",
        "serial_runs": serial,
        "jobs": orchestration.remaining_jobs("new", tmp_path / "runs"),
        "hours_per_run": 4,
        "gpu": "0",
        "baseline_dir": str(tmp_path),
        "serial_source_commit": "old",
        "git_commit": "new",
    }
    controller = orchestration.Controller(tmp_path, output, state, 0)
    monkeypatch.setattr(orchestration, "source_commit", lambda root: "new")
    monkeypatch.setattr(controller, "sample_resources", lambda: None)
    original_pool = orchestration.run_pool
    monkeypatch.setattr(orchestration, "run_pool", lambda *a, **kw: original_pool(*a, **kw, poll_seconds=0.01))
    launch_order = []

    def fake_popen(command, runtime_id, log_path):
        env = isolated_env(tmp_path, output / "runtime" / runtime_id, "0")
        launch_order.append("benchmark" if "--fixture" in command else "research")
        if "--fixture" in command:
            fixture = read_json(Path(command[command.index("--fixture") + 1]))
            workspace = Path(command[command.index("--workspace") + 1])
            benchmark.prepare_workspace(fixture, workspace)
            result = {
                "state": "finished",
                "recorder_id": runtime_id,
                "inputs_before": fixture["input_hashes"],
                "inputs_after": fixture["input_hashes"],
            }
            target = workspace / "evaluation.json"
        else:
            run_id = command[command.index("--run-id") + 1]
            job = next(j for j in state["jobs"] if j["run_id"] == run_id)
            directory = Path(job["run_dir"])
            directory.mkdir(parents=True, exist_ok=False)
            assert command[command.index("--duration") + 1] == "4h"
            target = directory / "status.json"
            result = {"state": "finished"}
        code = "import json,sys,time;from pathlib import Path;time.sleep(.06);Path(sys.argv[1]).write_text(sys.argv[2])"
        return subprocess.Popen(
            [sys.executable, "-c", code, str(target), json.dumps(result)], env=env, start_new_session=True
        )

    monkeypatch.setattr(controller, "popen", fake_popen)
    concurrency = controller.run_benchmark()
    controller.run_research(concurrency)
    assert state["state"] == "finished"
    assert launch_order == ["benchmark"] * 36 + ["research"] * 6
    assert len(list((output / "runtime").iterdir())) == 42
    assert all(j["state"] == "finished" for j in state["jobs"])
    assert len({j["run_dir"] for j in state["jobs"]}) == 6
    assert {r["concurrency"] for r in controller.benchmark["rounds"]} == {1, 3, 6}
    assert controller.benchmark["selected_concurrency"] == concurrency


def test_local_env_honors_private_temporary_cache(tmp_path, monkeypatch):
    from rdagent.utils.env import LocalConf, LocalEnv

    private = tmp_path / "private"
    monkeypatch.setenv("RDAGENT_LOCAL_CACHE_ROOT", str(private))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env = LocalEnv(conf=LocalConf(enable_cache=False, default_entry="true"))
    (workspace / "inspect_cache.py").write_text(
        f"from pathlib import Path\np=Path('workspace_cache')\nassert p.resolve()==Path('{private}/full')\np.joinpath('sentinel').write_text('private')\n"
    )
    command = f"{sys.executable} inspect_cache.py"
    result = env.run(entry=command, local_path=str(workspace))
    assert result.exit_code == 0
    assert (private / "full/sentinel").read_text() == "private"
    assert not (workspace / "workspace_cache").exists()
