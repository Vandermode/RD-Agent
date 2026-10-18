import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.scenarios.qlib.developer import model_runner
from rdagent.scenarios.qlib.experiment import workspace as workspace_module


@pytest.fixture
def environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_CACHE", "True")  # The old launcher's dangerous setting.
    monkeypatch.setattr(workspace_module.MODEL_COSTEER_SETTINGS, "env_type", "conda")
    monkeypatch.setattr(workspace_module.QlibCondaConf, "_update_bin_path", lambda self: None)
    monkeypatch.setattr(workspace_module.logger, "log_object", lambda *args, **kwargs: None)
    calls = []
    options = {"fail": False, "overwrite_inputs": False, "reuse_recorder": False}

    class FakeQlibEnvironment:
        def __init__(self, conf):
            self.conf = conf

        def prepare(self):
            pass

        def run(self, *, local_path, entry, env):
            path = Path(local_path)
            assert self.conf.enable_cache is False
            calls.append((entry, dict(env)))
            if entry.startswith("qrun"):
                if options["fail"]:
                    return SimpleNamespace(stdout="training failed", exit_code=1)
                identity = "old" if options["reuse_recorder"] else uuid.uuid4().hex
                recorder = path / "mlruns/1" / identity / "meta.yaml"
                recorder.parent.mkdir(parents=True, exist_ok=True)
                recorder.write_text("status: 3\n")
                (path / "recorder.json").write_text(json.dumps({"recorder_id": identity}))
                factors = pd.read_parquet(path / "combined_factors_df.parquet")
                value = float(factors.iloc[0, 0]) * float(env["lr"])
                pd.Series({"IC": value}).to_csv(path / "qlib_res.csv")
                pd.DataFrame({"return": [value]}).to_parquet(path / "ret.parquet")
                if options["overwrite_inputs"]:
                    pd.DataFrame({"old_factor": [99.0]}).to_parquet(path / "combined_factors_df.parquet")
                return SimpleNamespace(stdout="Epoch0: train 1.0, valid 1.0", exit_code=0)
            return SimpleNamespace(stdout="extracted", exit_code=0)

    monkeypatch.setattr(workspace_module, "QlibCondaEnv", FakeQlibEnvironment)
    workspace = object.__new__(workspace_module.QlibFBWorkspace)
    workspace.workspace_path = tmp_path
    (tmp_path / "conf.yaml").write_text("task: model\n")
    pd.DataFrame({"factor_v1": [1.0]}).to_parquet(tmp_path / "combined_factors_df.parquet")
    return workspace, calls, options


def test_changed_factor_data_and_training_settings_execute_fresh(environment):
    workspace, calls, _ = environment
    first, _ = workspace.execute(run_env={"lr": "0.1"})
    receipt1 = json.loads((workspace.workspace_path / "evaluation.json").read_text())
    pd.DataFrame({"factor_v2": [3.0]}).to_parquet(workspace.workspace_path / "combined_factors_df.parquet")
    second, _ = workspace.execute(run_env={"lr": "0.2"})
    receipt2 = json.loads((workspace.workspace_path / "evaluation.json").read_text())
    assert first["IC"] == pytest.approx(0.1)
    assert second["IC"] == pytest.approx(0.6)
    assert len(calls) == 4
    assert receipt1["recorder_id"] != receipt2["recorder_id"]
    assert receipt1["inputs_before"] != receipt2["inputs_before"]
    assert receipt2["inputs_before"] == receipt2["inputs_after"]
    assert receipt2["state"] == "finished"
    assert receipt2["command_cache_enabled"] is False


def test_failed_training_cannot_return_previous_success(environment):
    workspace, calls, options = environment
    workspace.execute(run_env={"lr": "0.1"})
    options["fail"] = True
    result, _ = workspace.execute(run_env={"lr": "0.2"})
    assert result is None
    assert len(calls) == 3  # Do not extract a stale previous recorder after failure.
    assert json.loads((workspace.workspace_path / "evaluation.json").read_text())["state"] == "failed"


def test_changed_inputs_during_execution_are_rejected(environment):
    workspace, _, options = environment
    options["overwrite_inputs"] = True
    result, message = workspace.execute(run_env={"lr": "0.1"})
    assert result is None
    assert "inputs changed" in message


def test_old_recorder_is_not_a_successful_new_evaluation(environment):
    workspace, _, options = environment
    old = workspace.workspace_path / "mlruns/1/old/meta.yaml"
    old.parent.mkdir(parents=True)
    old.write_text("status: 3\n")
    options["reuse_recorder"] = True
    result, message = workspace.execute(run_env={"lr": "0.1"})
    assert result is None
    assert "fresh MLflow recorder" in message


def test_model_runner_does_not_reuse_task_text_when_training_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(RD_AGENT_SETTINGS, "cache_with_pickle", True)
    monkeypatch.setattr(RD_AGENT_SETTINGS, "pickle_cache_folder_path_str", str(tmp_path / "cache"))
    workspace = Mock(workspace_path=tmp_path)
    workspace.execute.side_effect = [(pd.Series({"IC": 0.1}), "first"), (pd.Series({"IC": 0.2}), "second")]
    task = SimpleNamespace(name="Same task", training_hyperparameters={"lr": 0.001}, model_type="TimeSeries",
                           get_task_information=lambda: "The same task description")
    experiment = SimpleNamespace(
        based_experiments=[], base_features={"close": "$close"}, sub_tasks=[task],
        sub_workspace_list=[SimpleNamespace(file_dict={"model.py": "model_cls = Net"})],
        experiment_workspace=workspace,
    )
    runner = object.__new__(model_runner.QlibModelRunner)
    runner.develop(experiment)
    task.training_hyperparameters["lr"] = 0.002
    runner.develop(experiment)
    assert workspace.execute.call_count == 2
    assert experiment.result["IC"] == pytest.approx(0.2)
