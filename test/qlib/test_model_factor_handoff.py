from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from rdagent.scenarios.qlib.developer import model_runner
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


@pytest.mark.parametrize("factor_count", [1, 2])
def test_model_training_includes_accepted_factors(monkeypatch, tmp_path, factor_count):
    accepted = []
    for _ in range(factor_count):
        factor = object.__new__(QlibFactorExperiment)
        factor.running_info = SimpleNamespace(result=pd.Series({"IC": 0.03}))
        accepted.append(factor)
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2017-01-03"), "SH600000")], names=["datetime", "instrument"]
    )
    factors = pd.DataFrame(
        {f"accepted_factor_{i}": [0.1 + i] for i in range(factor_count)}, index=index
    )
    process = Mock(return_value=factors)
    monkeypatch.setattr(model_runner, "process_factor_data", process)
    workspace = Mock(workspace_path=tmp_path)
    workspace.execute.return_value = (pd.Series({"IC": 0.04}), "training log")
    experiment = SimpleNamespace(
        based_experiments=accepted,
        base_features={f"base_{i}": "$close" for i in range(20)},
        sub_tasks=[SimpleNamespace(name="GRU", training_hyperparameters={}, model_type="TimeSeries")],
        sub_workspace_list=[SimpleNamespace(file_dict={"model.py": "model_cls = Net"})],
        experiment_workspace=workspace,
    )

    runner = object.__new__(model_runner.QlibModelRunner)
    runner.develop(experiment)

    process.assert_called_once_with(accepted)
    execution = workspace.execute.call_args.kwargs
    assert execution["qlib_config_name"] == "conf_sota_factors_model.yaml"
    assert int(execution["run_env"]["num_features"]) == 20 + factor_count
    saved = pd.read_parquet(tmp_path / "combined_factors_df.parquet")
    assert list(saved.columns) == [("feature", name) for name in factors.columns]


def test_model_training_without_accepted_factors_uses_baseline(monkeypatch, tmp_path):
    process = Mock()
    monkeypatch.setattr(model_runner, "process_factor_data", process)
    workspace = Mock(workspace_path=tmp_path)
    workspace.execute.return_value = (pd.Series({"IC": 0.04}), "training log")
    experiment = SimpleNamespace(
        based_experiments=[],
        base_features={f"base_{i}": "$close" for i in range(20)},
        sub_tasks=[SimpleNamespace(name="GRU", training_hyperparameters={}, model_type="TimeSeries")],
        sub_workspace_list=[SimpleNamespace(file_dict={"model.py": "model_cls = Net"})],
        experiment_workspace=workspace,
    )

    runner = object.__new__(model_runner.QlibModelRunner)
    runner.develop(experiment)

    process.assert_not_called()
    assert workspace.execute.call_args.kwargs["qlib_config_name"] == "conf_baseline_factors_model.yaml"
    assert not (tmp_path / "combined_factors_df.parquet").exists()
