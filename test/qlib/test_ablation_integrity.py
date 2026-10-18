import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from examples.qtrade_fin_quant import workflow
from rdagent.core.exception import CoderError
from rdagent.oai.structured_response import ResponseSchemaError


def test_api_retry_time_cannot_extend_a_model_budget():
    timer = workflow.FixedBudgetTimer()
    timer.reset("4h")
    deadline = timer.target_time
    timer.add_duration(timedelta(minutes=15))
    assert timer.target_time == deadline


def test_preflight_exercises_repeated_factors_and_both_transitions():
    controller = workflow.PreflightController()
    actual = ["factor"] + [controller.decide(None) for _ in range(4)]
    assert actual == ["factor", "factor", "model", "factor", "model"]
    assert workflow.AuditedQuantRDLoop.steps == ["direct_exp_gen", "coding", "running", "feedback", "record"]


def test_duplicate_evaluation_is_rejected_before_it_can_enter_history(tmp_path):
    receipt = {"state": "finished", "command_cache_enabled": False, "recorder_id": "same-run",
               "inputs_before": {"data": "hash"}, "inputs_after": {"data": "hash"}}
    (tmp_path / "evaluation.json").write_text(json.dumps(receipt))
    experiment = SimpleNamespace(experiment_workspace=SimpleNamespace(workspace_path=tmp_path))
    seen = set()
    workflow.verify_evaluation(experiment, seen)
    with pytest.raises(RuntimeError, match="reused a recorder"):
        workflow.verify_evaluation(experiment, seen)


def test_exhausted_proposal_repairs_become_a_rejected_iteration(monkeypatch):
    async def invalid_proposal(self, prev_out):
        raise ResponseSchemaError("Required model_type still missing")

    monkeypatch.setattr(workflow.QuantRDLoop, "direct_exp_gen", invalid_proposal)
    monkeypatch.setattr(workflow, "QlibModelExperiment",
                        lambda **kwargs: SimpleNamespace(**kwargs, result=None))
    loop = object.__new__(workflow.AuditedQuantRDLoop)
    loop.hypothesis_gen = SimpleNamespace(targets="model")
    loop.plan = {"features": {"close": "$close"}}
    proposed = asyncio.run(loop.direct_exp_gen({}))
    assert proposed["exp_gen"].result is None
    assert proposed["exp_gen"].proposal_schema_error
    with pytest.raises(CoderError, match="Proposal rejected before coding"):
        loop.coding({"direct_exp_gen": proposed})


def test_preflight_rejects_one_iteration_and_repeated_factor_data(tmp_path):
    with pytest.raises(RuntimeError, match="required factor/model action sequence"):
        workflow.verify_preflight(tmp_path, [{"action": "factor", "metrics": {"IC": 0.1}}])
    repeated = [
        {"action": action, "metrics": {"IC": 0.1}, "evaluation": {
            "recorder_id": str(i), "inputs_before": {"combined_factors_df.parquet": {"sha256": "same-data"}}
        }}
        for i, action in enumerate(workflow.PREFLIGHT_ACTIONS)
    ]
    with pytest.raises(RuntimeError, match="distinct factor data"):
        workflow.verify_preflight(tmp_path, repeated)
