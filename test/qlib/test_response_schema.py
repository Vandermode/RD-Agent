import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from rdagent.oai.structured_response import ResponseSchemaError, validated_json_response
from rdagent.scenarios.qlib.developer import feedback as feedback_module
from rdagent.scenarios.qlib.proposal.model_proposal import QlibModelHypothesis2Experiment
from rdagent.scenarios.qlib.response_schema import FactorFeedbackResponse, ModelProposalResponse


def proposal():
    return {"GRU": {
        "description": "A small recurrent predictor", "formulation": "y = W h",
        "architecture": "GRU then linear", "variables": {"h": "hidden state"},
        "hyperparameters": {"hidden_size": 32},
        "training_hyperparameters": {"n_epochs": 20}, "model_type": "TimeSeries",
    }}


def feedback():
    return {"Observations": {"IC": "Improved", "Return": "Improved"},
            "Feedback for Hypothesis": {"Conclusion": "Supported by this backtest"},
            "New Hypothesis": {"Direction": "Test a shorter window"},
            "Reasoning": ["Check temporal stability", "Keep the data split fixed"],
            "Replace Best Result": "yes"}


def test_luna_nested_model_type_reaches_model_task(monkeypatch):
    payload = proposal()
    payload["GRU"]["hyperparameters"]["model_type"] = payload["GRU"].pop("model_type")
    monkeypatch.setattr(
        "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelExperiment",
        lambda tasks, hypothesis: SimpleNamespace(sub_tasks=tasks, hypothesis=hypothesis),
    )
    result = QlibModelHypothesis2Experiment().convert_response(
        json.dumps(payload), Mock(), SimpleNamespace(hist=[])
    )
    assert result.sub_tasks[0].model_type == "TimeSeries"
    assert result.sub_tasks[0].hyperparameters == {"hidden_size": 32}


def test_nested_feedback_preserves_evidence_and_explicit_decision(monkeypatch):
    backend = Mock()
    backend.build_messages_and_create_chat_completion.return_value = json.dumps(feedback())
    monkeypatch.setattr(feedback_module, "APIBackend", lambda: backend)
    result = feedback_module._validated_feedback("Evaluate the experiment", "metrics", FactorFeedbackResponse)
    assert result.decision is True
    assert json.loads(result.observations) == feedback()["Observations"]
    assert json.loads(result.reason) == feedback()["Reasoning"]
    assert backend.build_messages_and_create_chat_completion.call_count == 1


def test_missing_required_field_is_repaired_with_different_request():
    invalid = proposal()
    del invalid["GRU"]["model_type"]
    backend = Mock()
    backend.build_messages_and_create_chat_completion.side_effect = [json.dumps(invalid), json.dumps(proposal())]
    result = validated_json_response(backend, user_prompt="Propose a model", system_prompt="JSON only",
                                     response_model=ModelProposalResponse)
    assert json.loads(result)["GRU"]["model_type"] == "TimeSeries"
    first, repair = [call.kwargs for call in backend.build_messages_and_create_chat_completion.call_args_list]
    assert first["chat_cache_prefix"] != repair["chat_cache_prefix"]
    assert repair["former_messages"][1]["content"] == json.dumps(invalid)
    assert "model_type" in repair["user_prompt"]


def test_repair_exhaustion_is_bounded_and_feedback_cannot_promote(monkeypatch):
    invalid = feedback()
    del invalid["Replace Best Result"]
    backend = Mock()
    backend.build_messages_and_create_chat_completion.return_value = json.dumps(invalid)
    monkeypatch.setattr(feedback_module, "APIBackend", lambda: backend)
    result = feedback_module._validated_feedback("Evaluate", "metrics", FactorFeedbackResponse)
    assert result.decision is False
    assert result.acceptable is False
    assert isinstance(result.exception, ResponseSchemaError)
    assert backend.build_messages_and_create_chat_completion.call_count == 3


@pytest.mark.parametrize("decision", [None, {}, "probably", 2])
def test_ambiguous_feedback_decisions_are_rejected(decision):
    invalid = feedback()
    invalid["Replace Best Result"] = decision
    with pytest.raises(ValidationError):
        FactorFeedbackResponse.model_validate(invalid)


def test_conflicting_model_types_are_not_silently_reinterpreted():
    invalid = proposal()
    invalid["GRU"]["hyperparameters"]["model_type"] = "Tabular"
    with pytest.raises(ValidationError, match="Conflicting"):
        ModelProposalResponse.model_validate(invalid)


def test_two_proposed_models_cannot_silently_drop_the_second():
    invalid = proposal()
    invalid["Other"] = invalid["GRU"].copy()
    with pytest.raises(ValidationError):
        ModelProposalResponse.model_validate(invalid)
