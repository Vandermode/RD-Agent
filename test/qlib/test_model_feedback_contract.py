import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from rdagent.scenarios.qlib.developer import feedback as feedback_module
from rdagent.scenarios.qlib.experiment.workspace import _extract_training_log


def test_model_only_feedback_uses_the_decision_contract(monkeypatch):
    monkeypatch.setattr("rdagent.utils.agent.tpl.logger.log_object", lambda *args, **kwargs: None)
    scenario = Mock()
    scenario.get_scenario_all_desc.return_value = "Predict returns with a neural model."
    experiment = SimpleNamespace(
        hypothesis=SimpleNamespace(hypothesis="Attention improves the GRU", reason="Use the full sequence"),
        sub_tasks=[SimpleNamespace(get_task_information=lambda: "Attention GRU")],
        sub_workspace_list=[SimpleNamespace(file_dict={"model.py": "model_cls = Net"})],
        stdout="Epoch0: train 0.99, valid 0.99",
        result=pd.Series({"IC": 0.036, "1day.excess_return_with_cost.annualized_return": 0.071,
                          "1day.excess_return_with_cost.max_drawdown": -0.104}),
    )
    trace = Mock()
    trace.get_sota_hypothesis_and_experiment.return_value = (None, None)
    backend = Mock()

    # A conforming model returns the field requested by the system prompt.
    def answer(**kwargs):
        if '"Decision"' in kwargs["system_prompt"]:
            return json.dumps({"Decision": True})
        return json.dumps({"Replace Best Result": "yes"})

    backend.build_messages_and_create_chat_completion.side_effect = answer
    monkeypatch.setattr(feedback_module, "APIBackend", lambda: backend)
    evaluator = feedback_module.QlibModelExperiment2Feedback(scenario)
    result = evaluator.generate_feedback(experiment, trace)
    assert result.decision is True


def test_feedback_preserves_positive_qlib_training_losses():
    output = """[qlib] Epoch17: train 0.992301, valid 0.994756
[qlib] unrelated progress line
[qlib] best score: 0.994756 @ 17 epoch
"""
    assert _extract_training_log(output).splitlines() == [
        "Epoch17: train 0.992301, valid 0.994756",
        "best score: 0.994756 @ 17 epoch",
    ]


def test_feedback_preserves_legacy_negative_training_scores():
    output = "Epoch1: train -0.92, valid -0.94\nbest score: -0.94 @ 1 epoch"
    assert _extract_training_log(output) == output
