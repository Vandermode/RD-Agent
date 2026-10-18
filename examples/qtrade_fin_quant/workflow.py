"""Audited joint loop and disposable preflight scheduling."""

import json
from datetime import datetime, timezone
from pathlib import Path

from rdagent.app.qlib_rd_loop.quant import QuantRDLoop
from rdagent.core.exception import CoderError
from rdagent.core.serialization import load as load_artifact
from rdagent.log import rdagent_logger as logger
from rdagent.log.timer import RDAgentTimer
from rdagent.scenarios.qlib.proposal.bandit import EnvController
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesis
from rdagent.oai.structured_response import ResponseSchemaError


PREFLIGHT_ACTIONS = ("factor", "factor", "model", "factor", "model")


class FixedBudgetTimer(RDAgentTimer):
    def add_duration(self, duration):
        # Every model gets the same wall-clock budget, including failed requests.
        logger.info(f"API retry time {duration} remains charged to the research budget")


class PreflightController(EnvController):
    """Exercise both transitions without changing the measured research policy."""

    def __init__(self):
        super().__init__()
        self._next_action = 1  # Upstream always starts with a factor experiment.

    def decide(self, metrics):
        action = PREFLIGHT_ACTIONS[self._next_action]
        self._next_action += 1
        return action


def verify_evaluation(experiment, seen_recorders: set[str]) -> dict:
    path = experiment.experiment_workspace.workspace_path / "evaluation.json"
    if not path.is_file():
        raise RuntimeError("Backtest has metrics but no execution receipt")
    evaluation = json.loads(path.read_text())
    if evaluation.get("state") != "finished" or evaluation.get("command_cache_enabled") is not False:
        raise RuntimeError("Backtest did not finish with command caching disabled")
    if evaluation.get("inputs_before") != evaluation.get("inputs_after"):
        raise RuntimeError("Backtest input hashes changed during evaluation")
    recorder = evaluation.get("recorder_id")
    if not recorder or recorder in seen_recorders:
        raise RuntimeError("Backtest reused a recorder from another experiment")
    seen_recorders.add(recorder)
    return evaluation


class AuditedQuantRDLoop(QuantRDLoop):
    def __init__(self, settings):
        super().__init__(settings)
        self._seen_recorders = set()
        self._recorded_at = []

    def _propose(self):
        hypothesis = super()._propose()
        self._latest_hypothesis = hypothesis
        return hypothesis

    async def direct_exp_gen(self, prev_out):
        self._latest_hypothesis = None
        try:
            return await super().direct_exp_gen(prev_out)
        except ResponseSchemaError as error:
            # An invalid proposal consumes budget and is recorded as a failed
            # iteration. It must not crash the whole research trajectory.
            action = self.hypothesis_gen.targets
            if action not in ("factor", "model"):
                raise
            hypothesis = self._latest_hypothesis or QlibQuantHypothesis(
                hypothesis="Proposal rejected because its response schema was invalid.",
                reason=str(error), concise_reason="Schema validation failed",
                concise_observation="No valid proposal or backtest was produced.",
                concise_justification="A malformed response cannot establish a research result.",
                concise_knowledge="Follow the required response schema.", action=action,
            )
            experiment_type = QlibFactorExperiment if action == "factor" else QlibModelExperiment
            experiment = experiment_type(sub_tasks=[], hypothesis=hypothesis)
            experiment.base_features = dict(self.plan["features"])
            experiment.proposal_schema_error = str(error)
            logger.warning(f"Recording rejected {action} proposal after bounded schema repairs: {error}")
            return {"propose": hypothesis, "exp_gen": experiment}

    def coding(self, prev_out):
        error = getattr(prev_out["direct_exp_gen"]["exp_gen"], "proposal_schema_error", None)
        if error:
            raise CoderError(f"Proposal rejected before coding: {error}")
        return super().coding(prev_out)

    def record(self, prev_out):
        experiment = prev_out.get("running") or prev_out.get("coding") or prev_out["direct_exp_gen"]["exp_gen"]
        if experiment.result is not None:
            verify_evaluation(experiment, self._seen_recorders)
        super().record(prev_out)
        self._recorded_at.append(datetime.now(timezone.utc).isoformat())
        # Persist usable progress after every iteration, including on later faults.
        from examples.qtrade_fin_quant.run import summarize, write_json

        output = Path(self.session_folder).parent.parent
        summaries = summarize(self, "quant")
        write_json(output / "results.json", summaries)
        write_json(output / "status.json", {
            "state": "running", "updated_at": self._recorded_at[-1],
            "recorded_iterations": len(summaries),
            "successful_backtests": sum(bool(item["metrics"]) for item in summaries),
            "accepted_iterations": sum(item["accepted"] for item in summaries),
            "actions": [item["action"] for item in summaries],
        })


def verify_preflight(output: Path, summaries: list[dict]) -> dict:
    successful = [item for item in summaries if item["metrics"]]
    counts = {action: sum(item["action"] == action for item in successful) for action in ("factor", "model")}
    if [item["action"] for item in summaries] != list(PREFLIGHT_ACTIONS):
        raise RuntimeError("Preflight did not complete the required factor/model action sequence")
    if any(count < 2 for count in counts.values()):
        raise RuntimeError("Preflight requires at least two successful backtests for each action")
    if any(item.get("feedback_schema_error") or item.get("proposal_schema_error") for item in summaries):
        raise RuntimeError("Preflight exhausted response schema repairs")
    evaluations = [item["evaluation"] for item in successful]
    recorders = [item["recorder_id"] for item in evaluations]
    if len(set(recorders)) != len(recorders):
        raise RuntimeError("Preflight reused an evaluation recorder")
    factor_hashes = {
        item["evaluation"]["inputs_before"]["combined_factors_df.parquet"]["sha256"]
        for item in successful if item["action"] == "factor"
    }
    if len(factor_hashes) < 2:
        raise RuntimeError("Preflight did not evaluate distinct factor data")
    checkpoint = output / "trace/__session__" / str(len(summaries) - 1) / "4_record"
    with checkpoint.open("rb") as file:
        restored = load_artifact(file)
    if len(restored.trace.hist) != len(summaries) or any(
        restored.step_idx[index] != 5 for index in range(len(summaries))
    ):
        raise RuntimeError("Preflight checkpoint did not restore every completed iteration")
    knowledge_counts = {}
    for action in ("factor", "model"):
        with (output / f"{action}-knowledge.pkl").open("rb") as file:
            knowledge = load_artifact(file)
        knowledge_counts[action] = len(knowledge.success_task_to_knowledge_dict)
        if knowledge_counts[action] < 1:
            raise RuntimeError(f"Preflight did not restore successful {action} coding memory")
    result = {
        "passed": True, "actions": [item["action"] for item in summaries],
        "successful_backtests_by_action": counts, "distinct_factor_inputs": len(factor_hashes),
        "unique_recorder_ids": recorders, "all_input_hashes_unchanged": True,
        "command_cache_enabled": False,
        "checkpoint_reloaded": True, "knowledge_entries_reloaded": knowledge_counts,
        "model_feature_counts": [item["evaluation"]["run_env"].get("num_features", "20 (Alpha20)")
                                 for item in successful if item["action"] == "model"],
    }
    (output / "preflight-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
