from abc import abstractmethod
from typing import Tuple

from pydantic import BaseModel

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import (
    ExperimentPlan,
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisGen,
    Scenario,
    Trace,
)
from rdagent.oai.llm_utils import APIBackend
from rdagent.oai.structured_response import validated_json_response
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry


class LLMHypothesisGen(HypothesisGen):
    response_model: type[BaseModel] | None = None
    def __init__(self, scen: Scenario):
        super().__init__(scen)

    # The following methods are scenario related so they should be implemented in the subclass
    @abstractmethod
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str) -> Hypothesis: ...

    def gen(
        self,
        trace: Trace,
        plan: ExperimentPlan | None = None,
    ) -> Hypothesis:
        context_dict, json_flag = self.prepare_context(trace)

        system_prompt = T(".prompts:hypothesis_gen.system_prompt").r(
            targets=self.targets,
            scenario=(
                self.scen.get_scenario_all_desc(filtered_tag=self.targets)
                if self.targets in ["factor", "model"]
                else self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
            ),
            hypothesis_output_format=context_dict["hypothesis_output_format"],
            hypothesis_specification=context_dict["hypothesis_specification"],
            user_instruction=plan.get("user_instruction", None) if plan is not None else None,
        )
        user_prompt = T(".prompts:hypothesis_gen.user_prompt").r(
            targets=self.targets,
            hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
            last_hypothesis_and_feedback=(
                context_dict["last_hypothesis_and_feedback"] if "last_hypothesis_and_feedback" in context_dict else ""
            ),
            sota_hypothesis_and_feedback=(
                context_dict["sota_hypothesis_and_feedback"] if "sota_hypothesis_and_feedback" in context_dict else ""
            ),
            RAG=context_dict["RAG"],
        )

        if self.response_model is not None:
            resp = validated_json_response(
                APIBackend(), user_prompt=user_prompt, system_prompt=system_prompt,
                response_model=self.response_model,
                validation_context={"expected_action": self.targets},
            )
        else:
            resp = APIBackend().build_messages_and_create_chat_completion(
                user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, str]
            )

        hypothesis = self.convert_response(resp)

        return hypothesis


class FactorHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "factors"


class ModelHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "model tuning"


class FactorAndModelHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "feature engineering and model building"


class LLMHypothesis2Experiment(Hypothesis2Experiment[Experiment]):
    response_model: type[BaseModel] | None = None
    @abstractmethod
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> Experiment: ...

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        if self.response_model is not None:
            # The structured path owns its bounded repairs; do not multiply them
            # by the legacy retry decorator or retry an unchanged cached payload.
            return self._convert(hypothesis, trace)
        return self._convert_with_retry(hypothesis, trace)

    @wait_retry(retry_n=5)
    def _convert_with_retry(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        return self._convert(hypothesis, trace)

    def _convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        context, json_flag = self.prepare_context(hypothesis, trace)
        system_prompt = T(".prompts:hypothesis2experiment.system_prompt").r(
            targets=self.targets,
            scenario=trace.scen.get_scenario_all_desc(filtered_tag=self.targets),
            experiment_output_format=context["experiment_output_format"],
        )
        user_prompt = T(".prompts:hypothesis2experiment.user_prompt").r(
            targets=self.targets,
            target_hypothesis=context["target_hypothesis"],
            hypothesis_and_feedback=(
                context["hypothesis_and_feedback"] if "hypothesis_and_feedback" in context else ""
            ),
            last_hypothesis_and_feedback=(
                context["last_hypothesis_and_feedback"] if "last_hypothesis_and_feedback" in context else ""
            ),
            sota_hypothesis_and_feedback=(
                context["sota_hypothesis_and_feedback"] if "sota_hypothesis_and_feedback" in context else ""
            ),
            target_list=context["target_list"],
            RAG=context["RAG"],
        )

        if self.response_model is not None:
            resp = validated_json_response(
                APIBackend(), user_prompt=user_prompt, system_prompt=system_prompt,
                response_model=self.response_model,
            )
        else:
            resp = APIBackend().build_messages_and_create_chat_completion(
                user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, dict[str, str | dict]]
            )

        return self.convert_response(resp, hypothesis, trace)


class FactorHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "factors"


class ModelHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "model tuning"


class FactorAndModelHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "feature engineering and model building"
