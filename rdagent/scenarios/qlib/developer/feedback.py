from pathlib import Path

import pandas as pd

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.oai.structured_response import ResponseSchemaError, validated_json_response
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.scenarios.qlib.response_schema import FactorFeedbackResponse, FeedbackResponse
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent

IMPORTANT_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
]


def _validated_feedback(system_prompt: str, user_prompt: str, response_model) -> HypothesisFeedback:
    try:
        response = validated_json_response(
            APIBackend(), system_prompt=system_prompt, user_prompt=user_prompt,
            response_model=response_model,
        )
    except ResponseSchemaError as error:
        # A malformed evaluation must never promote a candidate or kill later
        # experiments. Preserve the failure in the research trace explicitly.
        logger.warning(f"Rejecting candidate because feedback could not be validated: {error}")
        return HypothesisFeedback(
            observations="Feedback response failed schema validation.",
            hypothesis_evaluation="No valid evaluation was obtained.",
            new_hypothesis="Repair the response contract before drawing conclusions about this candidate.",
            reason=str(error), decision=False, acceptable=False, exception=error,
        )
    return HypothesisFeedback(**response_model.model_validate_json(response).model_dump())


def process_results(current_result, sota_result):
    # Convert the results to dataframes
    current_df = pd.DataFrame(current_result)
    sota_df = pd.DataFrame(sota_result)

    # Set the metric as the index
    current_df.index.name = "metric"
    sota_df.index.name = "metric"

    # Rename the value column to reflect the result type
    current_df.rename(columns={"0": "Current Result"}, inplace=True)
    sota_df.rename(columns={"0": "SOTA Result"}, inplace=True)

    # Combine the dataframes on the Metric index
    combined_df = pd.concat([current_df, sota_df], axis=1)

    # Filter the combined DataFrame to retain only the important metrics
    filtered_combined_df = combined_df.loc[IMPORTANT_METRICS]

    def format_filtered_combined_df(filtered_combined_df: pd.DataFrame) -> str:
        results = []
        for metric, row in filtered_combined_df.iterrows():
            current = row["Current Result"]
            sota = row["SOTA Result"]
            results.append(f"{metric} of Current Result is {current:.6f}, of SOTA Result is {sota:.6f}")
        return "; ".join(results)

    return format_filtered_combined_df(filtered_combined_df)


class QlibFactorExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibFactorExperiment): The experiment to generate feedback for.
            hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="factor")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        usr_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis_text,
            task_details=tasks_factors,
            combined_result=combined_result,
        )

        return _validated_feedback(sys_prompt, usr_prompt, FactorFeedbackResponse)


class QlibModelExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibModelExperiment): The experiment to generate feedback for.
            hypothesis (QlibModelHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            HypothesisFeedback: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="model")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()
        user_prompt = T("scenarios.qlib.prompts:model_feedback_generation.user").r(
            sota_hypothesis=SOTA_hypothesis,
            sota_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
            sota_code=SOTA_experiment.sub_workspace_list[0].file_dict.get("model.py") if SOTA_hypothesis else None,
            sota_result=SOTA_experiment.result.loc[IMPORTANT_METRICS] if SOTA_hypothesis else None,
            hypothesis=hypothesis,
            exp=exp,
            exp_result=exp.result.loc[IMPORTANT_METRICS] if exp.result is not None else "execution failed",
        )

        return _validated_feedback(sys_prompt, user_prompt, FeedbackResponse)
