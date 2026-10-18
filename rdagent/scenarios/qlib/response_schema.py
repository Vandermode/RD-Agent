"""Typed contracts for Qlib research proposals and feedback."""

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, RootModel, ValidationInfo, field_validator, model_validator


def narrative(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        # Preserve structured explanations instead of rejecting or dropping them.
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    raise ValueError("Expected narrative text, an object, or a list")


Narrative = Annotated[str, BeforeValidator(narrative)]


class HypothesisResponse(BaseModel):
    hypothesis: Narrative
    reason: Narrative
    concise_reason: Narrative
    concise_observation: Narrative
    concise_justification: Narrative
    concise_knowledge: Narrative


class QuantHypothesisResponse(HypothesisResponse):
    action: Literal["factor", "model"]

    @model_validator(mode="after")
    def selected_action(self, info: ValidationInfo):
        expected = (info.context or {}).get("expected_action")
        if expected in ("factor", "model") and self.action != expected:
            raise ValueError(f"The controller selected action={expected}; keep that action")
        return self


class FactorSpecification(BaseModel):
    description: Narrative
    formulation: Narrative
    variables: dict[str, Any]


class FactorProposalResponse(RootModel):
    root: dict[str, FactorSpecification] = Field(min_length=1)


class ModelSpecification(FactorSpecification):
    architecture: Narrative
    hyperparameters: dict[str, Any]
    training_hyperparameters: dict[str, Any]
    model_type: Literal["Tabular", "TimeSeries"]

    @model_validator(mode="before")
    @classmethod
    def normalize_model_type(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        hyperparameters = value.get("hyperparameters")
        if isinstance(hyperparameters, dict) and "model_type" in hyperparameters:
            nested = hyperparameters["model_type"]
            if "model_type" in value and value["model_type"] != nested:
                raise ValueError("Conflicting top-level and nested model_type values")
            value["model_type"] = nested
            value["hyperparameters"] = {k: v for k, v in hyperparameters.items() if k != "model_type"}
        return value


class ModelProposalResponse(RootModel):
    # The runner evaluates one model, so never silently discard additional ones.
    root: dict[str, ModelSpecification] = Field(min_length=1, max_length=1)


class FeedbackResponse(BaseModel):
    observations: Narrative = Field(alias="Observations")
    hypothesis_evaluation: Narrative = Field(alias="Feedback for Hypothesis")
    new_hypothesis: Narrative = Field(alias="New Hypothesis")
    reason: Narrative = Field(alias="Reasoning")
    decision: bool = Field(alias="Decision")

    @field_validator("decision", mode="before")
    @classmethod
    def explicit_decision(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "yes", "false", "no"}:
            return value.strip().lower() in {"true", "yes"}
        if type(value) is int and value in (0, 1):
            return bool(value)
        raise ValueError("Decision must explicitly be true/false or yes/no")


class FactorFeedbackResponse(FeedbackResponse):
    decision: bool = Field(alias="Replace Best Result")
