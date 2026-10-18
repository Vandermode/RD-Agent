"""Validate complete response contracts and give bounded, explicit repair feedback."""

import hashlib
import json

from pydantic import BaseModel, ValidationError

from rdagent.core.exception import FormatError
from rdagent.log import rdagent_logger as logger


class ResponseSchemaError(FormatError):
    """The model did not satisfy the response contract after bounded repairs."""


def validated_json_response(
    backend,
    *,
    user_prompt: str,
    system_prompt: str,
    response_model: type[BaseModel],
    attempts: int = 3,
    validation_context: dict | None = None,
) -> str:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    schema = json.dumps(response_model.model_json_schema(), sort_keys=True)
    schema_id = hashlib.sha256((schema + json.dumps(validation_context, sort_keys=True)).encode()).hexdigest()
    system_prompt += (
        "\nReturn one JSON object satisfying this complete JSON schema. "
        "Required fields must be present at their specified level.\n" + schema
    )
    messages = []
    request_prompt = user_prompt
    for attempt in range(attempts):
        response = backend.build_messages_and_create_chat_completion(
            user_prompt=request_prompt,
            system_prompt=system_prompt,
            former_messages=messages,
            json_mode=True,
            # Validate the complete contract below, including cached responses.
            # A broad JSON target lets us preserve harmless nested narrative text.
            json_target_type=dict,
            chat_cache_prefix=f"schema:{schema_id}:attempt:{attempt}:",
        )
        try:
            validated = response_model.model_validate_json(response, context=validation_context)
            return validated.model_dump_json(by_alias=True)
        except ValidationError as error:
            details = json.dumps(error.errors(include_input=False, include_url=False), default=str)
            logger.warning(f"Response schema validation failed ({attempt + 1}/{attempts}): {details}")
            if attempt + 1 == attempts:
                raise ResponseSchemaError(
                    f"{response_model.__name__} failed after {attempts} attempts: {details}"
                ) from error
            # The next request carries the invalid payload and actionable errors.
            # Its cache key differs from the original request.
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ]
            request_prompt = (
                "Repair only the JSON structure or invalid field values in your previous response. "
                "Preserve the proposed experiment and evidence; do not invent results. "
                f"Validation errors: {details}"
            )
    raise AssertionError("unreachable")
