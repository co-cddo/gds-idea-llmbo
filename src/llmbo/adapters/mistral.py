import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from ..models import ModelInput
from .base import ModelProviderAdapter


class MistralAdapter(ModelProviderAdapter):
    """Adapter for Mistral models in AWS Bedrock.

    This adapter handles:
    1. Formatting inputs for Mistral models
    2. Building tool definitions in Mistral's format
    3. Validating tool-use responses from Mistral models
    """

    logger = logging.getLogger(f"{__name__}.MistralAdapter")

    @classmethod
    def build_tool(cls, output_model: type[BaseModel]) -> dict[str, Any]:
        """Build a tool definition in Mistral's format.

        Args:
            output_model: The Pydantic model to convert to a tool definition

        Returns:
            Dict with function definition for Mistral's tools format
        """
        cls.logger.debug(f"Building tool definition for model: {output_model.__name__}")

        schema = output_model.model_json_schema()

        tool = f"""
        The JSON Structure should be: 
        {schema}
        """

        cls.logger.debug(f"Created tool definition with name: {output_model.__name__}")
        return tool

    @classmethod
    def prepare_model_input(
        cls, model_input: ModelInput, output_model: type[BaseModel] | None = None
    ) -> ModelInput:
        """Prepare model input for Mistral models.

        Args:
            model_input: The original model input configuration
            output_model: The Pydantic model defining the expected output structure

        Returns:
            Modified model input with Mistral-specific configurations
        """
        cls.logger.debug("Preparing model input for Mistral")

        # Mistral doesn't use anthropic_version, remove if set
        model_input.anthropic_version = None

        # Build tool from output_model and add it to model_input
        if output_model:
            cls.logger.debug(f"Adding tool definition for {output_model.__name__}")
            tool = cls.build_tool(output_model)

            original_prompt = model_input.messages[0].get("content", "")
            if original_prompt:
                model_input.messages[0]["content"] = (
                    f"<s>[INST] Reply with a JSON object. {original_prompt + tool} [/INST]"
                )
            else:
                cls.logger.debug("Didnt find any content to adapt")

            # Ensure there are no other tools present
            model_input.tools = None
            model_input.tool_choice = None
        return model_input

    @classmethod
    def validate_result(
        cls, result: dict[str, Any], output_model: type[BaseModel]
    ) -> BaseModel | None:
        """Validate and parse output from Mistral models.

        Extracts structured data from Mistral's tool-use response format and
        validates it against the provided Pydantic model.

        Args:
            result: Raw model output from Mistral
            output_model: Pydantic model to validate against

        Returns:
            Validated model instance or None if validation fails
        """
        cls.logger.debug(f"Validating result against {output_model.__name__} schema")

        # Check we have choices
        choices = result.get("choices", [])
        if not choices:
            cls.logger.debug("No expected 'choices' key in result.")
            return None

        # Check that we stopped on purpose
        if choices[0].get("finish_reason", "") != "stop":
            cls.logger.debug("Did not have 'stop' as the finish_reason.")
            return None

        # Check that the assistant returned a message
        if choices[0].get("message", {}).get("role", "") != "assistant":
            cls.logger.debug("Did not get the expected 'assistant' role.")
            return None

        # Check the content has something json looking.
        content = choices[0].get("message", {}).get("content", "")
        if content:
            match = re.search(r"\{.*\}", content, re.DOTALL)

        if not match:
            cls.logger.debug(
                "Didnt find anything that looked like JSON in the response"
            )
            return None

        try:
            arguments = match.group(0)
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            cls.logger.debug(f"Failed to parse function arguments as JSON: {arguments}")
            return None

        tool_name = parsed_arguments.get("title", "")
        if tool_name != output_model.__name__:
            cls.logger.debug(
                f"Wrong schema name in response, "
                f"expected {output_model.__name__} got {tool_name}"
            )

        try:
            validated_model = output_model(**parsed_arguments)
            cls.logger.debug("Validation successful.")
            return validated_model
        except ValidationError as e:
            cls.logger.debug(f"Validation failed: {str(e)}")
            return None
