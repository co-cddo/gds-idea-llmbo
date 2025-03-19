import logging
from typing import Any, Dict, Literal

from pydantic import BaseModel, ValidationError

from ..models import ModelInput, ToolChoice
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
        tool = {
            "type": "function",
            "function": {
                "name": output_model.__name__,
                "description": schema.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }

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
            model_input.tools = [tool]
            model_input.tool_choice = "any"
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

        # Check if we have tool_calls in the response
        tool_calls = result.get("tool_calls", [])
        if not tool_calls:
            cls.logger.debug("No tool_calls found in result")
            return None

        # Find the matching tool call for our output model
        for tool_call in tool_calls:
            if (
                tool_call.get("type") == "function"
                and tool_call.get("function", {}).get("name") == output_model.__name__
            ):
                try:
                    # Parse arguments as JSON and validate against our model
                    arguments = tool_call.get("function", {}).get("arguments", {})
                    if isinstance(arguments, str):
                        import json

                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            cls.logger.debug(
                                f"Failed to parse arguments as JSON: {arguments}"
                            )
                            return None

                    instance = output_model(**arguments)
                    cls.logger.debug(
                        f"Successfully validated result as {output_model.__name__}"
                    )
                    return instance
                except ValidationError as e:
                    cls.logger.debug(f"Validation failed: {str(e)}")
                    return None

        cls.logger.debug(f"No matching tool call found for {output_model.__name__}")
        return None
