import logging
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ValidationError

from ..models import ModelInput, ToolChoice
from .base import ModelProviderAdapter


class AnthropicAdapter(ModelProviderAdapter):
    """Adapter for Anthropic Claude models in AWS Bedrock.

    This adapter handles:
    1. Setting the required anthropic_version
    2. Building tool definitions in Anthropic's format
    3. Validating tool-use responses from Claude models
    """

    logger = logging.getLogger(f"{__name__}.AnthropicAdapter")

    @classmethod
    @lru_cache(maxsize=1)
    def build_tool(cls, output_model: type[BaseModel]) -> dict[str, Any]:
        """Build a tool definition in Anthropic's format."""
        cls.logger.debug(f"Building tool definition for model: {output_model.__name__}")
        tool = {
            "name": output_model.__name__,
            "description": output_model.__doc__ or "Please fill in the schema",
            "input_schema": output_model.model_json_schema(),
        }
        cls.logger.debug(f"Created tool definition with name: {tool['name']}")
        return tool

    @classmethod
    def prepare_model_input(cls, model_input: ModelInput, output_model: type[BaseModel] | None = None) -> ModelInput:
        """Prepare model input for Anthropic Claude models."""
        cls.logger.debug("Preparing model input for Anthropic Claude")

        # Ensure anthropic_version is set (required for Anthropic models)
        if model_input.anthropic_version is None:
            cls.logger.debug("Setting default anthropic_version")
            model_input.anthropic_version = "bedrock-2023-05-31"

        # Build tool from output_model and add it to model_input
        if output_model:
            cls.logger.debug(f"Adding tool definition for {output_model.__name__}")
            tool = cls.build_tool(output_model)
            model_input.tools = [tool]
            model_input.tool_choice = ToolChoice(type="tool", name=tool["name"])

        return model_input

    @classmethod
    def validate_result(cls, result: dict[str, Any], output_model: type[BaseModel]) -> BaseModel | None:
        """Validate and parse output from Anthropic Claude models."""
        cls.logger.debug(f"Validating result against {output_model.__name__} schema")

        if result.get("stop_reason") != "tool_use":
            cls.logger.debug(f"Invalid stop_reason: {result.get('stop_reason')}")
            return None

        # Ensure content exists
        content = result.get("content", [])
        if not content:
            cls.logger.debug("Result contains no content")
            return None

        # Check that there's exactly one tool call
        tool_use_items = [item for item in content if item.get("type") == "tool_use"]
        if len(tool_use_items) != 1:
            cls.logger.debug(f"Expected exactly 1 tool_use item, found {len(tool_use_items)}")
            return None

        # Process the single tool use response
        tool_use = tool_use_items[0]
        try:
            # Parse tool use input as our output model
            instance = output_model(**tool_use["input"])
            cls.logger.debug(f"Successfully validated result as {output_model.__name__}")
            return instance
        except ValidationError as e:
            cls.logger.debug(f"Validation failed: {e!s}")
            return None
