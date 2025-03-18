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

    @classmethod
    def build_tool(cls, output_model: type[BaseModel]) -> dict[str, Any]:
        """Build a tool definition in Anthropic's format.

        Args:
            output_model: The Pydantic model to convert to a tool definition

        Returns:
            Dict with name, description, and input_schema keys
        """
        return {
            "name": output_model.__name__,
            "description": output_model.__doc__ or "Please fill in the schema",
            "input_schema": output_model.model_json_schema(),
        }

    @classmethod
    def prepare_model_input(
        cls, model_input: ModelInput, output_model: type[BaseModel]
    ) -> ModelInput:
        """Prepare model input for Anthropic Claude models.

        Args:
            model_input: The original model input configuration
            output_model: The Pydantic model defining the expected output structure

        Returns:
            Modified model input with Anthropic-specific configurations
        """
        # Ensure anthropic_version is set (required for Anthropic models)
        if model_input.anthropic_version is None:
            model_input.anthropic_version = "bedrock-2023-05-31"

        # Build tool from output_model and add it to model_input
        if output_model:
            tool = cls.build_tool(output_model)
            model_input.tools = [tool]
            model_input.tool_choice = ToolChoice(type="tool", name=tool["name"])

        return model_input

    @classmethod
    def validate_result(
        cls, result: dict[str, Any], output_model: type[BaseModel]
    ) -> BaseModel | None:
        """Validate and parse output from Anthropic Claude models.

        Extracts structured data from Claude's tool-use response format and
        validates it against the provided Pydantic model.

        Args:
            result: Raw model output from Claude
            output_model: Pydantic model to validate against

        Returns:
            Validated model instance or None if validation fails
        """
        # Check if the model used a tool (required for structured output)
        if result.get("stop_reason") != "tool_use":
            return None

        # Ensure content exists
        if not result.get("content", []):
            return None

        # Process tool use response
        if (
            len(result["content"]) > 0
            and result["content"][0].get("type") == "tool_use"
        ):
            try:
                # Parse tool use input as our output model
                return output_model(**result["content"][0]["input"])
            except ValidationError:
                return None

        return None
