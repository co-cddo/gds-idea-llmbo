import json
from dataclasses import dataclass
from typing import Literal


@dataclass
class Manifest:
    """Job manifest details."""

    totalRecordCount: int
    processedRecordCount: int
    successRecordCount: int
    errorRecordCount: int
    inputTokenCount: int | None = None
    outputTokenCount: int | None = None


@dataclass
class ToolChoice:
    """Toolchoice details."""

    type: Literal["any", "tool", "auto"]
    name: str | None = None


@dataclass
class ModelInput:
    """Configuration class for AWS Bedrock model inputs.

    This class defines the structure and parameters for model invocation requests
    following AWS Bedrock's expected format.

    See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

    Attributes:
        messages (List[dict]): List of message objects with role and content
        anthropic_version (str): Version string for Anthropic models
        max_tokens (int): Maximum number of tokens in the response
        system (Optional[str]): System message for the model
        stop_sequences (Optional[List[str]]): Custom stop sequences
        temperature (Optional[float]): Sampling temperature
        top_p (Optional[float]): Nucleus sampling parameter
        top_k (Optional[int]): Top-k sampling parameter
        tools (Optional[List[dict]]): Tool definitions for structured outputs
        tool_choice (Optional[ToolChoice]): Tool selection configuration
    """

    # These are required
    messages: list[dict]
    anthropic_version: str = "bedrock-2023-05-31"
    max_tokens: int = 2000

    system: str | None = None
    stop_sequences: list[str] | None | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    tools: list[dict] | None | None = None
    tool_choice: ToolChoice | None = None

    def to_dict(self):
        """Convert to dict."""
        result = {k: v for k, v in self.__dict__.items() if v is not None}
        if self.tool_choice:
            result["tool_choice"] = self.tool_choice.__dict__
        return result

    def to_json(self):
        """Convert to json string."""
        return json.dumps(self.to_dict())
