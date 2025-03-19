from conftest import ExampleOutput

from llmbo.adapters import MistralAdapter
from llmbo.models import ModelInput

expected_tool_definition = {
    "type": "function",
    "function": {
        "name": "ExampleOutput",
        "description": "Test output model.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name of the person."},
                "age": {"type": "integer", "description": "The age of the person."},
            },
            "required": ["name", "age"],
        },
    },
}


def test_build_tool():
    """Test building a tool definition for Mistral."""
    tool = MistralAdapter.build_tool(ExampleOutput)

    # Verify the tool structure
    assert tool["type"] == "function"
    assert "function" in tool
    assert tool["function"]["name"] == "ExampleOutput"
    assert "parameters" in tool["function"]
    assert "name" in tool["function"]["parameters"]["properties"]
    assert "age" in tool["function"]["parameters"]["properties"]


def test_prepare_model_input():
    """Test preparing model input for Mistral."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
        anthropic_version="bedrock-2023-05-31",
    )

    # Prepare for regular use
    result = MistralAdapter.prepare_model_input(model_input)
    assert result.anthropic_version is None

    # Prepare with schema
    result = MistralAdapter.prepare_model_input(model_input, ExampleOutput)
    assert result.tools is not None
    assert len(result.tools) == 1
    assert result.tools[0]["type"] == "function"
    assert result.tool_choice == "any"


def test_validate_result_valid():
    """Test validate_result with a valid input."""
    valid_result = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "ExampleOutput",
                    "arguments": {"name": "John Doe", "age": 30},
                },
            }
        ]
    }

    result = MistralAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "John Doe"
    assert result.age == 30


def test_validate_result_string_arguments():
    """Test validate_result with string arguments (which happens with some models)."""
    valid_result = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "ExampleOutput",
                    "arguments": '{"name": "John Doe", "age": 30}',
                },
            }
        ]
    }

    result = MistralAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "John Doe"
    assert result.age == 30


def test_validate_result_no_tool_calls():
    """Test validate_result with no tool calls."""
    invalid_result = {"content": "Some text response"}
    result = MistralAdapter.validate_result(invalid_result, ExampleOutput)
    assert result is None


def test_validate_result_wrong_tool():
    """Test validate_result with wrong tool name."""
    invalid_result = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "DifferentTool",
                    "arguments": {"name": "Jane Doe", "age": 25},
                },
            }
        ]
    }

    result = MistralAdapter.validate_result(invalid_result, ExampleOutput)
    assert result is None


def test_validate_result_invalid_schema():
    """Test validate_result with schema validation failure."""
    invalid_result = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "ExampleOutput",
                    "arguments": {
                        "name": "John Doe",
                        "age": "thirty",
                    },  # age should be int
                },
            }
        ]
    }

    result = MistralAdapter.validate_result(invalid_result, ExampleOutput)
    assert result is None
