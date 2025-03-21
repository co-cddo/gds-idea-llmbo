from conftest import ExampleOutput

from llmbo.adapters import MistralAdapter
from llmbo.models import ModelInput


def test_build_tool():
    """Test building a tool definition for Mistral."""
    tool = MistralAdapter.build_tool(ExampleOutput)

    # Verify the tool is a string with JSON structure reference
    assert isinstance(tool, str)
    assert "The JSON Structure should be:" in tool
    assert "ExampleOutput" in tool  # The model name should be referenced in the schema


def test_prepare_model_input():
    """Test preparing model input for Mistral."""
    # Test with anthropic_version set
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
        anthropic_version="bedrock-2023-05-31",
    )

    # Prepare for regular use without output model
    result = MistralAdapter.prepare_model_input(model_input)
    assert result.anthropic_version is None
    assert result.messages[0]["content"] == "Test"  # Should not be modified

    # Prepare with schema
    result = MistralAdapter.prepare_model_input(model_input, ExampleOutput)
    assert result.anthropic_version is None
    assert result.tools is None  # Tools are not used in this approach
    assert result.tool_choice is None
    # Check that content was wrapped properly
    assert result.messages[0]["content"].startswith(
        "<s>[INST] Reply with a JSON object."
    )
    assert "The JSON Structure should be:" in result.messages[0]["content"]
    assert result.messages[0]["content"].endswith("[/INST]")


def test_prepare_model_input_empty_content(caplog):
    """Test preparing model input with empty content."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": ""}],
    )

    with caplog.at_level("DEBUG"):
        result = MistralAdapter.prepare_model_input(model_input, ExampleOutput)

    assert result is None
    assert "Didnt find any content to adapt" in caplog.text


def test_validate_result_valid():
    """Test validate_result with a valid input."""
    valid_result = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": 'Here is the JSON object you requested: {"name": "John Doe", "age": 30, "title": "ExampleOutput"}',
                },
            }
        ],
    }

    result = MistralAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "John Doe"
    assert result.age == 30


def test_validate_result_bad_message(caplog):
    """Test the correct None and log is returned for an invalid message."""
    invalid_result = {"not_choices": {}}
    with caplog.at_level("DEBUG"):
        result = MistralAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "No expected 'choices' key in result." in caplog.text


def test_validate_result_wrong_finish_reason(caplog):
    """Test the correct None and log is returned for wrong finish reason."""
    invalid_result = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": '{"name": "John Doe", "age": 30}',
                },
            }
        ]
    }
    with caplog.at_level("DEBUG"):
        result = MistralAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Did not have 'stop' as the finish_reason." in caplog.text


def test_validate_result_wrong_role(caplog):
    """Test validate_result with wrong role."""
    invalid_result = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "user",
                    "content": '{"name": "John Doe", "age": 30}',
                },
            }
        ],
    }
    with caplog.at_level("DEBUG"):
        result = MistralAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Did not get the expected 'assistant' role." in caplog.text


def test_validate_result_no_json(caplog):
    """Test validate_result with no JSON in content."""
    invalid_result = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "There is no JSON here, just text.",
                },
            }
        ],
    }

    with caplog.at_level("DEBUG"):
        result = MistralAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Didnt find anything that looked like JSON in the response" in caplog.text


def test_validate_result_invalid_json(caplog):
    """Test validate_result with invalid JSON."""
    invalid_result = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"name": "John Doe", "age": 30, missing"}',
                },
            }
        ],
    }

    with caplog.at_level("DEBUG"):
        result = MistralAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Failed to parse function arguments as JSON" in caplog.text


def test_validate_result_wrong_schema_name(caplog):
    """Test validate_result with wrong schema name."""
    invalid_result = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"name": "John Doe", "age": 30, "title": "WrongOutput"}',
                },
            }
        ],
    }

    with caplog.at_level("DEBUG"):
        result = MistralAdapter.validate_result(invalid_result, ExampleOutput)

    assert "Wrong schema name in response" in caplog.text
    # Should still pass validation as long as required fields are present
    assert isinstance(result, ExampleOutput)


def test_validate_result_invalid_schema(caplog):
    """Test validate_result with schema validation failure."""
    invalid_result = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"name": "John Doe", "age": "thirty", "title": "ExampleOutput"}',
                },
            }
        ],
    }

    with caplog.at_level("DEBUG"):
        result = MistralAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Validation failed:" in caplog.text


def test_validate_result_with_markdown_json(caplog):
    """Test validate_result with JSON in markdown format."""
    valid_result = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": 'Here\'s the JSON object:\n\n```json\n{"name": "John Doe", "age": 30}\n```\n\nHope this helps!',
                },
            }
        ],
    }

    result = MistralAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "John Doe"
    assert result.age == 30
