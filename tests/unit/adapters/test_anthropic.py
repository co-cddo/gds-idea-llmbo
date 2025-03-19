from conftest import ExampleOutput

from llmbo.adapters import AnthropicAdapter


def test_validate_result_valid():
    """Test validate_result with a valid input."""
    valid_result = {
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "input": {"name": "John Doe", "age": 30}}],
    }
    result = AnthropicAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "John Doe"
    assert result.age == 30


def test_validate_result_wrong_stop_reason():
    """Test validate_result with wrong stop reason."""
    invalid_result = {
        "stop_reason": "max_tokens",
        "content": [{"type": "tool_use", "input": {"name": "Jane Doe", "age": 25}}],
    }
    result = AnthropicAdapter.validate_result(invalid_result, ExampleOutput)
    assert result is None


def test_validate_result_multiple_contents():
    """Test validate_result with multiple content items."""
    invalid_result = {
        "stop_reason": "tool_use",
        "content": [
            {"type": "tool_use", "input": {"name": "Alice", "age": 28}},
            {"type": "tool_use", "input": {"name": "Bob", "age": 32}},
        ],
    }
    result = AnthropicAdapter.validate_result(invalid_result, ExampleOutput)
    assert result is None


def test_validate_result_wrong_schema():
    """Test validate_result with input that doesn't match the schema."""
    invalid_result = {
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "tool_use",
                "input": {
                    "name": "Charlie",
                    "age": "thirty",  # Age should be an integer
                },
            }
        ],
    }
    result = AnthropicAdapter.validate_result(invalid_result, ExampleOutput)
    assert result is None


def test_validate_result_missing_content():
    """Test validate_result with missing content."""
    invalid_result = {"stop_reason": "tool_use"}
    result = AnthropicAdapter.validate_result(invalid_result, ExampleOutput)
    assert result is None


def test_validate_result_wrong_content_type():
    """Test validate_result with wrong content type."""
    invalid_result = {
        "stop_reason": "tool_use",
        "content": [{"type": "text", "text": "This is not a tool use."}],
    }
    result = AnthropicAdapter.validate_result(invalid_result, ExampleOutput)
    assert result is None
