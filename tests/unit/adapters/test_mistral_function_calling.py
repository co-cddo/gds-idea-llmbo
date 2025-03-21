from conftest import ExampleOutput

from llmbo.adapters import MistralFunctionAdapter
from llmbo.models import ModelInput

# expected_tool_definition = {
#     "type": "function",
#     "function": {
#         "name": "ExampleOutput",
#         "description": "Test output model.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "name": {"type": "string", "description": "The name of the person."},
#                 "age": {"type": "integer", "description": "The age of the person."},
#             },
#             "required": ["name", "age"],
#         },
#     },
# }

# {'choices': [{'context_logits': None,
#               'finish_reason': 'tool_calls',
#               'generation_logits': None,
#               'index': 0,
#               'logprobs': None,
#               'message': {'content': '',
#                           'index': None,
#                           'role': 'assistant',
#                           'tool_call_id': None,
#                           'tool_calls': [{'function': {'arguments': '{"name": '
#                                                                     '"Otis", '
#                                                                     '"breed": '
#                                                                     '"Schnauzer", '
#                                                                     '"age": 3}',
#                                                        'name': 'Dog'},
#                                           'id': '8GCjLhr7p',
#                                           'type': 'function'}]}}],
#  'created': 1742397496,
#  'id': '2a8ca221-74c2-457b-98ee-9cab78a43c1a',
#  'model': 'mistral-large-2407',
#  'object': 'chat.completion',
#  'usage': {'completion_tokens': 37, 'prompt_tokens': 119, 'total_tokens': 156}}


def test_build_tool():
    """Test building a tool definition for Mistral."""
    tool = MistralFunctionAdapter.build_tool(ExampleOutput)

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
    result = MistralFunctionAdapter.prepare_model_input(model_input)
    assert result.anthropic_version is None

    # Prepare with schema
    result = MistralFunctionAdapter.prepare_model_input(model_input, ExampleOutput)
    assert result.tools is not None
    assert len(result.tools) == 1
    assert result.tools[0]["type"] == "function"
    assert result.tool_choice == "any"


def test_validate_result_valid():
    """Test validate_result with a valid input."""
    valid_result = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_call_id": None,
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"name": "John Doe",  "age": 30}',
                                "name": "ExampleOutput",
                            },
                            "type": "function",
                        }
                    ],
                },
            }
        ],
    }

    result = MistralFunctionAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "John Doe"
    assert result.age == 30


def test_validate_result_bad_message(caplog):
    """Test the correct None and log is returned for an invalid message."""
    invalid_result = {"not_choices": {}}
    with caplog.at_level("DEBUG"):
        result = MistralFunctionAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "No expected 'choices' key in result." in caplog.text


def test_validate_result_no_tool_use(caplog):
    """Test the correct None and log is returned for an invalid message."""
    invalid_result = {
        "choices": [
            {
                "finish_reason": "stop",
            }
        ]
    }
    with caplog.at_level("DEBUG"):
        result = MistralFunctionAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Finish reason was not 'tool_choice'" in caplog.text


def test_validate_result_empty_tool_calls(caplog):
    """Test validate_result with no tool calls."""
    invalid_result = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_call_id": None,
                    "tool_calls": [],
                },
            }
        ],
    }
    with caplog.at_level("DEBUG"):
        result = MistralFunctionAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "No tool_calls in message" in caplog.text


def test_validate_result_too_many_tool_calls(caplog):
    """Test validate_result with no tool calls."""
    invalid_result = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_call_id": None,
                    "tool_calls": [0, 1, 2],
                },
            }
        ],
    }
    with caplog.at_level("DEBUG"):
        result = MistralFunctionAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Too many (3) tools called." in caplog.text


def test_validate_result_wrong_tool(caplog):
    """Test validate_result with wrong tool name."""
    invalid_result = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_call_id": None,
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"name": "John Doe",  "age": 30}',
                                "name": "WrongName",
                            },
                            "type": "function",
                        }
                    ],
                },
            }
        ],
    }

    with caplog.at_level("DEBUG"):
        result = MistralFunctionAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert (
        "Wrong tool encountered, expected ExampleOutput got WrongName." in caplog.text
    )


def test_validate_result_invalid_json(caplog):
    """Test validate_result with bad json.

    age should be an integer, here is is a string.
    """
    invalid_result = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_call_id": None,
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"name": "John Doe",  "age": 30, "missing"}',
                                "name": "ExampleOutput",
                            },
                            "type": "function",
                        }
                    ],
                },
            }
        ],
    }

    with caplog.at_level("DEBUG"):
        result = MistralFunctionAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Failed to parse" in caplog.text


def test_validate_result_invalid_schema(caplog):
    """Test validate_result with schema validation failure.

    age should be an integer, here is is a string.
    """
    invalid_result = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_call_id": None,
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"name": "John Doe",  "age": "thirty"}',
                                "name": "ExampleOutput",
                            },
                            "type": "function",
                        }
                    ],
                },
            }
        ],
    }

    with caplog.at_level("DEBUG"):
        result = MistralFunctionAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Validation failed:" in caplog.text
