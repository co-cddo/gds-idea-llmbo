from conftest import ExampleOutput

from llmbo.adapters import MistralAdapter
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

    result = MistralAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "John Doe"
    assert result.age == 30


# These need editing to reflect the structure of a real call and to check the log
# to make sure the right message pops up.
def test_validate_result_no_tool_calls():
    """Test validate_result with no tool calls."""
    invalid_result = valid_result = {
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
    result = MistralAdapter.validate_result(invalid_result, ExampleOutput)
    assert result is None


def test_validate_result_wrong_tool():
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

    result = MistralAdapter.validate_result(invalid_result, ExampleOutput)
    assert result is None


def test_validate_result_invalid_schema():
    """Test validate_result with schema validation failure."""
    invalid_result = valid_result = {
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

    result = MistralAdapter.validate_result(invalid_result, ExampleOutput)
    assert result is None
