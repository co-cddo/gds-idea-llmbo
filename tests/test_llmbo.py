from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from pydantic import BaseModel

from llmbo import BatchInferer, ModelInput, StructuredBatchInferer
from llmbo.adapters import AnthropicAdapter


class ExampleOutput(BaseModel):  # noqa: D101
    name: str
    age: int


@pytest.fixture
def mock_iam_client():
    """Create a mock iam client."""
    mock_client = MagicMock()
    mock_client.get_role.return_value = {}
    return mock_client


@pytest.fixture
def mock_bedrock_client():
    """Create a mock Bedrock client with expected responses."""
    mock_client = MagicMock()

    # Configure mock responses
    mock_client.create_model_invocation_job.return_value = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "jobArn": "arn:aws:bedrock:region:account:job/test-job",
    }

    mock_client.get_model_invocation_job.return_value = {
        "status": "Completed",
        "jobName": "test-job",
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "inputDataConfig": {
            "s3InputDataConfig": {"s3Uri": "s3://test-bucket/input/test.jsonl"}
        },
        "roleArn": "arn:aws:iam::123456789012:role/TestRole",
    }

    mock_client.stop_model_invocation_job.return_value = {
        "ResponseMetadata": {"HTTPStatusCode": 200}
    }

    return mock_client


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    mock_client = MagicMock()
    mock_client.upload_file.return_value = None
    mock_client.download_file.return_value = None
    mock_client.get_bucket_location.return_value = {"LocationConstraint": "test-region"}
    return mock_client


@pytest.fixture
def sample_inputs():
    """Create minimal valid inputs for testing."""
    return {
        f"{i:03}": ModelInput(
            messages=[{"role": "user", "content": "Test message"}],
        )
        for i in range(100)
    }


@pytest.fixture
def mock_boto3_session(
    mock_bedrock_client: MagicMock,
    mock_s3_client: MagicMock,
    mock_iam_client: MagicMock,
):
    """Create a mock boto3 client that returns appropriate service clients."""
    with patch("boto3.Session") as mock_session:
        mock_session_instance = mock_session.return_value

        def mock_client(service_name, region_name=None):
            return {
                "bedrock": mock_bedrock_client,
                "s3": mock_s3_client,
                "iam": mock_iam_client,
            }.get(service_name, MagicMock())

        mock_session_instance.client.side_effect = mock_client
        yield mock_session


@pytest.fixture
def batch_inferer(mock_boto3_session: MagicMock | AsyncMock):
    """Create a configured BatchInferer instance for testing."""
    return BatchInferer(
        model_name="test-supported-claude-model",
        bucket_name="test-bucket",
        region="test-region",
        job_name="test-job",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
    )


@pytest.fixture
def structured_batch_inferer(mock_boto3_session: MagicMock | AsyncMock):
    """Create a configured StructuredBatchInferer instance for testing."""
    return StructuredBatchInferer(
        model_name="test-supported-claude-model",
        bucket_name="test-bucket",
        region="test-region",
        job_name="test-job",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        output_model=ExampleOutput,
    )


def test_init(mock_boto3_session: MagicMock | AsyncMock):
    """Test BatchInferer initialisation."""
    inputs = {
        "model_name": "test-supported-claude-model",
        "bucket_name": "test-bucket",
        "job_name": "test-job",
        "region": "test-region",
        "role_arn": "arn:aws:iam::123456789012:role/TestRole",
    }
    bi = BatchInferer(**inputs)

    # Test attribute assignment
    assert bi.model_name == inputs["model_name"]
    assert bi.bucket_name == inputs["bucket_name"]
    assert bi.job_name == inputs["job_name"]
    assert bi.role_arn == inputs["role_arn"]
    assert bi.region == inputs["region"]
    assert bi.adapter is AnthropicAdapter

    # Test S3 bucket check was called
    mock_boto3_session.return_value.client("s3").head_bucket.assert_called_once_with(
        Bucket=inputs["bucket_name"]
    )

    # Test IAM role check was called
    mock_boto3_session.return_value.client("iam").get_role.assert_called_once_with(
        RoleName=inputs["role_arn"].split("/")[-1]  # Should be "TestRole"
    )

    # Test internal state initialization
    assert bi.job_arn is None
    assert bi.job_status is None
    assert bi.results is None
    assert bi.manifest is None
    assert bi.requests is None

    # Test derived attributes
    assert bi.bucket_uri == f"s3://{inputs['bucket_name']}"
    assert bi.file_name == f"{inputs['job_name']}.jsonl"

    # Test that boto3.Session().client was called for each service
    mock_boto3_session.return_value.client.assert_has_calls(
        [call("s3"), call("iam"), call("bedrock", region_name=inputs["region"])],
        any_order=True,
    )


def test_init_unsupported_model():
    """Test BatchInferer initialisation with an unsupported model raises the correct error"""
    inputs = {
        "model_name": "test-unsupported-model",
        "bucket_name": "test-bucket",
        "job_name": "test-job",
        "region": "test-region",
        "role_arn": "arn:aws:iam::123456789012:role/TestRole",
    }
    # Expect ValueError about no adapter available
    with pytest.raises(
        ValueError, match=f"No adapter available for model {inputs['model_name']}"
    ):
        BatchInferer(**inputs)


def test_prepare_requests(
    batch_inferer: BatchInferer, sample_inputs: dict[str, ModelInput]
):
    """Test that requests are prepared"""
    batch_inferer.prepare_requests(sample_inputs)

    assert len(batch_inferer.requests) == len(sample_inputs), (
        "requests do not have the expected length"
    )
    assert list(batch_inferer.requests[0].keys()) == ["recordId", "modelInput"]
    assert batch_inferer.requests[0]["recordId"] == "000"
    assert "anthropic_version" in batch_inferer.requests[0]["modelInput"].keys()
    assert all(
        [isinstance(request["modelInput"], dict) for request in batch_inferer.requests]
    ), "requests are not of expected type "


def test_prepare_requests_bad_batch_size(
    batch_inferer: BatchInferer, sample_inputs: dict[str, ModelInput]
):
    """Test that an error is raised for batch size < 100."""
    small_inputs = dict(list(sample_inputs.items())[:50])
    with pytest.raises(
        ValueError, match=f"Minimum Batch Size is 100, {len(small_inputs)} given"
    ):
        batch_inferer.prepare_requests(small_inputs)


def test_create_job(batch_inferer: BatchInferer, sample_inputs: dict[str, ModelInput]):
    """Test job creation with mocked AWS clients."""
    batch_inferer.prepare_requests(sample_inputs)
    response = batch_inferer.create()

    # Assert the job was created
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert batch_inferer.job_arn is not None

    # Check the corect function was called
    batch_inferer.client.create_model_invocation_job.assert_called_once()


def test_create_fail_no_requests(batch_inferer: BatchInferer):
    """Test failure with no set requests."""
    with pytest.raises(AttributeError):
        batch_inferer.create()


def test_create_fail_http_error(
    batch_inferer: BatchInferer,
    sample_inputs: dict[str, ModelInput],
):
    batch_inferer.client.create_model_invocation_job.return_value = {
        "ResponseMetadata": {"HTTPStatusCode": 400}
    }

    batch_inferer.prepare_requests(sample_inputs)

    with pytest.raises(
        RuntimeError,
        match=r"There was an error creating the job .*, non 200 response from bedrock",
    ):
        batch_inferer.create()


def test_create_fail_no_response(
    batch_inferer: BatchInferer, sample_inputs: dict[str, ModelInput]
):
    batch_inferer.client.create_model_invocation_job.return_value = None

    batch_inferer.prepare_requests(sample_inputs)

    with pytest.raises(
        RuntimeError,
        match="There was an error creating the job, no response from bedrock",
    ):
        batch_inferer.create()


def test_structured_init(mock_boto3_session: MagicMock | AsyncMock):
    """Test BatchInferer initialisation."""
    inputs = {
        "model_name": "test-supported-claude-model",
        "bucket_name": "test-bucket",
        "job_name": "test-job",
        "region": "test-region",
        "role_arn": "arn:aws:iam::123456789012:role/TestRole",
        "output_model": ExampleOutput,
    }
    bi = StructuredBatchInferer(**inputs)

    # Test attribute assignment
    assert bi.model_name == inputs["model_name"]
    assert bi.bucket_name == inputs["bucket_name"]
    assert bi.job_name == inputs["job_name"]
    assert bi.role_arn == inputs["role_arn"]
    assert bi.region == inputs["region"]
    assert bi.output_model == ExampleOutput
    assert bi.adapter is AnthropicAdapter

    # Test S3 bucket check was called
    mock_boto3_session.return_value.client("s3").head_bucket.assert_called_once_with(
        Bucket=inputs["bucket_name"]
    )

    # Test IAM role check was called
    mock_boto3_session.return_value.client("iam").get_role.assert_called_once_with(
        RoleName=inputs["role_arn"].split("/")[-1]  # Should be "TestRole"
    )

    # Test internal state initialization
    assert bi.job_arn is None
    assert bi.job_status is None
    assert bi.results is None
    assert bi.manifest is None
    assert bi.requests is None

    # Test derived attributes
    assert bi.bucket_uri == f"s3://{inputs['bucket_name']}"
    assert bi.file_name == f"{inputs['job_name']}.jsonl"

    # Test that boto3.Session().client was called for each service
    mock_boto3_session.return_value.client.assert_has_calls(
        [call("s3"), call("iam"), call("bedrock", region_name=inputs["region"])],
        any_order=True,
    )


def test_validate_result_valid(structured_batch_inferer: StructuredBatchInferer):
    """Test validate_result with a valid input."""
    valid_result = {
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "input": {"name": "John Doe", "age": 30}}],
    }
    result = structured_batch_inferer.adapter.validate_result(
        valid_result, ExampleOutput
    )
    assert isinstance(result, ExampleOutput)
    assert result.name == "John Doe"
    assert result.age == 30


def test_validate_result_wrong_stop_reason(
    structured_batch_inferer: StructuredBatchInferer,
):
    """Test validate_result with wrong stop reason."""
    invalid_result = {
        "stop_reason": "max_tokens",
        "content": [{"type": "tool_use", "input": {"name": "Jane Doe", "age": 25}}],
    }
    result = structured_batch_inferer.adapter.validate_result(
        invalid_result, ExampleOutput
    )
    assert result is None


def test_validate_result_multiple_contents(
    structured_batch_inferer: StructuredBatchInferer,
):
    """Test validate_result with multiple content items."""
    invalid_result = {
        "stop_reason": "tool_use",
        "content": [
            {"type": "tool_use", "input": {"name": "Alice", "age": 28}},
            {"type": "tool_use", "input": {"name": "Bob", "age": 32}},
        ],
    }
    result = structured_batch_inferer.adapter.validate_result(
        invalid_result, ExampleOutput
    )
    assert result is None


def test_validate_result_wrong_schema(structured_batch_inferer: StructuredBatchInferer):
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
    result = structured_batch_inferer.adapter.validate_result(
        invalid_result, ExampleOutput
    )
    assert result is None


def test_validate_result_missing_content(
    structured_batch_inferer: StructuredBatchInferer,
):
    """Test validate_result with missing content."""
    invalid_result = {"stop_reason": "tool_use"}
    result = structured_batch_inferer.adapter.validate_result(
        invalid_result, ExampleOutput
    )
    assert result is None


def test_validate_result_wrong_content_type(
    structured_batch_inferer: StructuredBatchInferer,
):
    """Test validate_result with wrong content type."""
    invalid_result = {
        "stop_reason": "tool_use",
        "content": [{"type": "text", "text": "This is not a tool use."}],
    }
    result = structured_batch_inferer.adapter.validate_result(
        invalid_result, ExampleOutput
    )
    assert result is None
