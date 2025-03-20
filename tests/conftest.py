from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llmbo import ModelInput


class ExampleOutput(BaseModel):
    """Test output model."""

    name: str
    age: int


@pytest.fixture
def mock_iam_client():
    """Create a mock IAM client."""
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
    from llmbo import BatchInferer

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
    from llmbo import StructuredBatchInferer

    return StructuredBatchInferer(
        model_name="test-supported-claude-model",
        bucket_name="test-bucket",
        region="test-region",
        job_name="test-job",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        output_model=ExampleOutput,
    )
