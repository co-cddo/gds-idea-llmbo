from unittest.mock import MagicMock

from conftest import ExampleOutput

from llmbo import StructuredBatchInferer
from llmbo.adapters import AnthropicAdapter


def test_structured_init(mock_boto3_session: MagicMock):
    """Test StructuredBatchInferer initialization."""
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

    # Test internal state initialization
    assert bi.job_arn is None
    assert bi.job_status is None
    assert bi.results is None
    assert bi.manifest is None
    assert bi.requests is None


def test_structured_init_default_output_dir(mock_boto3_session: MagicMock):
    """Test StructuredBatchInferer defaults output_dir to current directory."""
    sbi = StructuredBatchInferer(
        output_model=ExampleOutput,
        model_name="test-supported-claude-model",
        bucket_name="test-bucket",
        region="test-region",
        job_name="test-job",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
    )
    assert sbi.output_dir == "."


def test_structured_init_custom_output_dir(mock_boto3_session: MagicMock):
    """Test StructuredBatchInferer passes output_dir through to BatchInferer."""
    sbi = StructuredBatchInferer(
        output_model=ExampleOutput,
        model_name="test-supported-claude-model",
        bucket_name="test-bucket",
        region="test-region",
        job_name="test-job",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        output_dir="structured_outputs",
    )
    assert sbi.output_dir == "structured_outputs"
    assert sbi._local_path("foo.jsonl") == "structured_outputs/foo.jsonl"
