from unittest.mock import AsyncMock, MagicMock, call

import pytest

from llmbo import BatchInferer, ModelInput
from llmbo.adapters import DefaultAdapter


def test_init(mock_boto3_session: MagicMock):
    """Test BatchInferer initialization."""
    from llmbo.adapters import AnthropicAdapter

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


def test_init_unsupported_model(mock_boto3_session: MagicMock | AsyncMock):
    """Test BatchInferer initialisation with an unsupported model raises the correct error."""
    inputs = {
        "model_name": "test-unsupported-model",
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
    assert bi.adapter is DefaultAdapter


def test_prepare_requests(
    batch_inferer: BatchInferer, sample_inputs: dict[str, ModelInput]
):
    """Test that requests are prepared."""
    batch_inferer.prepare_requests(sample_inputs)

    assert len(batch_inferer.requests) == len(sample_inputs)
    assert list(batch_inferer.requests[0].keys()) == ["recordId", "modelInput"]
    assert batch_inferer.requests[0]["recordId"] == "000"
    assert "anthropic_version" in batch_inferer.requests[0]["modelInput"]
    assert all(
        [isinstance(request["modelInput"], dict) for request in batch_inferer.requests]
    )


def test_prepare_requests_bad_batch_size(
    batch_inferer: BatchInferer, sample_inputs: dict[str, ModelInput]
):
    """Test that an error is raised for batch size < 100."""
    small_inputs = dict(list(sample_inputs.items())[:50])
    with pytest.raises(
        ValueError, match=f"Minimum Batch Size is 100, {len(small_inputs)} given"
    ):
        batch_inferer.prepare_requests(small_inputs)
