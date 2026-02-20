from unittest.mock import AsyncMock, MagicMock, call, patch

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
    mock_boto3_session.return_value.client("s3").head_bucket.assert_called_once_with(Bucket=inputs["bucket_name"])

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


def test_prepare_requests(batch_inferer: BatchInferer, sample_inputs: dict[str, ModelInput]):
    """Test that requests are prepared."""
    batch_inferer.prepare_requests(sample_inputs)

    assert len(batch_inferer.requests) == len(sample_inputs)
    assert list(batch_inferer.requests[0].keys()) == ["recordId", "modelInput"]
    assert batch_inferer.requests[0]["recordId"] == "000"
    assert "anthropic_version" in batch_inferer.requests[0]["modelInput"]
    assert all([isinstance(request["modelInput"], dict) for request in batch_inferer.requests])


def test_prepare_requests_bad_batch_size(batch_inferer: BatchInferer, sample_inputs: dict[str, ModelInput]):
    """Test that an error is raised for batch size < 100."""
    small_inputs = dict(list(sample_inputs.items())[:50])
    with pytest.raises(ValueError, match=f"Minimum Batch Size is 100, {len(small_inputs)} given"):
        batch_inferer.prepare_requests(small_inputs)


def test_load_results_with_unknown_manifest_fields(
    batch_inferer: BatchInferer,
):
    """Test that load_results succeeds when manifest contains unknown fields.

    AWS Bedrock may add new fields (e.g. inputAudioSecond) to the manifest.
    These should be silently ignored rather than causing a TypeError.
    """
    mock_results = [{"recordId": "001", "modelOutput": {"content": "test"}}]
    mock_manifest = [
        {
            "totalRecordCount": 100,
            "processedRecordCount": 95,
            "successRecordCount": 90,
            "errorRecordCount": 5,
            "inputTokenCount": 50_000,
            "outputTokenCount": 100_000,
            "inputAudioSecond": 42,
            "someUnexpectedField": "value",
        }
    ]

    batch_inferer.output_file_name = "test-job_out.jsonl"
    batch_inferer.manifest_file_name = "test-job_manifest.jsonl"

    def mock_read_jsonl(file_path):
        if "manifest" in file_path:
            return mock_manifest
        return mock_results

    with (
        patch.object(BatchInferer, "_read_jsonl", side_effect=mock_read_jsonl),
        patch("os.path.isfile", return_value=True),
    ):
        batch_inferer.load_results()

    assert batch_inferer.manifest is not None
    assert batch_inferer.manifest.totalRecordCount == 100
    assert batch_inferer.manifest.successRecordCount == 90
    assert batch_inferer.manifest.inputTokenCount == 50_000
    assert batch_inferer.manifest.outputTokenCount == 100_000
    assert batch_inferer.manifest.model_extra == {
        "inputAudioSecond": 42,
        "someUnexpectedField": "value",
    }
    assert batch_inferer.results == mock_results


def test_init_default_output_dir(batch_inferer: BatchInferer):
    """Test that output_dir defaults to current directory."""
    assert batch_inferer.output_dir == "."


def test_init_custom_output_dir(mock_boto3_session: MagicMock):
    """Test that a custom output_dir is stored and created."""
    with patch("os.makedirs") as mock_makedirs:
        bi = BatchInferer(
            model_name="test-supported-claude-model",
            bucket_name="test-bucket",
            region="test-region",
            job_name="test-job",
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            output_dir="my_outputs",
        )
        assert bi.output_dir == "my_outputs"
        mock_makedirs.assert_called_with("my_outputs", exist_ok=True)


def test_local_path_default(batch_inferer: BatchInferer):
    """Test _local_path joins output_dir with bare filename."""
    assert batch_inferer._local_path("foo.jsonl") == "./foo.jsonl"


def test_local_path_custom(mock_boto3_session: MagicMock):
    """Test _local_path with a custom output_dir."""
    bi = BatchInferer(
        model_name="test-supported-claude-model",
        bucket_name="test-bucket",
        region="test-region",
        job_name="test-job",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        output_dir="my_dir",
    )
    assert bi._local_path("foo.jsonl") == "my_dir/foo.jsonl"


def test_write_requests_locally_uses_output_dir(mock_boto3_session: MagicMock):
    """Test that _write_requests_locally writes to output_dir."""
    bi = BatchInferer(
        model_name="test-supported-claude-model",
        bucket_name="test-bucket",
        region="test-region",
        job_name="test-job",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        output_dir="my_outputs",
    )
    bi.requests = [{"recordId": "001", "modelInput": {"messages": []}}]

    mock_open = MagicMock()
    with patch("builtins.open", mock_open):
        bi._write_requests_locally()

    mock_open.assert_called_once_with("my_outputs/test-job.jsonl", "w")


def test_push_requests_uses_output_dir_for_local_but_not_s3(
    mock_boto3_session: MagicMock,
    mock_s3_client: MagicMock,
):
    """Test that push_requests_to_s3 uses output_dir for local file but bare name for S3 key."""
    bi = BatchInferer(
        model_name="test-supported-claude-model",
        bucket_name="test-bucket",
        region="test-region",
        job_name="test-job",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        output_dir="my_outputs",
    )
    bi.requests = [{"recordId": "001", "modelInput": {"messages": []}}]

    with patch("builtins.open", MagicMock()):
        bi.push_requests_to_s3()

    mock_s3_client.upload_file.assert_called_once_with(
        Filename="my_outputs/test-job.jsonl",
        Bucket="test-bucket",
        Key="input/test-job.jsonl",
        ExtraArgs={"ContentType": "application/json"},
    )


def test_download_results_uses_output_dir(
    mock_boto3_session: MagicMock,
    mock_s3_client: MagicMock,
    mock_bedrock_client: MagicMock,
):
    """Test that download_results writes to output_dir but uses bare S3 keys."""
    bi = BatchInferer(
        model_name="test-supported-claude-model",
        bucket_name="test-bucket",
        region="test-region",
        job_name="test-job",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        output_dir="my_outputs",
    )
    bi.job_arn = "arn:aws:bedrock:region:account:job/test-id"
    bi.job_status = "Completed"

    bi.download_results()

    mock_s3_client.download_file.assert_any_call(
        Bucket="test-bucket",
        Key="output/test-id/test-job.jsonl.out",
        Filename="my_outputs/test-job_out.jsonl",
    )
    mock_s3_client.download_file.assert_any_call(
        Bucket="test-bucket",
        Key="output/test-id/manifest.json.out",
        Filename="my_outputs/test-job_manifest.jsonl",
    )


def test_load_results_uses_output_dir(batch_inferer: BatchInferer):
    """Test that load_results reads from output_dir paths."""
    batch_inferer.output_file_name = "test-job_out.jsonl"
    batch_inferer.manifest_file_name = "test-job_manifest.jsonl"

    mock_results = [{"recordId": "001", "modelOutput": {"content": "test"}}]
    mock_manifest = [
        {
            "totalRecordCount": 10,
            "processedRecordCount": 10,
            "successRecordCount": 10,
            "errorRecordCount": 0,
        }
    ]

    def mock_read_jsonl(file_path):
        if "manifest" in file_path:
            return mock_manifest
        return mock_results

    def mock_isfile(path):
        """Verify that isfile is called with output_dir-prefixed paths."""
        return path in ("./test-job_out.jsonl", "./test-job_manifest.jsonl")

    with (
        patch.object(BatchInferer, "_read_jsonl", side_effect=mock_read_jsonl),
        patch("os.path.isfile", side_effect=mock_isfile),
    ):
        batch_inferer.load_results()

    assert batch_inferer.results == mock_results
    assert batch_inferer.manifest is not None
