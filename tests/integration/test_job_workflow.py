from unittest.mock import MagicMock, patch

import pytest

from llmbo import BatchInferer, ModelInput


def test_create_job(batch_inferer: BatchInferer, sample_inputs: dict[str, ModelInput]):
    """Test job creation with mocked AWS clients."""
    batch_inferer.prepare_requests(sample_inputs)
    response = batch_inferer.create()

    # Assert the job was created
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert batch_inferer.job_arn is not None

    # Check the correct function was called
    batch_inferer.client.create_model_invocation_job.assert_called_once()


def test_create_fail_no_requests(batch_inferer: BatchInferer):
    """Test failure with no set requests."""
    with pytest.raises(AttributeError):
        batch_inferer.create()


def test_create_fail_http_error(
    batch_inferer: BatchInferer,
    sample_inputs: dict[str, ModelInput],
):
    """Test failure when HTTP error is returned."""
    batch_inferer.client.create_model_invocation_job.return_value = {"ResponseMetadata": {"HTTPStatusCode": 400}}

    batch_inferer.prepare_requests(sample_inputs)

    with pytest.raises(
        RuntimeError,
        match=r"There was an error creating the job .*, non 200 response from bedrock",
    ):
        batch_inferer.create()


def test_create_fail_no_response(batch_inferer: BatchInferer, sample_inputs: dict[str, ModelInput]):
    """Test failure when no response is returned."""
    batch_inferer.client.create_model_invocation_job.return_value = None

    batch_inferer.prepare_requests(sample_inputs)

    with pytest.raises(
        RuntimeError,
        match="There was an error creating the job, no response from bedrock",
    ):
        batch_inferer.create()


def test_complete_workflow(batch_inferer: BatchInferer, sample_inputs: dict[str, ModelInput]):
    """Test the complete workflow from preparation to downloading results."""
    # Setup for success
    batch_inferer.prepare_requests(sample_inputs)
    batch_inferer.create()

    # Mock the check_complete method to avoid polling
    batch_inferer.check_complete = MagicMock(return_value="Completed")

    # Create fake output and manifest files for testing
    with (
        patch("os.path.isfile", return_value=True),
        patch.object(batch_inferer, "_read_jsonl") as mock_read,
    ):
        mock_read.side_effect = [
            [{"recordId": "001", "modelOutput": {"content": "test"}}],  # Results
            [
                {
                    "totalRecordCount": 100,
                    "processedRecordCount": 100,
                    "successRecordCount": 100,
                    "errorRecordCount": 0,
                }
            ],  # Manifest
        ]

        # Test the workflow
        batch_inferer.download_results()
        batch_inferer.load_results()

        # Verify results were loaded
        assert batch_inferer.results is not None
        assert batch_inferer.manifest is not None
