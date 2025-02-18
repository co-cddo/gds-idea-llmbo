import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Type
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class Manifest:
    totalRecordCount: int
    processedRecordCount: int
    successRecordCount: int
    errorRecordCount: int
    inputTokenCount: Optional[int]
    outputTokenCount: Optional[int]


@dataclass
class ToolChoice:
    type: Literal["any", "tool", "auto"]
    name: Optional[str] = None


@dataclass
class ModelInput:
    """Configuration class for AWS Bedrock model inputs.

    This class defines the structure and parameters for model invocation requests
    following AWS Bedrock's expected format.

    See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

    Attributes:
        messages (List[dict]): List of message objects with role and content
        anthropic_version (str): Version string for Anthropic models
        max_tokens (int): Maximum number of tokens in the response
        system (Optional[str]): System message for the model
        stop_sequences (Optional[List[str]]): Custom stop sequences
        temperature (Optional[float]): Sampling temperature
        top_p (Optional[float]): Nucleus sampling parameter
        top_k (Optional[int]): Top-k sampling parameter
        tools (Optional[List[dict]]): Tool definitions for structured outputs
        tool_choice (Optional[ToolChoice]): Tool selection configuration
    """

    # These are required
    messages: List[dict]
    anthropic_version: str = "bedrock-2023-05-31"
    max_tokens: int = 2000

    system: Optional[str] = None
    stop_sequences: Optional[List[str]] | None = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    tools: Optional[List[dict]] | None = None
    tool_choice: Optional[ToolChoice] = None

    def to_dict(self):
        result = {k: v for k, v in self.__dict__.items() if v is not None}
        if self.tool_choice:
            result["tool_choice"] = self.tool_choice.__dict__
        return result

    def to_json(self):
        return json.dumps(self.to_dict())


VALID_FINISHED_STATUSES = ["Completed", "Failed", "Stopped", "Expired"]


class BatchInferer:
    """A class to manage batch inference jobs using AWS Bedrock.

    This class handles the creation, monitoring, and retrieval of batch inference jobs
    for large-scale model invocations using AWS Bedrock service.

    Args:
        model_name (str): The name/ID of the AWS Bedrock model to use
        bucket_name (str): The S3 bucket name for storing input/output data
        region (str): The region to run the batch inference job in.
        job_name (str): A unique name for the batch inference job
        role_arn (str): The AWS IAM role ARN with necessary permissions
        time_out_duration_hours (int, optional): Maximum job runtime in hours. Defaults to 24.
        session (boto3.session, optional): A boto3 session to be used for calls to AWS, If one if not provided a new one will be  created

    Attributes:
        job_arn (str): The ARN of the created batch inference job
        results (List[dict]): The results of the batch inference job. Available after job completion.
        manifest (Manifest): Job execution statistics. Available after job completion.
        job_status (str): Current status of the batch job. One of VALID_FINISHED_STATUSES.
    """

    logger = logging.getLogger(f"{__name__}.BatchInferer")

    def __init__(
        self,
        model_name: str,  # this should be an enum...
        bucket_name: str,
        region: str,
        job_name: str,
        role_arn: str,
        time_out_duration_hours: int = 24,
        session: boto3.Session | None = None,
    ):
        """Initialize a BatchInferer for AWS Bedrock batch processing.

        Creates a configured batch inference manager that handles the end-to-end process
        of submitting and managing batch jobs on AWS Bedrock.

        Args:
            model_name (str): The AWS Bedrock model identifier (e.g., 'anthropic.claude-3-haiku-20240307-v1:0')
            bucket_name (str): Name of the S3 bucket for storing job inputs and outputs
            region (str): The region containing the llm to call, must match the bucket region
            job_name (str): Unique identifier for this batch job. Used in file naming.
            role_arn (str): AWS IAM role ARN with permissions for Bedrock and S3 access
            time_out_duration_hours (int, optional): Maximum runtime for the batch job. Defaults to 24 hours.
            session (boto3.session, optional): A boto3 session to be used for calls to AWS, If one if not provided a new one will be  created

        Raises:
            KeyError: If AWS_PROFILE environment variable is not set
            ValueError: If the provided role_arn doesn't exist or is invalid

        Example:
        ```python
            >>> bi = BatchInferer(
                    model_name="anthropic.claude-3-haiku-20240307-v1:0",
                    bucket_name="my-inference-bucket",
                    job_name="batch-job-2024-01-01",
                    role_arn="arn:aws:iam::123456789012:role/BedrockBatchRole"
                )
        ```

        Note:
            - Requires valid AWS credentials and configuration
            - The S3 bucket must exist and be accessible via the provided role
            - Job name will be used to create unique file names for inputs and outputs
        """

        self.logger.info("Intialising BatchInferer")
        # model parameters
        self.model_name = model_name
        self.time_out_duration_hours = time_out_duration_hours

        self.session: boto3.Session = session or boto3.Session()

        # file/bucket parameters
        self._check_bucket(bucket_name, region)
        self.bucket_name = bucket_name
        self.bucket_uri = "s3://" + bucket_name
        self.job_name = job_name or "batch_inference_" + str(uuid4())[:6]
        self.file_name = job_name + ".jsonl"
        self.output_file_name = None
        self.manifest_file_name = None

        self.check_for_profile()
        self._check_arn(role_arn)
        self.role_arn = role_arn
        self.region = region

        self.client: boto3.client = self.session.client("bedrock", region_name=region)

        # internal state - created by the class later.
        self.job_arn = None
        self.job_status = None
        self.results = None
        self.manifest = None
        self.requests = None

        self.logger.info("Initialized BatchInferer")

    @property
    def unique_id_from_arn(self):
        if not self.job_arn:
            self.logger.error("Job ARN not set")
            raise ValueError("Job ARN not set")
        return self.job_arn.split("/")[-1]

    def check_for_profile(self):
        if not os.getenv("AWS_PROFILE"):
            self.logger.error("AWS_PROFILE environment variable not set")
            raise KeyError("AWS_PROFILE environment variable not set")

    @staticmethod
    def _read_jsonl(file_path):
        data = []
        with open(file_path, "r") as file:
            for line in file:
                data.append(json.loads(line.strip()))
        return data

    def _get_bucket_location(self, bucket_name: str) -> str | None:
        """
        get the location of the s3 bucket

        Args:
            bucket_name (str): the name of a bucket

        Raises:
            ValueError: If the bucket is not accessible

        Returns:
            str: a region, e.g. "eu-west-2"
        """
        try:
            s3_client = self.session.client("s3")
            response = s3_client.get_bucket_location(Bucket=bucket_name)

            if response:
                region = response["LocationConstraint"]
                # aws returns None if the region is us-east-1 otherwise it returns the region
                return region if region else "us-east-1"
        except ClientError as e:
            self.logger.error(f"Bucket {bucket_name} is not accessible: {e}")
            raise ValueError(f"Bucket {bucket_name} is not accessible")

    def _check_bucket(self, bucket_name: str, region: str) -> None:
        """
        Validate if the bucket_name provided exists

        Args:
            bucket_name (str): the name of a bucket
            region (str): the name of a region

        Raises:
            ValueError: If the bucket is not accessible
            ValueError: If the bucket is not in the same region as the LLM.
        """
        try:
            s3_client = self.session.client("s3")
            s3_client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            self.logger.error(f"Bucket {bucket_name} is not accessible: {e}")
            raise ValueError(f"Bucket {bucket_name} is not accessible")

        if (bucket_region := self._get_bucket_location(bucket_name)) != region:
            self.logger.error(
                f"Bucket {bucket_name} is not located in the same region [{region}] as the llm [{bucket_region}]"
            )
            raise ValueError(
                f"Bucket {bucket_name} is not located in the same region [{region}] as the llm [{bucket_region}]"
            )

    def _check_arn(self, role_arn: str) -> bool:
        """Validate if an IAM role exists and is accessible.

        Attempts to retrieve the IAM role using the provided ARN to verify its
        existence and accessibility.

        Args:
            role_arn (str): The AWS ARN of the IAM role to check.
                Format: 'arn:aws:iam::<account-id>:role/<role-name>'

        Returns:
            bool: True if the role exists and is accessible.

        Raises:
            ValueError: If the role does not exist.
        ClientError: If there are AWS API issues unrelated to role existence."""

        if not role_arn.startswith("arn:aws:iam::"):
            self.logger.error("Invalid ARN format")
            raise ValueError("Invalid ARN format")

        # Extract the role name from the ARN
        role_name = role_arn.split("/")[-1]

        iam_client = self.session.client("iam")

        try:
            # Try to get the role
            iam_client.get_role(RoleName=role_name)
            self.logger.info(f"Role '{role_name}' exists.")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                self.logger.error(f"Role '{role_name}' does not exist.")
                raise ValueError(f"Role '{role_name}' does not exist.")
            else:
                raise e

    def prepare_requests(self, inputs: Dict[str, ModelInput]) -> None:
        """Prepare batch inference requests from a dictionary of model inputs.

        Formats model inputs into the required JSONL structure for AWS Bedrock batch processing.
        Each request is formatted as:
            {
                "recordId": str,
                "modelInput": dict
            }

        Args:
            inputs (Dict[str, ModelInput]): Dictionary mapping record IDs to their corresponding
                ModelInput configurations. The record IDs will be used to track results.

        Raises:
            ValueError: If len(inputs) < 100, as AWS Bedrock requires minimum batch size of 100

        Example:
            >>> inputs = {
            ...     "001": ModelInput(
            ...         messages=[{"role": "user", "content": "Hello"}],
            ...         temperature=0.7
            ...     ),
            ...     "002": ModelInput(
            ...         messages=[{"role": "user", "content": "Hi"}],
            ...         temperature=0.7
            ...     )
            ... }
            >>> bi.prepare_requests(inputs)

        Note:
            - This method must be called before push_requests_to_s3()
            - The prepared requests are stored in self.requests
            - Each ModelInput is converted to a dict using its to_dict() method
        """
        # maybe a data class conforming to this???
        #

        self.logger.info(f"Preparing {len(inputs)} requests")
        if len(inputs) < 100:
            self.logger.error(f"Minimum Batch Size is 100, {len(inputs)} given.")
            raise ValueError(f"Minimum Batch Size is 100, {len(inputs)} given.")

        self.requests = [
            {
                "recordId": id,
                "modelInput": model_input.to_dict(),
            }
            for id, model_input in inputs.items()
        ]

    def _write_requests_locally(self) -> None:
        """Write batch inference requests to a local JSONL file.

        Creates or overwrites a local JSONL file containing the prepared inference
        requests. Each line contains a JSON object with recordId and modelInput.

        Raises:
            IOError: If unable to write to the file
            AttributeError: If called before prepare_requests()

        Note:
            - File is named according to self.file_name
            - Internal method used by push_requests_to_s3()
            - Will overwrite existing files with the same name
        """
        self.logger.info(f"Writing {len(self.requests)} requests to {self.file_name}")
        with open(self.file_name, "w") as file:
            for record in self.requests:
                file.write(json.dumps(record) + "\n")

    def push_requests_to_s3(self) -> Dict[str, Any]:
        """Upload batch inference requests to S3.

        Writes the prepared requests to a local JSONL file and uploads it to the
        configured S3 bucket in the 'input/' prefix.

        Returns:
            dict: The S3 upload response from boto3

        Raises:
            IOError: If local file operations fail
            ClientError: If S3 upload fails
            AttributeError: If called before prepare_requests()

        Note:
            - Creates/overwrites files both locally and in S3
            - S3 path: {bucket_name}/input/{job_name}.jsonl
            - Sets Content-Type to 'application/json'
        """
        # do I want to write this file locally? - maybe stream it or write it to
        # temp file instead
        self._write_requests_locally()
        s3_client = self.session.client("s3")
        self.logger.info(f"Pushing {len(self.requests)} requests to {self.bucket_name}")
        response = s3_client.upload_file(
            Filename=self.file_name,
            Bucket=self.bucket_name,
            Key=f"input/{self.file_name}",
            ExtraArgs={"ContentType": "application/json"},
        )
        return response

    def create(self) -> Dict[str, Any]:
        """Create a new batch inference job in AWS Bedrock.

        Initializes a new model invocation job using the configured parameters
        and uploaded input data.

        Returns:
            dict: The complete response from the create_model_invocation_job API call

        Raises:
            RuntimeError: If job creation fails
            ClientError: For AWS API errors
            ValueError: If required configurations are missing

        Note:
            - Sets self.job_arn on successful creation
            - Input data must be uploaded to S3 before calling this method
            - Job will timeout after self.time_out_duration_hours
        """
        if self.requests:
            self.logger.info(f"Creating job {self.job_name}")
            response = self.client.create_model_invocation_job(
                jobName=self.job_name,
                roleArn=self.role_arn,
                clientRequestToken="string",
                modelId=self.model_name,
                inputDataConfig={
                    "s3InputDataConfig": {
                        "s3InputFormat": "JSONL",
                        "s3Uri": f"{self.bucket_uri}/input/{self.file_name}",
                    }
                },
                outputDataConfig={
                    "s3OutputDataConfig": {
                        "s3Uri": f"{self.bucket_uri}/output/",
                    }
                },
                timeoutDurationInHours=self.time_out_duration_hours,
                tags=[{"key": "bedrock_batch_inference", "value": self.job_name}],
            )

            if response:
                response_status = response["ResponseMetadata"]["HTTPStatusCode"]
                if response_status == 200:
                    self.logger.info(f"Job {self.job_name} created successfully")
                    self.logger.info(f"Assigned jobArn: {response['jobArn']}")
                    self.job_arn = response["jobArn"]
                    return response
                else:
                    self.logger.error(
                        f"There was an error creating the job {self.job_name}, non 200 response from bedrock"
                    )
                    raise RuntimeError(
                        f"There was an error creating the job {self.job_name}, non 200 response from bedrock"
                    )
            else:
                self.logger.error(
                    "There was an error creating the job, no response from bedrock"
                )
                raise RuntimeError(
                    "There was an error creating the job, no response from bedrock"
                )
        else:
            self.logger.error("There were no prepared requests")
            raise AttributeError("There were no prepared requests")

    def download_results(self) -> None:
        """Download batch inference results from S3.

        Retrieves both the results and manifest files from S3 once the job
        has completed. Files are downloaded to:
            - {job_name}_out.jsonl: Contains model outputs
            - {job_name}_manifest.jsonl: Contains job statistics

        Raises:
            ClientError: For S3 download failures
            ValueError: If job hasn't completed or job_arn isn't set

        Note:
            - Only downloads if job status is in VALID_FINISHED_STATUSES
            - Files are downloaded to current working directory
            - Existing files will be overwritten
            - Call check_complete() first to ensure job is finished
        """
        if self.check_complete() in VALID_FINISHED_STATUSES:
            file_name_, ext = os.path.splitext(self.file_name)
            self.output_file_name = f"{file_name_}_out{ext}"
            self.manifest_file_name = f"{file_name_}_manifest{ext}"
            self.logger.info(
                f"Job:{self.job_arn} Complete. Downloading results from {self.bucket_name}"
            )
            s3_client = self.session.client("s3")
            s3_client.download_file(
                Bucket=self.bucket_name,
                Key=f"output/{self.unique_id_from_arn}/{self.file_name}.out",
                Filename=self.output_file_name,
            )
            self.logger.info(f"Downloaded results file to {self.output_file_name}")

            s3_client.download_file(
                Bucket=self.bucket_name,
                Key=f"output/{self.unique_id_from_arn}/manifest.json.out",
                Filename=self.manifest_file_name,
            )
            self.logger.info(f"Downloaded manifest file to {self.manifest_file_name}")
        else:
            self.logger.info(
                f"Job:{self.job_arn} was not marked one of {VALID_FINISHED_STATUSES}, could not download."
            )

    def load_results(self) -> None:
        """Load batch inference results and manifest from local files.

        Reads and parses the output files downloaded from S3, populating:
            - self.results: List of inference results from the output JSONL file
            - self.manifest: Statistics about the job execution (total records, success/error counts, etc.)

        The method expects two files to exist locally:
            - {job_name}_out.jsonl: Contains the model outputs
            - {job_name}_manifest.jsonl: Contains execution statistics

        Raises:
            FileExistsError: If either the results or manifest files are not found locally

        Note:
            - Must call download_results() before calling this method
            - The manifest provides useful metrics like success rate and token counts
        """
        if os.path.isfile(self.output_file_name) and os.path.isfile(
            self.manifest_file_name
        ):
            self.results = self._read_jsonl(self.output_file_name)
            self.manifest = Manifest(**self._read_jsonl(self.manifest_file_name)[0])
        else:
            self.logger.error(
                "Result files do not exist, you may need to call .download_results() first."
            )
            raise FileExistsError(
                "Result files do not exist, you may need to call .download_results() first."
            )

    def cancel_batch(self) -> None:
        """Cancel a running batch inference job.

        Attempts to stop the currently running batch inference job identified by self.job_arn.

        Returns:
            None

        Raises:
            RuntimeError: If the job cancellation request fails
            ValueError: If no job_arn is set (i.e., no job has been created)
        """

        if not self.job_arn:
            self.logger.error("No job_arn set - no job to cancel")
            raise ValueError("No job_arn set - no job to cancel")

        response = self.client.stop_model_invocation_job(jobIdentifier=self.job_arn)

        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            self.logger.info(
                f"Job {self.job_name} with id={self.job_arn} was cancelled"
            )
            self.job_status = "Stopped"
        else:
            self.logger.error(
                f"Failed to cancel job {self.job_name}. Status: {response['ResponseMetadata']['HTTPStatusCode']}"
            )
            raise RuntimeError(f"Failed to cancel job {self.job_name}")

    def check_complete(self) -> Optional[str]:
        """Check if the batch inference job has completed.

        Returns:
            Optional[str]: The job status if completed (one of VALID_FINISHED_STATUSES), None otherwise
        """
        if self.job_status not in VALID_FINISHED_STATUSES:
            self.logger.info(f"Checking status of job {self.job_arn}")
            response = self.client.get_model_invocation_job(jobIdentifier=self.job_arn)

            self.job_status = response["status"]
            self.logger.info(f"Job status is {self.job_status}")

            if self.job_status in VALID_FINISHED_STATUSES:
                return self.job_status
            return None
        else:
            self.logger.info(f"Job {self.job_arn} is already {self.job_status}")
            return self.job_status

    def poll_progress(self, poll_interval_seconds: int = 60) -> bool:
        """Polls the progress of a job

        Args:
            poll_interval_seconds (int, optional): Number of seconds between checks. Defaults to 60.

        Returns:
            bool: True if job is complete.
        """
        self.logger.info(f"Polling for progress every {poll_interval_seconds} seconds")
        while not self.check_complete():
            time.sleep(poll_interval_seconds)
        return True

    def auto(self, inputs: Dict[str, ModelInput]) -> dict[str, ModelInput]:
        """Execute the complete batch inference workflow automatically.

        This method combines the preparation, execution, monitoring, and result retrieval
        steps into a single operation.

        Args:
            inputs (Dict[str, ModelInput]): Dictionary of record IDs mapped to their ModelInput configurations

        Returns:
            List[dict]: The results of the batch inference job
        """
        self.prepare_requests(inputs)
        self.push_requests_to_s3()
        self.create()
        self.poll_progress(10 * 60)
        self.download_results()
        self.load_results()
        return self.results

    @classmethod
    def recover_details_from_job_arn(
        cls, job_arn: str, region: str, session: boto3.Session | None = None
    ) -> "BatchInferer":
        """Recover a BatchInferer instance from an existing job ARN.

        Used to reconstruct a BatchInferer object when the original Python process
        has terminated but the AWS job is still running or complete.

        Args:
            job_arn: (str) The AWS ARN of the existing batch inference job
            region: (str) the region where the job was scheduled
            session (boto3.session, optional): A boto3 session to be used for calls to AWS,
                    If one if not provided a new one will be  created

        Returns:
            BatchInferer: A configured instance with the job's details

        Raises:
            ValueError: If the job cannot be found or response is invalid

        Example:
            >>> job_arn = "arn:aws:bedrock:region:account:job/xyz123"
            >>> bi = BatchInferer.recover_details_from_job_arn(job_arn)
            >>> bi.check_complete()
            'Completed'
        """

        cls.logger.info(f"Attempting to Recover BatchInferer from {job_arn}")
        session = session or boto3.Session()
        response = cls.check_for_existing_job(job_arn, region, session)

        try:
            # Extract required parameters from response
            job_name = response["jobName"]
            model_id = response["modelId"]
            bucket_name = response["inputDataConfig"]["s3InputDataConfig"][
                "s3Uri"
            ].split("/")[2]
            role_arn = response["roleArn"]

            # Validate required files exist
            input_file = f"{job_name}.jsonl"
            if not os.path.exists(input_file):
                cls.logger.error(f"Required input file not found: {input_file}")
                raise FileNotFoundError(f"Required input file not found: {input_file}")

            requests = cls._read_jsonl(input_file)

            bi = cls(
                model_name=model_id,
                job_name=job_name,
                region=region,
                bucket_name=bucket_name,
                role_arn=role_arn,
                session=session,
            )
            bi.job_arn = job_arn
            bi.requests = requests
            bi.job_status = response["status"]

            return bi

        except (KeyError, IndexError) as e:
            cls.logger.error(f"Invalid job response format: {str(e)}")
            raise ValueError(f"Invalid job response format: {str(e)}") from e
        except Exception as e:
            cls.logger.error(f"Failed to recover job details: {str(e)}")
            raise RuntimeError(f"Failed to recover job details: {str(e)}") from e

    @classmethod
    def check_for_existing_job(
        cls, job_arn, region, session: boto3.Session | None = None
    ) -> Dict[str, Any]:
        """Check if a job exists and return its details.

        Args:
            job_arn (str): The AWS ARN of the job to check
            region (str): The AWS region where the job was created
            session (boto3.Session, optional): A boto3 session to be used for AWS API calls.
                                           If not provided, a new session will be created.

        Returns:
            Dict[str, Any]: The job details from AWS Bedrock

        Raises:
            ValueError: If the job ARN is invalid or the job is not found
            RuntimeError: For other AWS API errors
        """
        if not job_arn.startswith("arn:aws:bedrock:"):
            cls.logger.error(f"Invalid Bedrock ARN format: {job_arn}")
            raise ValueError(f"Invalid Bedrock ARN format: {job_arn}")
        session = session or boto3.Session()
        client = session.client("bedrock", region_name=region)

        try:
            response = client.get_model_invocation_job(jobIdentifier=job_arn)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                cls.logger.error(f"Job not found: {job_arn}")
                raise ValueError(f"Job not found: {job_arn}") from e
            cls.logger.error(f"AWS API error: {str(e)}")
            raise RuntimeError(f"AWS API error: {str(e)}") from e

        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            cls.logger.error(
                f"Unexpected response status: {response['ResponseMetadata']['HTTPStatusCode']}"
            )
            raise RuntimeError(
                f"Unexpected response status: {response['ResponseMetadata']['HTTPStatusCode']}"
            )

        return response


class StructuredBatchInferer(BatchInferer):
    """A specialized BatchInferer that enforces structured outputs using Pydantic models.

    Inspired by the instructor package, see: https://python.useinstructor.com/
    This class extends BatchInferer to add schema validation and structured output
    handling using Pydantic models.

    Args:
        output_model (BaseModel): A Pydantic model defining the expected output structure
        model_name (str): The name/ID of the AWS Bedrock model to use
        bucket_name (str): The S3 bucket name for storing input/output data
        region (str): The region to run the batch inference job in.
        job_name (str): A unique name for the batch inference job
        role_arn (str): The AWS IAM role ARN with necessary permissions
        time_out_duration_hours (int, optional): Maximum job runtime in hours. Defaults to 24.
        session (boto3.Session, optional): A boto3 session to be used for AWS API calls.
                                           If not provided, a new session will be created.


    """

    logger = logging.getLogger(f"{__name__}.StructuredBatchInferer")

    def __init__(
        self,
        output_model: Type[BaseModel],
        model_name: str,  # this should be an enum...
        bucket_name: str,
        region: str,
        job_name: str,
        role_arn: str,
        time_out_duration_hours: int = 24,
        session: boto3.Session | None = None,
    ):
        """Initialize a StructuredBatchInferer for schema-validated batch processing.

        Creates a batch inference manager that enforces structured outputs using
        a Pydantic model schema. Automatically configures the model to use tools
        for enforcing the output structure.

        Args:
            output_model (BaseModel): Pydantic model class defining the expected output structure
            model_name (str): The AWS Bedrock model identifier
            bucket_name (str): Name of the S3 bucket for storing job inputs and outputs
            region (str): Region of the LLM must match the bucket
            job_name (str): Unique identifier for this batch job
            role_arn (str): AWS IAM role ARN with permissions for Bedrock and S3 access
            session (boto3.Session, optional): A boto3 session to be used for AWS API calls.
                                           If not provided, a new session will be created.

        Raises:
            KeyError: If AWS_PROFILE environment variable is not set
            ValueError: If the provided role_arn doesn't exist or is invalid

        Example:
            >>> class PersonInfo(BaseModel):
            ...     name: str
            ...     age: int
            ...
            >>> sbi = StructuredBatchInferer(
            ...     output_model=PersonInfo,
            ...     model_name="anthropic.claude-3-haiku-20240307-v1:0",
            ...     bucket_name="my-inference-bucket",
            ...     job_name="structured-batch-2024",
            ...     role_arn="arn:aws:iam::123456789012:role/BedrockBatchRole"
            ... )

        Note:
            - Converts the Pydantic model into a tool definition for the LLM
            - All results will be validated against the provided schema
            - Failed schema validations will raise errors during result processing
            - Inherits all base BatchInferer functionality
        """
        self.output_model = output_model
        self.tool = self._build_tool()
        self.logger.info(
            f"Initialized StructuredBatchInferer with {output_model.__name__} schema"
        )

        super().__init__(
            model_name=model_name,
            bucket_name=bucket_name,
            region=region,
            job_name=job_name,
            role_arn=role_arn,
            time_out_duration_hours=time_out_duration_hours,
            session=session,
        )

    def _build_tool(self) -> dict:
        """Convert a Pydantic model into a tool definition for the model.

        Returns:
            dict: A tool description containing name, description, and input schema
        """
        return {
            "name": self.output_model.__name__,
            "description": self.output_model.__doc__ or "please fill in the schema",
            "input_schema": self.output_model.model_json_schema(),
        }

    def prepare_requests(self, inputs: Dict[str, ModelInput]):
        """Prepare structured batch inference requests with tool configurations.

        Extends the base preparation by adding tool definitions and tool choice
        parameters to each ModelInput. The tool definition is derived from the
        Pydantic output_model specified during initialization.

        Args:
            inputs (Dict[str, ModelInput]): Dictionary mapping record IDs to their corresponding
                ModelInput configurations. The record IDs will be used to track results.

        Raises:
            ValueError: If len(inputs) < 100, as AWS Bedrock requires minimum batch size of 100

        Example:
            >>> class PersonInfo(BaseModel):
            ...     name: str
            ...     age: int
            >>> sbi = StructuredBatchInferer(output_model=PersonInfo, ...)
            >>> inputs = {
            ...     "001": ModelInput(
            ...         messages=[{"role": "user", "content": "John is 25 years old"}],
            ...     )
            ... }
            >>> sbi.prepare_requests(inputs)

        Note:
            - Automatically adds the output_model schema as a tool definition
            - Sets tool_choice to force use of the defined schema
            - Original ModelInputs are modified to include tool configurations
        """
        self.logger.info(f"Adding tool {self.tool['name']} to model input")
        with_tools = {
            id: self._add_tool_to_model_input(model_input)
            for id, model_input in inputs.items()
        }
        super().prepare_requests(with_tools)

    def _add_tool_to_model_input(self, model_input: ModelInput) -> ModelInput:
        """Add tool definition and configuration to a ModelInput instance.

        Updates the ModelInput by:
            1. Adding the Pydantic model schema as a tool definition
            2. Setting tool_choice to force use of this specific tool

        Args:
            model_input (ModelInput): The original model input configuration

        Returns:
            ModelInput: The modified model input with tool configurations added
        """

        model_input.tools = [self.tool]
        model_input.tool_choice = ToolChoice(
            type="tool", name=self.output_model.__name__
        )
        return model_input

    def load_results(self):
        """Load and validate batch inference results against the output schema.

        Reads the output files downloaded from S3 and validates each result against
        the Pydantic output_model specified during initialization. Populates:
            - self.results: Raw inference results from the output JSONL file
            - self.manifest: Statistics about the job execution
            - self.instances: List of validated Pydantic model instances

        Raises:
            FileExistsError: If either the results or manifest files are not found locally
            ValueError: If any result fails schema validation or tool use validation

        Note:
            - Must call download_results() before calling this method
            - All results must conform to the specified output_model schema
            - Results must show successful tool use
        """
        super().load_results()
        self.instances = [
            {
                "recordId": result["recordId"],
                "outputModel": self.validate_result(result["modelOutput"]),
            }
            if result.get("modelOutput")
            else None
            for result in self.results
        ]

    def validate_result(
        self,
        result: dict,
    ) -> BaseModel | None:
        """Validate and parse a single model output against the schema.

        Checks that the model used the specified tool correctly and validates
        the output against the Pydantic model schema.

        Args:
            result (dict): The raw model output containing content and metadata

        Returns:
            BaseModel: An instance of the output_model containing the validated data
            or None if the return could not be validated.

        Example:
            >>> result = {"stop_reason": "tool_use",
            ...          "content": [{"type": "tool_use",
            ...                      "input": {"name": "John", "age": 30}}]}
            >>> instance = sbi.validate_result(result)
            >>> print(instance.name)
            'John'
        """
        if not result["stop_reason"] == "tool_use":
            self.logger.warning("Validation warning: Model did not use tool")
            return None
        if result.get("content", []) == []:
            self.logger.warning("Validation warning: No content found in response")
        if not len(result.get("content", [])) == 1:
            self.logger.warning(
                "Validation warning: Multiple instances of tool use per execution"
            )
            return None
        if result["content"][0]["type"] == "tool_use":
            try:
                output = self.output_model(**result["content"][0]["input"])
                return output
            except ValidationError as e:
                self.logger.warning(
                    f"Validation warning: Could not validate output {e}"
                )
                return None

    @classmethod
    def recover_details_from_job_arn(
        cls,
        job_arn: str,
        region: str,
        session: boto3.Session | None = None,
    ) -> "StructuredBatchInferer":
        raise TypeError(
            "Cannot recover structured job without output_model. Use recover_structured_job instead."
        )

    @classmethod
    def recover_structured_job(
        cls,
        job_arn: str,
        region: str,
        output_model: Type[BaseModel],
        session: boto3.Session | None = None,
    ) -> "StructuredBatchInferer":
        """Recover a StructuredBatchInferer instance from an existing job ARN.

        Used to reconstruct a StructuredBatchInferer object when the original Python process
        has terminated but the AWS job is still running or complete.

        Args:
            job_arn: (str) The AWS ARN of the existing batch inference job
            region: (str) the region where the job was scheduled
            output_model: (Type[BaseModel]) A pydantic model describing the required output
            session (boto3.Session, optional): A boto3 session to be used for AWS API calls.
                                           If not provided, a new session will be created.

        Returns:
            StructuredBatchInferer: A configured instance with the job's details

        Raises:
            ValueError: If the job cannot be found or response is invalid

        Example:
            >>> job_arn = "arn:aws:bedrock:region:account:job/xyz123"
            >>> region = us-east-1"
            >>> sbi = StructuredBatchInferer.recover_details_from_job_arn(job_arn, region, some_model)
            >>> sbi.check_complete()
            'Completed'
        """

        cls.logger.info(f"Attempting to Recover BatchInferer from {job_arn}")
        session = session or boto3.Session()
        response = cls.check_for_existing_job(job_arn, region, session)

        try:
            # Extract required parameters from response
            job_name = response["jobName"]
            model_id = response["modelId"]
            bucket_name = response["inputDataConfig"]["s3InputDataConfig"][
                "s3Uri"
            ].split("/")[2]
            role_arn = response["roleArn"]

            # Validate required files exist
            input_file = f"{job_name}.jsonl"
            if not os.path.exists(input_file):
                cls.logger.error(f"Required input file not found: {input_file}")
                raise FileNotFoundError(f"Required input file not found: {input_file}")

            requests = cls._read_jsonl(input_file)

            sbi = cls(
                model_name=model_id,
                output_model=output_model,
                job_name=job_name,
                region=region,
                bucket_name=bucket_name,
                role_arn=role_arn,
                session=session,
            )
            sbi.job_arn = job_arn
            sbi.requests = requests
            sbi.job_status = response["status"]

            return sbi

        except (KeyError, IndexError) as e:
            cls.logger.error(f"Invalid job response format: {str(e)}")
            raise ValueError(f"Invalid job response format: {str(e)}") from e
        except Exception as e:
            cls.logger.error(f"Failed to recover job details: {str(e)}")
            raise RuntimeError(f"Failed to recover job details: {str(e)}") from e


class NameAgeModel(BaseModel):
    name: str
    age: int


def batch_inference_example():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    load_dotenv()
    boto3.setup_default_session()

    # Prepare your modelInputs, I think this makes it a bit easier to ensure your model
    # inputs are correct
    inputs = {
        f"{i:03}": ModelInput(
            temperature=1,
            top_k=250,
            messages=[
                {"role": "user", "content": "Give me a random name, occupation and age"}
            ],
        )
        for i in range(0, 100, 1)
    }

    bi = BatchInferer(
        model_name="anthropic.claude-3-haiku-20240307-v1:0",
        job_name=f"my-first-inference-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        region="us-east-1",
        bucket_name="cddo-af-bedrock-batch-inference-us-east-1",
        role_arn="arn:aws:iam::992382722318:role/BatchInferenceRole",
    )

    bi.prepare_requests(inputs)
    bi.push_requests_to_s3()
    bi.create()
    bi.poll_progress(10 * 60)
    bi.download_results()
    bi.load_results()
    print("success")


def structured_batch_inference_example():
    class NameJobAge(BaseModel):
        """A class to store details about people and their jobs"""

        first_name: str
        last_name: str
        age: int
        occupation: str

    sbi = StructuredBatchInferer(
        model_name="anthropic.claude-3-haiku-20240307-v1:0",
        job_name=f"my-first-structured-inference-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        region="eu-west-2",
        bucket_name="cddo-af-bedrock-batch-inference",
        role_arn="arn:aws:iam::992382722318:role/BatchInferenceRole",
        output_model=NameJobAge,
    )

    names_and_that = [
        item["modelOutput"]["content"][0]["text"]
        for item in sbi._read_jsonl("my-first-inference-20250115-152412_out.jsonl")
    ]

    inputs = {
        f"{index:03}": ModelInput(
            temperature=0.1,
            messages=[{"role": "user", "content": item}],
        )
        for index, item in enumerate(names_and_that)
    }

    sbi.prepare_requests(inputs)
    sbi.push_requests_to_s3()
    sbi.create()
    print(sbi.job_arn)
    sbi.poll_progress(10 * 60)
    sbi.download_results()
    sbi.load_results()


# Example configuration (should be done in main application entry point)
def setup_logging(log_level: Optional[str] = "INFO"):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    setup_logging(log_level="INFO")  # or get from environment variable
    logger.info("Starting batch inference process")
    try:
        batch_inference_example()
        logger.info("Successfully completed batch inference")
    except Exception:
        logger.error("Batch inference failed", exc_info=True)
        raise


if __name__ == "__main__":
    main()
