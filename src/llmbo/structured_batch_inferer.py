import logging
import os

import boto3
from pydantic import BaseModel

from .batch_inferer import BatchInferer
from .models import ModelInput


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
        output_model: type[BaseModel],
        model_name: str,  # this should be an enum...
        bucket_name: str,
        region: str,
        job_name: str,
        role_arn: str,
        time_out_duration_hours: int = 24,
        session: boto3.Session | None = None,
        output_dir: str = ".",
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
            time_out_duration_hours (int): Number of hours before the job times out
            session (boto3.Session, optional): A boto3 session to be used for AWS API calls.
                If not provided, a new session will be created.
            output_dir (str, optional): Directory for local JSONL files. Defaults to ".".

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

        self.logger.info(f"Initialized StructuredBatchInferer with {output_model.__name__} schema")

        super().__init__(
            model_name=model_name,
            bucket_name=bucket_name,
            region=region,
            job_name=job_name,
            role_arn=role_arn,
            time_out_duration_hours=time_out_duration_hours,
            session=session,
            output_dir=output_dir,
        )

    def prepare_requests(self, inputs: dict[str, ModelInput]):
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
        self.logger.info(f"Adding tool {self.output_model.__name__} to model input")
        self._check_input_length(inputs)
        for id, model_input in inputs.items():
            inputs[id] = self.adapter.prepare_model_input(model_input, self.output_model)

        self.requests = self._to_requests(inputs)

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
                "outputModel": self.adapter.validate_result(result["modelOutput"], self.output_model),
            }
            if result.get("modelOutput")
            else None
            for result in self.results
        ]

    @classmethod
    def recover_details_from_job_arn(
        cls,
        job_arn: str,
        region: str,
        session: boto3.Session | None = None,
        output_dir: str = ".",
    ) -> "StructuredBatchInferer":
        """Placeholder method for interface consistency.

        This method exists to maintain compatibility with the parent class but
        is not implemented for structured jobs. Use `recover_structured_job`
        instead.

        Raises:
            NotImplementedError: Always raised when called.
        """
        raise NotImplementedError(
            "Cannot recover structured job without output_model. Use recover_structured_job instead."
        )

    @classmethod
    def recover_structured_job(
        cls,
        job_arn: str,
        region: str,
        output_model: type[BaseModel],
        session: boto3.Session | None = None,
        output_dir: str = ".",
    ) -> "StructuredBatchInferer":
        """Recover a StructuredBatchInferer instance from an existing job ARN.

        Used to reconstruct a StructuredBatchInferer object when the original Python
        process has terminated but the AWS job is still running or complete.

        Args:
            job_arn: (str) The AWS ARN of the existing batch inference job
            region: (str) the region where the job was scheduled
            output_model: (type[BaseModel]) A pydantic model describing the required output
            session (boto3.Session, optional): A boto3 session to be used for AWS API calls.
                                            If not provided, a new session will be created.
            output_dir (str, optional): Directory for local JSONL files. Defaults to ".".

        Returns:
            StructuredBatchInferer: A configured instance with the job's details

        Raises:
            ValueError: If the job cannot be found or response is invalid

        Example:
            >>> job_arn = "arn:aws:bedrock:region:account:job/xyz123"
            >>> region = us-east-1"
            >>> sbi = StructuredBatchInferer.recover_structured_job(job_arn, region, some_model)
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
            bucket_name = response["inputDataConfig"]["s3InputDataConfig"]["s3Uri"].split("/")[2]
            role_arn = response["roleArn"]

            # Validate required files exist
            input_file = os.path.join(output_dir, f"{job_name}.jsonl")
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
                output_dir=output_dir,
            )
            sbi.job_arn = job_arn
            sbi.requests = requests
            sbi.job_status = response["status"]

            return sbi

        except (KeyError, IndexError) as e:
            cls.logger.error(f"Invalid job response format: {e!s}")
            raise ValueError(f"Invalid job response format: {e!s}") from e
        except Exception as e:
            cls.logger.error(f"Failed to recover job details: {e!s}")
            raise RuntimeError(f"Failed to recover job details: {e!s}") from e
