import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, Optional
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from pydantic import BaseModel

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
    """A data class conforming to the modelInputs as expected by AWS bedrock

    See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

    """

    #

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
    def __init__(
        self,
        model_name: str,  # this should be an enum...
        bucket_name: str,
        job_name: str,
        role_arn: str,
        time_out_duration_hours: int = 24,
    ):
        self.logger = logging.getLogger(f"{__name__}.BatchInferer")
        self.logger.info("Intialising BatchInferer")
        # model parameters
        self.model_name = model_name
        self.time_out_duration_hours = time_out_duration_hours

        # file/bucket parameters
        self.bucket_name = bucket_name
        self.bucket_uri = "s3://" + bucket_name
        self.job_name = job_name or "batch_inference" + uuid4.uuid4()[:6]
        self.file_name = job_name + ".jsonl"

        self.role_arn = role_arn

        # validate inputs
        # should probably check that the s3 bucket exists
        self.check_for_profile()
        self._check_arn(self.role_arn)
        self.client = boto3.client("bedrock")

        self.job_arn = None
        self.requests = None

        self.logger.info("Intialised BatchInferer")

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

    def _check_arn(self, role_arn: str):
        """
        Check if an IAM role with the given ARN exists.

        :param role_arn: The ARN of the IAM role to check
        :return: True if the role exists, False otherwise
        """
        # Extract the role name from the ARN
        role_name = role_arn.split("/")[-1]

        # Create an IAM client
        iam_client = boto3.client("iam")

        try:
            # Try to get the role
            iam_client.get_role(RoleName=role_name)
            print(f"Role '{role_name}' exists.")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                self.error(f"Role '{role_name}' does not exist.")
                raise ValueError(f"Role '{role_name}' does not exist.")
            else:
                raise e

    def prepare_requests(self, inputs: Dict[str, ModelInput]):
        "this should create the jsonl"
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

    def write_requests_locally(self):
        self.logger.info(f"Writing {len(self.requests)} requests to {self.file_name}")
        with open(self.file_name, "w") as file:
            for record in self.requests:
                file.write(json.dumps(record) + "\n")

    def push_requests_to_s3(self):
        "this should push the jsonl to s3"
        # do I want to write this file locally? - maybe stream it or write it to
        # temp file instead
        self.write_requests_locally()
        s3_client = boto3.client("s3")
        self.logger.info(f"Pushing {len(self.requests)} requests to {self.bucket_name}")
        response = s3_client.upload_file(
            Filename=self.file_name,
            Bucket=self.bucket_name,
            Key=f"input/{self.file_name}",
            ExtraArgs={"ContentType": "application/json"},
        )
        return response

    def create(self):
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
            self.logger.error(f"There was an error creating the job {self.job_name}")
            raise RuntimeError(f"There was an error creating the job {self.job_name}")

    def download_results(self):
        if self.check_complete() in VALID_FINISHED_STATUSES:
            file_name_, ext = os.path.splitext(self.file_name)
            self.output_file_name = f"{file_name_}_out{ext}"
            self.manifest_file_name = f"{file_name_}_manifest{ext}"
            self.logger.info(
                f"Job:{self.job_arn} Complete. Downloadingresults from {self.bucket_name}"
            )
            s3_client = boto3.client("s3")
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

    def load_results(self):
        # check the files exists
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

    def cancel_batch(self):
        response = self.client.stop_model_invocation_job(jobIdentifier=self.job_arn)

        # This should check for a status 200 I think
        if response == {}:
            print(f"{self.job_name} with id={self.job_arn} was cancelled")
        else:
            self.logger.error(f"There was an error cancelling the job {self.job_name}")
            raise RuntimeError(f"There was an error cancelling the job {self.job_name}")

    def check_complete(self):
        if self.job_status is not VALID_FINISHED_STATUSES:
            self.logger.info(f"Checking status of job {self.job_arn}")
            response = self.client.get_model_invocation_job(jobIdentifier=self.job_arn)

            self.logger.info(f"Job status is {response['status']}")

            if response["status"] in VALID_FINISHED_STATUSES:
                return response["status"]
            else:
                return None
        else:
            self.logger.info(f"Job {self.job_arn} is already {self.job_status}")
            return self.job_status

    def poll_progress(self, poll_interval_seconds=60):
        self.logger.info(f"Polling for progress every {poll_interval_seconds} seconds")
        while not self.check_complete():
            time.sleep(poll_interval_seconds)
        return True

    def auto(self, inputs: Dict[str, ModelInput]):
        self.prepare_requests(inputs)
        self.push_requests_to_s3()
        self.create()
        self.poll_progress(10 * 60)
        self.download_results()
        self.load_results()
        return self.results

    def recover_details_from_job_arn(job_arn):
        "I think i might need something like this to recover jobs when python has failed"
        client = boto3.client("bedrock")

        response = client.get_model_invocation_job(jobIdentifier=job_arn)

        if response:
            assert response["ResponseMetadata"]["HTTPStatusCode"] == 200, (
                "Didnt get a 200 response from Bedrock"
            )
            requests = BatchInferer._read_jsonl(response["jobName"] + ".jsonl")

            bi = BatchInferer(
                model_name=response["modelId"],
                # job_name=f"my-first-inference-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                job_name=response["jobName"],
                bucket_name=response["inputDataConfig"]["s3InputDataConfig"][
                    "s3Uri"
                ].split("/")[2],
                role_arn=response["roleArn"],
            )
            bi.job_arn = job_arn
            bi.requests = requests

            return bi
        else:
            raise ValueError("No response from Bedrock")


class StructuredBatchInferer(BatchInferer):
    def __init__(
        self,
        output_model: BaseModel,
        model_name: str,  # this should be an enum...
        bucket_name: str,
        job_name: str,
        role_arn: str,
        max_tokens: int = 2000,
    ):
        self.output_model = output_model
        self.tool = self._build_tool()
        super().__init__(model_name, bucket_name, job_name, role_arn, max_tokens)

    def _build_tool(self) -> dict:
        """convert a pydantic model into a tool defintion

        Returns:
            dict: tool description
        """
        return {
            "name": self.output_model.__name__,
            "description": self.output_model.__doc__ or "please fill in the schema",
            "input_schema": self.output_model.model_json_schema(),
        }

    def prepare_requests(self, inputs):
        "Add the tool then call the super method to prepare the requests"

        with_tools = {
            id: self._add_tool_to_model_input(model_input)
            for id, model_input in inputs.items()
        }
        super().prepare_requests(with_tools)

    def _add_tool_to_model_input(self, model_input: ModelInput) -> ModelInput:
        # perhaps this should be a method of the ModelInput??
        self.logger.info(f"Adding tool {self.tool['name']} to model input")
        model_input.tools = [self.tool]
        model_input.tool_choice = ToolChoice(
            type="tool", name=self.output_model.__name__
        )
        return model_input

    def load_results(self):
        # TODO: modify this to return a dict
        super().load_results()
        self.instances = [
            self.validate_result(result["modelOutput"]) for result in self.results
        ]

    def validate_result(
        self,
        result: dict,
    ):
        if not result["stop_reason"] == "tool_use":
            self.logger.error("Model did not use tool")
            raise ValueError("Model did not use tool")
        if not len(result["content"]) == 1:
            self.logger.error("Multiple instances of tool use per execution")
            raise ValueError("Multiple instances of tool use per execution")
        if result["content"][0]["type"] == "tool_use":
            try:
                output = self.output_model(**result["content"][0]["input"])
                return output
            except TypeError as e:
                self.logger.error(f"Could not validate output {e}")
                raise ValueError(f"Could not validate output {e}")


class NameAgeModel(BaseModel):
    name: str
    age: int


def batch_inference_example():
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
        bucket_name="cddo-af-bedrock-batch-inference",
        role_arn="arn:aws:iam::992382722318:role/BatchInferenceRole",
    )

    bi.prepare_requests(inputs)
    bi.push_requests_to_s3()
    bi.create()
    # arn:aws:bedrock:eu-west-2:992382722318:model-invocation-job/x3ddw33feqwu
    bi.poll_progress(10 * 60)
    bi.download_results()
    bi.load_results()

    # bi = BatchInferer.recover_details_from_job_arn(
    #     "arn:aws:bedrock:eu-west-2:992382722318:model-invocation-job/onrw6s8rcdgb"
    # )


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
    pass
    # setup_logging(log_level="INFO")  # or get from environment variable
    # logger.info("Starting batch inference process")
    # try:
    #     batch_inference_example()
    #     logger.info("Successfully completed batch inference")
    # except Exception as e:
    #     logger.error("Batch inference failed", exc_info=True)
    #     raise


if __name__ == "__main__":
    main()
