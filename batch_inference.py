import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from pydantic import BaseModel


@dataclass
class Manifest:
    totalRecordCount: int
    processedRecordCount: int
    successRecordCount: int
    errorRecordCount: int
    inputTokenCount: int
    outputTokenCount: int


@dataclass
class ToolChoice:
    type: Literal["any", "tool", "auto"]
    name: Optional[str] = None


@dataclass
class ModelInput:
    """A helper class to create model inputs"""

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


class BatchInferer:
    def __init__(
        self,
        model_name: str,  # this should be an enum...
        bucket_name: str,
        job_name: str,
        role_arn: str,
        max_tokens: int = 2000,
    ):
        # model parameters
        self.model_name = model_name
        self.max_tokens = max_tokens

        # file/bucket parameters
        self.bucket_name = bucket_name
        self.bucket_uri = "s3://" + bucket_name
        self.job_name = job_name or "batch_inference" + uuid4.uuid4()[:6]
        self.file_name = job_name + ".jsonl"

        self.role_arn = role_arn

        # validate inputs
        # should probably check that the s3 bucket exists
        self._check_for_profile()
        self._check_arn(self.role_arn)
        self.client = boto3.client("bedrock")

        self.job_arn = None
        self.requests = None

    @property
    def unique_id_from_arn(self):
        if not self.job_arn:
            raise ValueError("Job ARN not set")
        return self.job_arn.split("/")[-1]

    @staticmethod
    def _check_for_profile():
        if not os.getenv("AWS_PROFILE"):
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
                raise ValueError(f"Role '{role_name}' does not exist.")
            else:
                raise e

    def prepare_requests(self, inputs: List[ModelInput]):
        "this should create the jsonl"
        # maybe a data class conforming to this???
        #  https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

        if len(inputs) < 100:
            raise ValueError("Minimum Batch Size is 100")

        self.requests = [
            {
                "recordId": key,
                "modelInput": {
                    "anthropic_version": "bedrock-2023-05-31",
                    "system": self.system_prompt,
                    "max_tokens": self.max_tokens,
                    "messages": val,
                },
            }
            for key, val in inputs.items()
        ]

    def write_requests_locally(self):
        with open(self.file_name, "w") as file:
            for record in self.requests:
                file.write(json.dumps(record) + "\n")

    def push_requests_to_s3(self):
        "this should push the jsonl to s3"
        # do I want to write this file locally? - maybe stream it or write it to
        # temp file instead
        self.write_requests_locally()
        s3_client = boto3.client("s3")
        response = s3_client.upload_file(
            Filename=self.file_name,
            Bucket=self.bucket_name,
            Key=f"input/{self.file_name}",
            ExtraArgs={"ContentType": "application/json"},
        )
        return response

    def create(self):
        response = self.client.create_model_invocation_job(
            jobName=self.job_name,
            roleArn=self.role_arn,
            clientRequestToken="string",
            modelId=self.model_name,
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3InputFormat": "JSONL",
                    "s3Uri": f"{self.bucket_uri}/input/{self.file_name}",
                    # "s3BucketOwner": "string",
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": f"{self.bucket_uri}/output/",
                    # "s3EncryptionKeyId": "string",
                    # "s3BucketOwner": "string",
                }
            },
            timeoutDurationInHours=24,
            tags=[{"key": "bedrock_batch_inference", "value": self.job_name}],
        )

        if response:
            response_status = response["ResponseMetadata"]["HTTPStatusCode"]
            if response_status == 200:
                self.job_arn = response["jobArn"]
                return response
        else:
            raise RuntimeError("There was an error creating the job")

    def download_results(self):
        # TODO: This maybe should all check for "Stopped" -look into this
        valid_statuses = ["Completed"]
        if self.check_complete() in valid_statuses:
            file_name_, ext = os.path.splitext(self.file_name)
            self.output_file_name = f"{file_name_}_out{ext}"
            self.manifest_file_name = f"{file_name_}_manifest{ext}"

            s3_client = boto3.client("s3")
            s3_client.download_file(
                Bucket=self.bucket_name,
                Key=f"output/{self.unique_id_from_arn}/{self.file_name}.out",
                Filename=self.output_file_name,
            )
            print(f"Downloaded results file to {self.output_file_name}")

            s3_client.download_file(
                Bucket=self.bucket_name,
                Key=f"output/{self.unique_id_from_arn}/manifest.json.out",
                Filename=self.manifest_file_name,
            )
            print(f"Downloaded manifest file to {self.manifest_file_name}")
        else:
            print(f"Batch was not marked one of {valid_statuses}, could not download.")

    def load_results(self):
        # check the files exists
        if os.path.isfile(self.output_file_name) and os.path.isfile(
            self.manifest_file_name
        ):
            self.results = self._read_jsonl(self.output_file_name)
            self.manifest = self._read_jsonl(self.manifest_file_name)[0]
        else:
            raise FileExistsError(
                "Result files do not exist, you may need to call .download_results() first."
            )

    def cancel_batch(self):
        response = self.client.stop_model_invocation_job(jobIdentifier=self.job_arn)

        # This should check for a status 200 I think
        if response == {}:
            print(f"{self.job_name} with id={self.job_arn} was cancelled")
        else:
            raise RuntimeError("There was an error cancelling the job")

    def check_complete(self):
        response = self.client.get_model_invocation_job(jobIdentifier=self.job_arn)

        # This should be a log
        print(f"Job status {response['status']}")

        if response["status"] in ["Completed", "Failed", "Stopped", "Expired"]:
            return response["status"]
        else:
            return None

    def poll_progress(self, poll_interval_seconds=60):
        while not self.check_progress():
            time.sleep(poll_interval_seconds)
        print(f"Batch {self.job_arn} ended with status ")

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


class BatchInfererStructured(BatchInferer):
    def __init__(self, output_model):
        self.output_model = output_model
        self.tool = self._build_tool()
        super().__init__()

    def _build_tool(self):
        return [
            {
                "name": self.output_model.__name__,
                "description": "please fill in the schema",
                "input_schema": self.output_model.model_json_schema(),
            }
        ]

    def prepare_requests(self, inputs):
        "Call the super method to prepare the requests then add the tools"
        requests = super().prepare_requests(inputs)
        for request in requests:
            request["modelInput"]["tools"] = self.tool
        return requests

    def validate_outputs(self):
        raise NotImplementedError("Not Implemented Yet.")


class NameAgeModel(BaseModel):
    name: str
    age: int


def main():
    load_dotenv()
    boto3.setup_default_session()

    bi = BatchInferer.recover_details_from_job_arn(
        "arn:aws:bedrock:eu-west-2:992382722318:model-invocation-job/onrw6s8rcdgb"
    )

    bi.download_results()
    bi.load_results()

    print(bi.manifest)
    print(bi.results)
    # model_name = "anthropic.claude-3-haiku-20240307-v1:0"

    # bso = BatchInferer(
    #     model_name,
    #     # job_name=f"my-first-inference-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    #     job_name="my-first-inference-20250114-151301",
    #     input_bucket_name="cddo-af-bedrock-batch-inference",
    #     role_arn="arn:aws:iam::992382722318:role/BatchInferenceRole",
    #     system_prompt="Extract the information into the schema provided",
    # )

    # # minimum data quantity is 100 rows
    # data = {f"{i:03}": "tell me a short programming joke" for i in range(0, 100, 1)}

    # # convert data to user messages

    # messages = {key: [{"role": "user", "content": val}] for key, val in data.items()}

    # print(type(messages))

    # bso.prepare_requests(inputs=messages)
    # print(bso.requests)

    # bso.push_requests_to_s3()

    # # bso.create()
    # bso.job_arn = (
    #     "arn:aws:bedrock:eu-west-2:992382722318:model-invocation-job/onrw6s8rcdgb"
    # )

    # bso.check_complete()


if __name__ == "__main__":
    main()
