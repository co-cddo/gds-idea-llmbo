import json
import os
import time
from datetime import datetime
from genericpath import isfile
from select import poll
from typing import Dict, List
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from pydantic import BaseModel


class BatchInferer:
    def __init__(
        self,
        model_name: str,  # this should be an enum...404809
        system_prompt: str,
        input_bucket_name: str,
        job_name: str,
        role_arn: str,
        max_tokens: int = 2000,
    ):
        # model parameters
        self.model_name = model_name
        self.system_prompt = (
            system_prompt or "Extract the information into the schema provided"
        )
        self.max_tokens = max_tokens

        # file/bucket parameters
        self.bucket_name = input_bucket_name
        self.bucket_uri = "s3://" + input_bucket_name
        self.job_name = job_name or "batch_inference" + uuid4.uuid4()[:6]
        self.file_name = job_name + ".jsonl"

        self.role_arn = role_arn

        # validate inputs
        # should probably check that the s3 bucket exists
        self._check_for_profile()
        self._check_arn(self.role_arn)
        self.client = boto3.client("bedrock")

        self.batch_id = None
        self.requests = None

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

    def prepare_requests(self, inputs: Dict[str, List]):
        "this should create the jsonl"
        # maybe a data class conforming to this???
        #  https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
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
                self.batch_id = response["jobArn"]
                return response
        else:
            raise RuntimeError("There was an error creating the job")

    def download_results(self):
        if self.check_complete() == "COMPLETED":
            s3_client = boto3.client("s3")
            s3_client.download_file(
                Bucket=self.input_bucket,
                Key=f"output/{self.file_name}",
                Filename=f"results_{self.file_name}",
            )
            print("Downloaded results file")

            s3_client.download_file(
                Bucket=self.input_bucket,
                Key="output/manifest.json.out",
                Filename=f"{self.job_name}_manifest.json.out",
            )
            print("Downloaded manifest file")
        else:
            print("Batch was not marked COMPLETED")

    def load_results(self):
        # check the files exists
        if os.path.isfile(f"results_{self.file_name}") and os.path.isfile(
            f"{self.job_name}_manifest.json.out"
        ):
            self.results = self._read_jsonl(f"results_{self.file_name}")
            self.manifest = self._read_jsonl(f"{self.job_name}_manifest.json.out")
        else:
            raise FileExistsError("Result files do not exist")

    def cancel_batch(self):
        response = self.client.stop_model_invocation_job(jobIdentifier=self.batch_id)

        # This should check for a status 200 I think
        if response == {}:
            print(f"{self.job_name} with id={self.batch_id} was cancelled")
        else:
            raise RuntimeError("There was an error cancelling the job")

    def check_complete(self):
        response = self.client.get_model_invocation_job(jobArn=self.batch_id)

        # This should be a log
        print(f"Job status {response['status']}")

        if response["status"] in ["COMPLETED", "FAILED", "STOPPED", "EXPIRED"]:
            return response["status"]
        else:
            return None

    def poll_progress(self, poll_interval_seconds=60):
        while not self.check_progress():
            time.sleep(poll_interval_seconds)
        print(f"Batch {self.batch_id} ended with status ")


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

    model_name = "anthropic.claude-3-haiku-20240307-v1:0"

    bso = BatchInferer(
        model_name,
        job_name=f"my-first-inference-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        input_bucket_name="cddo-af-bedrock-batch-inference",
        role_arn="arn:aws:iam::992382722318:role/BatchInferenceRole",
        system_prompt="Extract the information into the schema provided",
    )

    # minimum data quantity is 100 rows
    data = {f"{i:03}": "tell me a short programming joke" for i in range(0, 100, 1)}

    # convert data to user messages

    messages = {key: [{"role": "user", "content": val}] for key, val in data.items()}

    print(type(messages))

    bso.prepare_requests(inputs=messages)
    print(bso.requests)

    bso.push_requests_to_s3()

    bso.create()

    bso.check_complete()


if __name__ == "__main__":
    main()
