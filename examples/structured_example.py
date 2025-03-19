import logging

import boto3
from dotenv import load_dotenv
from pydantic import BaseModel

from llmbo import ModelInput, StructuredBatchInferer

logger = logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

load_dotenv()
boto3.setup_default_session()


class Dog(BaseModel):
    """An instance of a dog."""

    name: str
    breed: str
    age: int


inputs = {
    f"{i:03}": ModelInput(
        temperature=1,
        messages=[{"role": "user", "content": "I ❤ dogs! Give me a random dog."}],
    )
    for i in range(0, 100, 1)
}

sbi = StructuredBatchInferer(
    model_name="mistral.mistral-large-2407-v1:0",
    job_name="my-first-mistral-inference-job-1234567",
    region="us-west-2",
    bucket_name="cddo-af-bedrock-batch-inference-us-west-2",
    role_arn="arn:aws:iam::992382722318:role/BatchInferenceRole",
    output_model=Dog,
)

sbi.auto(inputs, poll_time_secs=15)

sbi.instances[0]
# {'recordId': '000',
#  'outputModel': Dog(name='Buddy', breed='Labrador Retriever', age=5)
# }
