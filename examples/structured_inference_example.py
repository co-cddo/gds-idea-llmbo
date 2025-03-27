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


class ListDogs(BaseModel):
    """A list of random dogs"""

    dogs: list[Dog]


inputs = {
    f"{i:03}": ModelInput(
        temperature=0.1,
        system="You are an expert in all kinds of dogs, not just labradors.",
        messages=[
            {
                "role": "user",
                "content": "I ❤ dogs! Give me 3 random dogs using the output schema.",
            }
        ],
    )
    for i in range(0, 100, 1)
}

sbi = StructuredBatchInferer(
    model_name="mistral.mistral-large-2407-v1:0",
    job_name="mistral-more-complicated-model-with-system-3-low-temp",
    region="us-west-2",
    bucket_name="cddo-af-bedrock-batch-inference-us-west-2",
    role_arn="arn:aws:iam::992382722318:role/BatchInferenceRole",
    output_model=ListDogs,
)

sbi.auto(inputs, poll_time_secs=15)
print(len([r for r in sbi.instances if r["outputModel"] is None]))
sbi.instances[0]
# {
#   'recordId': '000',
#   'outputModel': ListDogs(dogs=[
#       Dog(name='Max', breed='Golden Retriever', age=3),
#       Dog(name='Bella', breed='Labrador', age=5)
#       Dog(name='Bella', br...harlie', breed='German Shepherd', age=2)]
#   )
# }
