import boto3
from dotenv import load_dotenv
from pydantic import BaseModel

from llmbo import ModelInput, StructuredBatchInferer

load_dotenv()
boto3.setup_default_session()


class Dog(BaseModel):
    name: str
    breed: str
    age: int


inputs = {
    f"{i:03}": ModelInput(
        temperature=1,
        top_k=250,
        messages=[{"role": "user", "content": "I ❤ dogs! Give me a random dog."}],
    )
    for i in range(0, 100, 1)
}

sbi = StructuredBatchInferer(
    model_name="anthropic.claude-3-haiku-20240307-v1:0",
    job_name="my-first-inference-job-1234",
    region="us-east-1",
    bucket_name="cddo-af-bedrock-batch-inference-us-east-1",
    role_arn="arn:aws:iam::992382722318:role/BatchInferenceRole",
    output_model=Dog,
)

sbi.auto(inputs)

sbi.instances[0]
# {'recordId': '000',
#  'outputModel': Dog(name='Buddy', breed='Labrador Retriever', age=5)
# }
