import boto3
from dotenv import load_dotenv

from llmbo import BatchInferer, ModelInput

load_dotenv()
boto3.setup_default_session()


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
    job_name="my-first-inference-job",
    region="us-east-1",
    bucket_name="cddo-af-bedrock-batch-inference-us-east-1",
    role_arn="arn:aws:iam::992382722318:role/BatchInferenceRole",
)

bi.auto(inputs)
