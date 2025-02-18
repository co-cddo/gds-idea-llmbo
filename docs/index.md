# LLMbo - Large Language model batch operations

AWS Bedrock offers powerful capabilities for running batch inference jobs with large language models. 
However, orchestrating these jobs, managing inputs and outputs, and ensuring consistent result structures can be arduous. 
LLMbo aims to solve these problems by providing an intuitive, Pythonic interface for Bedrock batch operations.

Additionally, it provides a method of using batch inference for structured responses, 
taking inspiration from the likes of [instructor](https://pypi.org/project/instructor/), [mirascope](https://pypi.org/project/mirascope/) and [pydanticai](https://pypi.org/project/pydantic-ai/). You provide a model output as a pydantic model and llmbo creates takes care of the rest.

Currently the library has only been texted with compatible Anthropic models. See the AWS
documentation for [models that support batch inference.](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html)

## Prerequisites 

- A `.env` file with an entry for `AWS_PROFILE=`. This profile should have sufficient 
permissions to create and schedule a batch inference job. See the [AWS instructions](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-permissions.html)
- [A service role with the required permissions to execute the job.](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-permissions.html#batch-inference-permissions-service), 
- A s3 bucket to store the input and outputs for the job. S3 buckets must exists in the same region as you execute the job.
    - Inputs will be written to `f{s3_bucket}/input/{job_name}.jsonl`
    - Outputs will be written to `f{s3_bucket}/output/{job_id}/{job_name}.jsonl.out` and 
      `f{s3_bucket}/output/{job_id}/manifest.json.out`


## Install
```bash 
pip install llmbo-bedrock
```

## Getting started

Here's a quick example of how to use LLMbo:

### BatchInferer
```python
from llmbo import BatchInferer, ModelInput

bi = BatchInferer(
    model_name="anthropic.claude-v2",
    bucket_name="my-inference-bucket",
    region="us-east-1",
    job_name="example-batch-job",
    role_arn="arn:aws:iam::123456789012:role/BedrockBatchRole"
)

# Prepare your inputs using the ModelInput class, you also need to include an id 
# input Dict[job_id: ModelInput ]

inputs = {
    f"{i:03}": ModelInput(
        messages=[{"role": "user", "content": f"Question {i}"}]
    ) for i in range(100)
}


# Run the batch job, this prepares, uploads, creates the job, monitors the progress 
# and downloads the results
results = bi.auto(inputs)
```

### StructuredInferer
For structured inference, simply define a Pydantic model and use StructuredBatchInferer:

```python
from pydantic import BaseModel
from llmbo import StructuredBatchInferer

class ResponseSchema(BaseModel):
    answer: str
    confidence: float

sbi = StructuredBatchInferer(
    output_model=ResponseSchema,
    model_name="anthropic.claude-v2",
    # ... other parameters
)

# The results will be validated against ResponseSchema
structured_results = sbi.auto(inputs)
```

For more detailed examples see the followingin llmbo.py:

- `batch_inference_example()`: for an example of free text response
- `structured_batch_inference_example()`: for an example of structured response ala instructor