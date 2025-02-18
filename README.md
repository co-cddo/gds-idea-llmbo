# LLMbo-bedrock - Large language model batch operations

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

## Documentation

[See the full documentation.](https://co-cddo.github.io/gds-idea-llmbo/)

## Install
```bash 
pip install llmbo-bedrock
```

## Usage

See:
- `batch_inference_example()`: for an example of free text response
- `structured_batch_inference_example()`: for an example of structured response ala instructor


## Developing 

The project uses [uv](https://docs.astral.sh/uv/) for development. To contribute:
```bash
git clone https://github.com/co-cddo/gds-idea-llmbo.git
cd gds-idea-llmbo
uv sync --dev 
```
Please create a pull request.


## To Do
- example data folder: Move the examples to a dedicated folder with some example data to demo the operations
- Tests: There are some but more are needed
- `utils` module for creating the required AWS role etc.