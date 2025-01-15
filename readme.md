# Batch Inference

A library to make working with batch inference of LLM call in AWS easier. 
Currently support is limited to Anthropic models.

## Prerequisites 

- A `.env` file with an entry for `AWS_PROFILE=`. This profile should have sufficient 
permissions to execute a batch inference job. [find the link]
- A role with the required permissions [find details]
- A s3 bucket to store the input and outputs for the job.   
    - Inputs will be written to `f{s3_bucket}/input/{job_name}.jsonl`
    - Outputs will be written to `f{s3_bucket}/output/{job_id}/{job_name}.jsonl.out` and 
      `f{s3_bucket}/output/{job_id}/manifest.json.out`


## Usage

```python 

from batch_inferer import BatchInferer
# assuming you have a .env as per the prerequisites 
load_dotenv()
boto3.setup_default_session()

# Intiate the inference object
bi = BatchInferer(
    model_name = "anthropic.claude-3-haiku-20240307-v1:0",
    # it is recommended to create a unique job name with a timestamp or uuid
    job_name=f"my-first-inference-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    # job_name="my-first-inference-20250114-151301",
    input_bucket_name="cddo-af-bedrock-batch-inference",
    role_arn="arn:aws:iam::992382722318:role/BatchInferenceRole",
    )



```