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

See:
- `batch_inference_example()`: for an example of free text response
- `structured_batch_inference_example`: for an example of structured response ala instructor

## To Do
- example data folder
- logging 
- Tests 
- `utils` module for creating the required AWS role etc.