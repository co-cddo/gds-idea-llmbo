# Batch Inference

A library to make working with batch inference of LLM call in AWS easier. 

## Prerequisites 

- A `.env` file with an entry for `AWS_PROFILE=`. This profile should have sufficient 
permissions to execute a batch inference job. [find the linl 
- A s3 bucket to store the input and outputs for the job.   
    - Inputs will be written to `f{s3_bucket}/input/{job_name}.jsonl`
    - Outputs will be written to `f{s3_bucket}/output/{job_name}.jsonl