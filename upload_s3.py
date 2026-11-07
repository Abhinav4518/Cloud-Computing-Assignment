import os
import time
import boto3
from botocore.config import Config

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:8333",
    aws_access_key_id="any",
    aws_secret_access_key="any",
    region_name="us-east-1",
    config=Config(retries={"max_attempts": 0}),
)

print("Waiting for SeaweedFS S3 gateway on port 8333 to become ready...")
for attempt in range(45):
    try:
        s3.list_buckets()
        print("SeaweedFS S3 is online!")
        break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("SeaweedFS S3 did not become ready in time.")

for b in ["datasets", "models"]:
    try:
        s3.create_bucket(Bucket=b)
    except Exception:
        pass

if os.path.exists("data/commands_pool.jsonl"):
    s3.upload_file("data/commands_pool.jsonl", "datasets", "commands_pool.jsonl")
    print("Uploaded data/commands_pool.jsonl to s3://datasets/commands_pool.jsonl")

if os.path.exists("datasets/v1/train.jsonl"):
    s3.upload_file("datasets/v1/train.jsonl", "datasets", "v1/train.jsonl")
    print("Uploaded datasets/v1/train.jsonl to s3://datasets/v1/train.jsonl")

for v in ["v1", "v2"]:
    v_dir = f"artifacts/models/{v}"
    if os.path.exists(v_dir):
        for fname in os.listdir(v_dir):
            fpath = os.path.join(v_dir, fname)
            if os.path.isfile(fpath):
                s3.upload_file(fpath, "models", f"{v}/{fname}")
        print(f"Uploaded model {v} to s3://models/{v}/")
