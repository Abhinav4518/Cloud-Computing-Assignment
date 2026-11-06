import os
import json
import boto3
import requests

SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_COUNT = int(os.environ.get("SHARD_COUNT", "2"))
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
PROMPT_PATH = os.environ.get("PROMPT_PATH", "prompts/teacher_prompt.txt")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "any"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "any"),
    region_name="us-east-1",
)

def main():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    obj = s3.get_object(Bucket="datasets", Key="commands_pool.jsonl")
    lines = obj["Body"].read().decode("utf-8").splitlines()

    out_lines = []
    for idx, line in enumerate(lines):
        if idx % SHARD_COUNT != SHARD_INDEX:
            continue
        record = json.loads(line)
        prompt = prompt_template.replace("{command}", record["text"])

        payload = {
            "model": "phi3.5",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": 42, "num_predict": 100},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        raw_reply = resp.json().get("response", "").strip()

        out_record = {
            "id": record["id"],
            "text": record["text"],
            "teacher_raw": raw_reply,
        }
        out_lines.append(json.dumps(out_record))
        if len(out_lines) % 25 == 0:
            print(f"[Shard {SHARD_INDEX}] Processed {len(out_lines)} commands...")

    out_key = f"teacher/shard-{SHARD_INDEX}.jsonl"
    s3.put_object(
        Bucket="datasets",
        Key=out_key,
        Body=("\n".join(out_lines) + "\n").encode("utf-8"),
    )
    print(f"[Shard {SHARD_INDEX}] Done. Uploaded {len(out_lines)} records to datasets/{out_key}")

if __name__ == "__main__":
    main()
