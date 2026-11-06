import os
import json
from urllib.parse import urlparse
import boto3
import torch
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

MODEL_URI = os.environ.get("MODEL_URI", "s3://models/v1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
POD_NAME = os.environ.get("HOSTNAME", "local-pod")
SCHEMA_PATH = os.environ.get("SCHEMA_PATH", "data/schema.json")

ALLOWED_INTENTS = {
    "alarm_set", "alarm_query", "calendar_set", "calendar_query",
    "calendar_remove", "weather_query", "datetime_query", "recommendation_events"
}
REQUIRED_KEYS = {"intent", "date", "time", "place"}

state = {"ready": False, "model": None, "tokenizer": None}

class ParseRequest(BaseModel):
    text: str

def validate_against_schema(raw_output: str) -> dict | None:
    if not raw_output:
        return None
    s = raw_output.strip()
    if s.startswith('"intent"') and not s.startswith("{"):
        s = "{" + s
    try:
        data = json.loads(s)
        if not isinstance(data, dict):
            return None
        if set(data.keys()) != REQUIRED_KEYS:
            return None
        if data.get("intent") not in ALLOWED_INTENTS:
            return None
        for k in ("date", "time", "place"):
            if data.get(k) is not None and not isinstance(data.get(k), str):
                return None
        return data
    except Exception:
        return None

@app.on_event("startup")
def load_model_from_s3():
    if os.environ.get("SKIP_MODEL_LOAD") == "1":
        return
    parsed = urlparse(MODEL_URI)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    local_dir = f"/tmp/model_{MODEL_VERSION}"
    os.makedirs(local_dir, exist_ok=True)

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "any"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "any"),
        region_name="us-east-1",
    )
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        fname = os.path.basename(key)
        if fname:
            s3.download_file(bucket, key, os.path.join(local_dir, fname))

    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(local_dir)
    model.eval()

    state["tokenizer"] = tokenizer
    state["model"] = model
    state["ready"] = True

@app.get("/healthz")
def healthz():
    return Response(content='{"status":"ok"}', media_type="application/json", status_code=200)

@app.get("/readyz")
def readyz():
    if not state["ready"]:
        return Response(content='{"status":"loading"}', media_type="application/json", status_code=503)
    return Response(content='{"status":"ready"}', media_type="application/json", status_code=200)

@app.post("/parse")
def parse_command(req: ParseRequest):
    headers = {
        "X-Model-Version": MODEL_VERSION,
        "X-Served-By": os.environ.get("HOSTNAME", POD_NAME),
    }

    if os.environ.get("FAULT_INJECT") == "1":
        raw_output = "not json"
    else:
        if not state["ready"]:
            return JSONResponse(status_code=503, content={"error": "model_not_ready"}, headers=headers)
        prompt = f"Command: {req.text}\nJSON:"
        inputs = state["tokenizer"](prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = state["model"].generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=state["tokenizer"].eos_token_id,
                eos_token_id=state["tokenizer"].eos_token_id,
            )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = state["tokenizer"].decode(gen_ids, skip_special_tokens=True).strip()
        raw_output = decoded.splitlines()[0].strip() if decoded else ""

    validated = validate_against_schema(raw_output)
    if validated is None:
        return JSONResponse(
            status_code=500,
            content={"error": "invalid_model_output"},
            headers=headers,
        )

    return JSONResponse(status_code=200, content=validated, headers=headers)