import os
import json
import hashlib
import boto3

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
ALLOWED_INTENTS = {
    "alarm_set", "alarm_query", "calendar_set", "calendar_query",
    "calendar_remove", "weather_query", "datetime_query", "recommendation_events"
}
REQUIRED_KEYS = {"intent", "date", "time", "place"}

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "any"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "any"),
    region_name="us-east-1",
)

def main():
    resp = s3.list_objects_v2(Bucket="datasets", Prefix="teacher/shard-")
    records = []
    for obj_meta in resp.get("Contents", []):
        body = s3.get_object(Bucket="datasets", Key=obj_meta["Key"])["Body"].read().decode("utf-8")
        for line in body.splitlines():
            if line.strip():
                records.append(json.loads(line))

    records.sort(key=lambda r: r["id"])

    counts = {
        "input": len(records),
        "a_invalid_json": 0,
        "b_wrong_keys": 0,
        "c_invalid_intent": 0,
        "d_invalid_slot": 0,
        "e_duplicates": 0,
        "kept": 0,
    }

    seen_texts = set()
    cleaned_lines = []

    for rec in records:
        raw = rec.get("teacher_raw", "").strip()
        cmd_lower = rec["text"].lower()

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                counts["a_invalid_json"] += 1
                continue
        except Exception:
            counts["a_invalid_json"] += 1
            continue

        if set(parsed.keys()) != REQUIRED_KEYS:
            counts["b_wrong_keys"] += 1
            continue

        if parsed.get("intent") not in ALLOWED_INTENTS:
            counts["c_invalid_intent"] += 1
            continue

        slot_ok = True
        for slot in ["date", "time", "place"]:
            val = parsed.get(slot)
            if val is not None:
                if not isinstance(val, str) or val.strip() == "" or val.lower() not in cmd_lower:
                    slot_ok = False
                    break
        if not slot_ok:
            counts["d_invalid_slot"] += 1
            continue

        norm_text = " ".join(cmd_lower.split())
        if norm_text in seen_texts:
            counts["e_duplicates"] += 1
            continue
        seen_texts.add(norm_text)

        clean_json_str = json.dumps(
            {
                "intent": parsed["intent"],
                "date": parsed["date"],
                "time": parsed["time"],
                "place": parsed["place"],
            },
             ensure_ascii=False,
        )
        cleaned_lines.append(
            json.dumps({"id": rec["id"], "text": rec["text"], "target_json": clean_json_str})
        )
        counts["kept"] += 1

    train_content = "\n".join(cleaned_lines) + "\n"
    sha256_hash = hashlib.sha256(train_content.encode("utf-8")).hexdigest()

    os.makedirs("results", exist_ok=True)
    with open("results/clean_counts.json", "w") as f:
        json.dump(counts, f, indent=2)

    s3.put_object(Bucket="datasets", Key="v1/train.jsonl", Body=train_content.encode("utf-8"))
    s3.put_object(Bucket="datasets", Key="v1/train.sha256", Body=f"{sha256_hash}  train.jsonl\n".encode("utf-8"))

    print("Cleaning Summary:", json.dumps(counts, indent=2))
    print("SHA-256:", sha256_hash)

if __name__ == "__main__":
    main()
