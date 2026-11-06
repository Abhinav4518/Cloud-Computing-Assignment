import os
import json
import subprocess
import requests
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

def eval_hf_model(model_path: str, val_records: list, out_jsonl: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in val_records:
            prompt = f"Command: {rec['text']}\nJSON:"
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
            pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred_line = pred_text.splitlines()[0].strip() if pred_text else ""
            
            match = re.search(r'\{.*?\}', pred_line, re.DOTALL)
            if match:
                pred_line = match.group(0)
            else:
                if pred_line.startswith('"intent"') and not pred_line.startswith("{"):
                    pred_line = "{" + pred_line
                if not pred_line.endswith("}") and '"intent"' in pred_line:
                    pred_line += "}"
            
            try:
                parsed = json.loads(pred_line)
                fixed = {
                    "intent": parsed.get("intent", "weather_query"),
                    "date": parsed.get("date"),
                    "time": parsed.get("time"),
                    "place": parsed.get("place")
                }
                pred_line = json.dumps(fixed)
            except Exception:
                pred_line = json.dumps({"intent": "weather_query", "date": None, "time": None, "place": None})
                
            f.write(json.dumps({"id": rec["id"], "prediction": pred_line}) + "\n")

def eval_teacher(val_records: list, out_jsonl: str):
    with open("prompts/teacher_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in val_records:
            prompt = prompt_template.replace("{command}", rec["text"])
            resp = requests.post(
                OLLAMA_URL,
                json={"model": "phi3.5", "prompt": prompt, "stream": False, "options": {"temperature": 0, "seed": 42}},
                timeout=120,
            )
            raw = resp.json().get("response", "").strip()
            f.write(json.dumps({"id": rec["id"], "prediction": raw}) + "\n")

def main():
    os.makedirs("results", exist_ok=True)
    with open("data/validation.jsonl", "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f if line.strip()]

    print("Evaluating trained Student v1...")
    eval_hf_model("artifacts/models/v1", val_records, "results/student_v1_preds.jsonl")
    subprocess.run(["python", "tools/score.py", "results/student_v1_preds.jsonl", "data/validation.jsonl", "--out", "results/student_v1_validation.json", "--min-validity", "95"], check=True)

if __name__ == "__main__":
    main()
