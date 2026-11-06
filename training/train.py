import os
import csv
import json
import yaml
import hashlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CommandDataset(Dataset):
    def __init__(self, records, tokenizer, max_len):
        max_len = max(max_len, 256)
        self.samples = []
        for r in records:
            prompt = f"Command: {r.get('text', '')}\nJSON: "
            target = r.get("target_json") or r.get("teacher_raw") or "{}"
            if not isinstance(target, str):
                target = json.dumps(target)
            full_text = prompt + target + tokenizer.eos_token

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = full_enc["input_ids"].squeeze(0)
            attention_mask = full_enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            # Mask prompt tokens and padding tokens
            mask_len = min(len(prompt_ids), max_len - 1)
            labels[:mask_len] = -100
            labels[attention_mask == 0] = -100

            # Guarantee at least valid tokens exist so loss never divides by 0
            if (labels != -100).sum() == 0:
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

            self.samples.append((input_ids, attention_mask, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def compute_dir_sha256(directory: str) -> str:
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(directory)):
        for fname in sorted(files):
            if fname.endswith(".sha256") or fname.endswith(".tar.gz"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
    return h.hexdigest()

def main():
    with open("training/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(os.environ.get("SEED_OVERRIDE", cfg.get("seed", 42)))
    epochs = int(os.environ.get("EPOCHS_OVERRIDE", cfg.get("epochs", 4)))
    version = os.environ.get("MODEL_VERSION", cfg.get("model_version", "v1"))
    subset = os.environ.get("SUBSET")

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("datasets/v1/train.jsonl", "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    limit = int(subset) if subset else 1500
    records = lines[:limit]
    print(f"Training model {version} on {len(records)} records (seed={seed}, epochs={epochs})...")

    model_name = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CRITICAL FIX: Force torch.float32 so AdamW does not overflow fp16 weights into NaN
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    ).to(device)

    dataset = CommandDataset(records, tokenizer, cfg.get("max_seq_length", 256))
    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 4), shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, eps=1e-8)

    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    model.train()
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        valid_batches = 0
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            valid_batches += 1

        avg_loss = total_loss / max(1, valid_batches)
        epoch_losses.append((epoch, round(avg_loss, 6)))
        print(f"Epoch {epoch}/{epochs} Loss: {avg_loss:.6f}")

    os.makedirs("training", exist_ok=True)
    with open("training/loss.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        writer.writerows(epoch_losses)

    out_dir = f"artifacts/models/{version}"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    digest = compute_dir_sha256(out_dir)
    sha_file = os.path.join(out_dir, "model.sha256")
    with open(sha_file, "w") as f:
        f.write(f"{digest}\n")

    print(f"Saved model {version} locally to {out_dir} with SHA256={digest}")

if __name__ == "__main__":
    main()
