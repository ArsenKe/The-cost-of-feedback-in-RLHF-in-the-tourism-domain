import os
import argparse
import json
import torch
import pandas as pd
from firebase_admin import initialize_app, credentials, db
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from trl import DPOConfig, DPOTrainer
from peft import PeftModel
from huggingface_hub import HfApi

BASE_MODEL = os.environ.get("BASE_MODEL")
if not BASE_MODEL:
    raise RuntimeError("BASE_MODEL environment variable is required")

FIREBASE_DB_URL = os.environ.get("FIREBASE_DB_URL")
if not FIREBASE_DB_URL:
    raise RuntimeError("FIREBASE_DB_URL environment variable is required")

# DPO hyperparameters
DPO_OUTPUT_DIR = os.environ.get("DPO_OUTPUT_DIR", "dpo_out")
DPO_EPOCHS = int(os.environ.get("DPO_EPOCHS", "3"))
DPO_BATCH = int(os.environ.get("DPO_BATCH", "1")) 
DPO_LR = float(os.environ.get("DPO_LR", "1e-6"))
DPO_BETA = float(os.environ.get("DPO_BETA", "1e-3"))

# Initialize Firebase
creds_dict = json.loads(os.environ["FIREBASE_CREDENTIALS"])
cred = credentials.Certificate(creds_dict)
initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
fb_ref = db.reference("/feedback")

# Fetch feedback data
all_fb = fb_ref.get() or {}
records = []
for entry in all_fb.values():
    sel = entry.get("responses", {}).get("selected", "")
    if sel in ["Response 1", "Response 2"]:
        r1, r2 = entry["responses"]["response1"], entry["responses"]["response2"]
        chosen = r1 if sel == "Response 1" else r2
        rejected = r2 if sel == "Response 1" else r1
        records.append({
            "prompt": entry.get("prompt", ""),
            "chosen": chosen,
            "rejected": rejected
        })

if not records:
    print(" No valid feedback found. Exiting.")
    exit(0)

df = pd.DataFrame(records)
ds = Dataset.from_pandas(df)

# Load models and adapter
print("Loading base model and LoRA adapter…")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, os.environ["HF_ADAPTER_REPO"], is_trainable=True)
ref_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

# Tokenize data
def _prep(ex):
    return {
        "prompt_input_ids": tokenizer(ex["prompt"], truncation=True, padding="max_length", max_length=128).input_ids,
        "chosen_input_ids": tokenizer(ex["chosen"], truncation=True, padding="max_length", max_length=128).input_ids,
        "rejected_input_ids": tokenizer(ex["rejected"], truncation=True, padding="max_length", max_length=128).input_ids,
    }

tok_ds = ds.map(_prep, remove_columns=ds.column_names)

# DPO training
print("Starting DPO training…")
dpo_cfg = DPOConfig(
    output_dir=DPO_OUTPUT_DIR,
    per_device_train_batch_size=DPO_BATCH,
    gradient_accumulation_steps=1,
    num_train_epochs=DPO_EPOCHS,
    learning_rate=DPO_LR,
    bf16=torch.cuda.is_bf16_supported(),
    beta=DPO_BETA,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_cfg,
    train_dataset=tok_ds,
    processing_class=tokenizer,
)

trainer.train()

# Push adapter to Hugging Face
print("Pushing updated LoRA adapter back to HF…")
hf = HfApi()
hf.upload_folder(
    folder_path=DPO_OUTPUT_DIR,
    path_in_repo=".",
    repo_id=os.environ["HF_ADAPTER_REPO"],
    repo_type="model",
    token=os.environ["HF_TOKEN"]
)
print(" Done.")