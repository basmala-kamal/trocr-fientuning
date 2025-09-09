# finetuning.py

import os
import numpy as np
from PIL import Image

from datasets import load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.data.data_collator import default_data_collator

import evaluate
import torch

# -----------------------------
# 0) Config
# -----------------------------
CKPT = "anuashok/ocr-captcha-v3"     # base checkpoint
MAX_LABEL_LEN = 12                   # small (e.g., 6-char CAPTCHA + specials)
OUTPUT_DIR = "trocr-captcha-finetune"

TRAIN_CSV = "train.csv"
VALID_CSV = "valid.csv"

# Auto fp16 only if we have a CUDA device
FP16 = torch.cuda.is_available()

# -----------------------------
# 1) Processor & Model
# -----------------------------
processor = TrOCRProcessor.from_pretrained(CKPT)
model = VisionEncoderDecoderModel.from_pretrained(CKPT)

# Tighter generation config for short captchas (optional)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = MAX_LABEL_LEN
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 0

# -----------------------------
# 2) Load CSVs and normalize column names
# -----------------------------
ds = load_dataset(
    "csv",
    data_files={"train": TRAIN_CSV, "validation": VALID_CSV},
    delimiter=",",
)

def _pick_col(preferred, cols):
    low = {c: c.strip().lower() for c in cols}
    for want in preferred:
        for real, norm in low.items():
            if norm == want:
                return real
    return None

cols = ds["train"].column_names
img_col = _pick_col(["image", "img", "path", "filepath", "filename", "file"], cols)
lbl_col = _pick_col(["label", "text", "target"], cols)

if img_col is None or lbl_col is None:
    raise ValueError(f"Could not find image/label columns in CSV. Columns were: {cols}")

rename = {}
if img_col != "image":
    rename[img_col] = "image"
if lbl_col != "label":
    rename[lbl_col] = "label"
if rename:
    ds = ds.rename_columns(rename)

# Make image paths absolute so PIL can find them no matter where you run the script from
DATA_ROOT = os.getcwd()
def _make_abs_path(example):
    p = example["image"]
    if not os.path.isabs(p):
        example["image"] = os.path.normpath(os.path.join(DATA_ROOT, p))
    return example

ds = ds.map(_make_abs_path)

print("Columns after rename:", ds["train"].column_names)
print("Sample row:", ds["train"][0])

# -----------------------------
# 3) Preprocess → materialize tensors
# (batched=True is fine here; we’ll return torch tensors)
# -----------------------------
PAD_ID = processor.tokenizer.pad_token_id

def preprocess(batch):
    paths = batch["image"]
    imgs = [Image.open(p).convert("RGB") for p in paths]
    pixel_values = processor(images=imgs, return_tensors="pt").pixel_values  # [B,3,H,W]

    # tokenize labels to fixed length and mask pads with -100
    token_ids = processor.tokenizer(
        batch["label"],
        padding="max_length",
        max_length=MAX_LABEL_LEN,
        truncation=True,
    ).input_ids

    labels = [
        [(t if t != PAD_ID else -100) for t in ex_ids]
        for ex_ids in token_ids
    ]

    return {"pixel_values": pixel_values, "labels": labels}

# Remove original string columns so the Trainer sees only tensors/ids
cols_to_remove = ds["train"].column_names  # e.g. ["image","label"]
ds_proc = ds.map(preprocess, batched=True, remove_columns=cols_to_remove)

# Ensure torch format for Trainer
ds_proc.set_format(type="torch")

# -----------------------------
# 4) Metric (CER) — requires `pip install evaluate jiwer`
# -----------------------------
cer_metric = evaluate.load("cer")

def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits: [B, T, V]
    pred_ids = np.argmax(logits, axis=-1)

    preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    # Convert -100 back to PAD for decoding reference
    labels = np.where(labels == -100, PAD_ID, labels)
    refs = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

    cer = cer_metric.compute(
        predictions=[p.strip() for p in preds],
        references=[r.strip() for r in refs],
    )
    return {"cer": cer}

# -----------------------------
# 5) Training args (handle transformers version differences)
# -----------------------------
def build_args():
    common = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1.5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        num_train_epochs=5,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        predict_with_generate=False,   # we’re training with logits, not generation
        fp16=FP16,
        report_to="none",
    )
    # Try modern param first
    try:
        return Seq2SeqTrainingArguments(evaluation_strategy="steps", **common)
    except TypeError:
        # Fallback for older Transformers that used a different signature
        # or didn’t support `evaluation_strategy`.
        return Seq2SeqTrainingArguments(**common)

args = build_args()

# -----------------------------
# 6) Trainer — IMPORTANT: use tensor-friendly collator and DO NOT pass tokenizer
# (passing tokenizer makes Trainer pick a seq2seq collator that pads text)
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=ds_proc["train"],
    eval_dataset=ds_proc["validation"],
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

# -----------------------------
# 7) Train & Save
# -----------------------------
trainer.train()
trainer.save_model("trocr-captcha-finetuned")
processor.save_pretrained("trocr-captcha-finetuned")

print("✅ Training complete. Model saved to ./trocr-captcha-finetuned")
