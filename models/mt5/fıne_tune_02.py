import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
from sklearn.model_selection import train_test_split
import numpy as np

# --------------------------------------------------
# GPU check
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU found: {gpu_props.name}, VRAM: {gpu_props.total_memory / 1024**3:.2f} GB")
else:
    print("No GPU found, using CPU.")

# --------------------------------------------------
# Model
# --------------------------------------------------
model_name = "Turkish-NLP/t5-efficient-small-MLSUM-TR-fine-tuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# --------------------------------------------------
# Data
# --------------------------------------------------
DATA_PATH = "./data/processed/cleaned_dataset.csv"
df = pd.read_csv(DATA_PATH)[["cleaned_article", "cleaned_summary"]].dropna()

# Remove empty / too short entries
df = df[df["cleaned_article"].str.strip().str.len() > 5]
df = df[df["cleaned_summary"].str.strip().str.len() > 3]

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Loaded {len(df)} samples.")

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# --------------------------------------------------
# Tokenization
# --------------------------------------------------
max_input_length = 256
max_target_length = 64

def preprocess(examples):
    model_inputs = tokenizer(
        examples["cleaned_article"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=examples["cleaned_summary"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    # Replace padding token id's with -100 to ignore in loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"],
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --------------------------------------------------
# Training setup
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results_t5_fixed",
    eval_strategy="steps",
    eval_steps=400,
    save_steps=400,
    learning_rate=2e-5,           # safer LR
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
    fp16=False,                   # disable mixed precision to avoid NaN
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
)

callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callbacks,
)

# --------------------------------------------------
# Training
# --------------------------------------------------
print("\U0001f680 Starting training (no FP16, NaN-protected)...")
trainer.train()
print("\u2705 Training complete!")

# --------------------------------------------------
# Save model
# --------------------------------------------------
trainer.save_model("./fine_tuned_t5_fixed")
tokenizer.save_pretrained("./fine_tuned_t5_fixed")
print("\U0001f4be Model saved to ./fine_tuned_t5_fixed")

# --------------------------------------------------
# Example inference
# --------------------------------------------------
text = df["cleaned_article"].iloc[0]
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
summary_ids = model.generate(**inputs, max_length=64, num_beams=4)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\n\U0001f4dd Example summary:\n", summary)
