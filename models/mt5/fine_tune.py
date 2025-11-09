## Fine-tune mT5 Model for Turkish Text Summarization
## failure model !!!

from transformers import AutoTokenizer, MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import torch
import math
import pandas as pd
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


MODEL_ID = "csebuetnlp/mT5_multilingual_XLSum"
DATA_PATH = "./data/processed/cleaned_dataset.csv"
OUTPUT_DIR = "fine_tuned_mt5_turkish"
LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "training_log.csv")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target effective batch size
TARGET_EFFECTIVE_BATCH_SIZE = 2 

# GPU Settings
dynamic_batch_size = 1
dynamic_grad_accum_steps = max(1, TARGET_EFFECTIVE_BATCH_SIZE // dynamic_batch_size)
dynamic_eval_batch_size = dynamic_batch_size * 2
#use_fp16 = False

if torch.cuda.is_available():
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU found: {device_name}, Total VRAM: {total_vram_gb:.2f} GB")
    use_fp16 = True

    if total_vram_gb < 8:
        dynamic_batch_size = 1
    elif total_vram_gb < 16:
        dynamic_batch_size = 2
    else:
        dynamic_batch_size = 4

    dynamic_grad_accum_steps = math.ceil(TARGET_EFFECTIVE_BATCH_SIZE / dynamic_batch_size)
    dynamic_eval_batch_size = dynamic_batch_size * 2
else:
    print("GPU not found. Using CPU...")
    dynamic_batch_size = 4
    dynamic_grad_accum_steps = TARGET_EFFECTIVE_BATCH_SIZE // dynamic_batch_size
    dynamic_eval_batch_size = dynamic_batch_size
    use_fp16 = False

print("-" * 50)
print(f"Batch size per device: {dynamic_batch_size}")
print(f"Gradient accumulation: {dynamic_grad_accum_steps}")
print(f"Eval batch size:       {dynamic_eval_batch_size}")
print(f"FP16:                  {use_fp16}")
print("-" * 50)

# Load Model and Tokenizer
tokenizer = MT5Tokenizer.from_pretrained(MODEL_ID, legacy=False)

model = MT5ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",          # ✅ يجعل التحميل ذكيًا حسب VRAM
    torch_dtype="auto",         # ✅ يختار fp16 تلقائيًا إذا متاح
    low_cpu_mem_usage=True
)

# Reduce VRAM usage
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Load Dataset
dataset = load_dataset("csv", data_files=DATA_PATH)
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

# Preprocessing
def preprocess_function(batch):
    inputs = ["summarize: " + doc for doc in batch["cleaned_article"]]

    model_inputs = tokenizer(
        inputs,
        max_length=160,  # reduced for VRAM
        truncation=True,
        padding="longest"
    )

    labels = tokenizer(
        batch["cleaned_summary"],
        max_length=48,  # reduced for VRAM
        truncation=True,
        padding="longest"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function, 
    batched=True,
    remove_columns=["url", "title", "summary", "article_text", "cleaned_summary", "cleaned_article"]
)

# بعد التحويل أفرّغ الذاكرة من المتغيرات الكبيرة غير اللازمة
del dataset
torch.cuda.empty_cache()

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    #learning_rate=3e-5,
    per_device_train_batch_size=dynamic_batch_size,
    per_device_eval_batch_size=dynamic_eval_batch_size,
    gradient_accumulation_steps=dynamic_grad_accum_steps,
    fp16=True, 
    bf16=False,
    learning_rate=1e-5, 
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir=LOGS_DIR,
    logging_steps=50,
    logging_strategy="steps",
    report_to="none"
)
print("Tokekaadsda ada t::",tokenized_datasets["train"][0]["labels"][:20])

# Trainer with Early Stopping
class CSVLoggerCallback(EarlyStoppingCallback):
    def __init__(self, patience=2):
        super().__init__(early_stopping_patience=patience)
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["epoch"] = state.epoch
            self.logs.append(logs)
            pd.DataFrame(self.logs).to_csv(LOG_FILE, index=False)

callbacks = [CSVLoggerCallback(patience=2)]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    callbacks=callbacks
)

# Training Start
try:
    trainer.train()
    print("Fine-tuning completed successfully!")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to '{OUTPUT_DIR}'.")

except RuntimeError as e:
    # catch OOM runtime error and give hint
    if "out of memory" in str(e).lower():
        print("\nCUDA Out of Memory! Try lowering TARGET_EFFECTIVE_BATCH_SIZE or seq lengths.")
    raise

