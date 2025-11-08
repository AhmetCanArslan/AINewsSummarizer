from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import math

MODEL_ID = "csebuetnlp/mT5_multilingual_XLSum"
DATA_PATH = "../../data/processed/cleaned_dataset.csv"
OUTPUT_DIR = "fine_tuned_mt5_turkish"
LOGS_DIR = "./logs"

TARGET_EFFECTIVE_BATCH_SIZE = 8

# dynamic hardware settings

dynamic_batch_size = 1
dynamic_grad_accum_steps = TARGET_EFFECTIVE_BATCH_SIZE // dynamic_batch_size
dynamic_eval_batch_size = dynamic_batch_size * 2
use_fp16 = False

if torch.cuda.is_available():

    # if GPU is available, get its properties
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU found: {device_name}, Total VRAM: {total_vram_gb:.2f} GB")
    use_fp16 = True

    # Determine batch size based on VRAM
    if total_vram_gb < 8:
        dynamic_batch_size = 1
        print(f"VRAM < 8GB. Memory saver mode: batch_size={dynamic_batch_size}")
    elif total_vram_gb < 16:
        # For 8GB / 10GB / 12GB cards
        dynamic_batch_size = 2
        print(f"Medium VRAM. Settings: batch_size={dynamic_batch_size}")
    else:
        # For 16GB+ cards (e.g. 4090, 3090, V100)
        dynamic_batch_size = 4
        print(f"High VRAM. Settings: batch_size={dynamic_batch_size}")

    # Calculate gradient accumulation steps based on the determined batch size
    # We use Math.ceil to ensure we meet the target even if it's not perfectly divisible
    dynamic_grad_accum_steps = math.ceil(TARGET_EFFECTIVE_BATCH_SIZE / dynamic_batch_size)
    dynamic_eval_batch_size = dynamic_batch_size * 2 # Evaluation step uses less VRAM

else:
    print("GPU not found. Training will continue on CPU.")
    # For CPU (usually RAM abundant), we cannot use fp16
    dynamic_batch_size = 4
    dynamic_grad_accum_steps = TARGET_EFFECTIVE_BATCH_SIZE // dynamic_batch_size
    dynamic_eval_batch_size = dynamic_batch_size
    use_fp16 = False

print("-" * 50)
print(f"Dynamic Settings:")
print(f"  per_device_train_batch_size: {dynamic_batch_size}")
print(f"  gradient_accumulation_steps: {dynamic_grad_accum_steps}")
print(f"  per_device_eval_batch_size:  {dynamic_eval_batch_size}")
print(f"  fp16 (Mixed Precision):    {use_fp16}")
print(f"  Effective Total Batch Size:    {dynamic_batch_size * dynamic_grad_accum_steps}")
print("-" * 50)

# model setup
# Load model and tokenizer
tokenizer = MT5Tokenizer.from_pretrained(MODEL_ID)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_ID)

# Load dataset
dataset = load_dataset("csv", data_files=DATA_PATH)
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

# Tokenization function
def preprocess_function(batch):
    # Add task prefix and use cleaned_article column
    inputs = ["summarize: " + doc for doc in batch["cleaned_article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Summaries (labels) use cleaned_summary column
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["cleaned_summary"], max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply to the entire dataset
tokenized_datasets = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=["url", "title", "summary", "article_text", "cleaned_summary", "cleaned_article"]
)

# Training arguments (using dynamic variables)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",        # Evaluate at the end of each epoch
    save_strategy="epoch",            # Save at the end of each epoch
    learning_rate=3e-5,
    
    per_device_train_batch_size=dynamic_batch_size,
    per_device_eval_batch_size=dynamic_eval_batch_size,
    gradient_accumulation_steps=dynamic_grad_accum_steps,
    fp16=use_fp16,
    
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,               # Keep only the best 2 checkpoints
    load_best_model_at_end=True,      # Load the best model at the end of training
    logging_dir=LOGS_DIR,
    logging_steps=50,
)

# Create Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Start training
try:
    trainer.train()
    print("✅ Fine-tuning completed!")

    # Save the results (the best model is already loaded with `load_best_model_at_end=True`)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model successfully saved to '{OUTPUT_DIR}'.")

except torch.cuda.OutOfMemoryError:
    print("\n" + "="*50)
    print("❌ ERROR: CUDA Out of Memory!")
    print("Your GPU VRAM is insufficient for this operation.")
    print(f"Current settings: batch_size={dynamic_batch_size}, grad_accum={dynamic_grad_accum_steps}")
    print("Try reducing the TARGET_EFFECTIVE_BATCH_SIZE variable at the beginning of the script (e.g., from 8 to 4).")
    print("="*50)