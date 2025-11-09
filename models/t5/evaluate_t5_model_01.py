# try fine-tune model, and compare its summaries with original pre-trained model summaries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import evaluate
import os

# Paths
MODEL_PATH = "./fine_tuned_t5_fixed"
DATA_PATH = "./data/processed/cleaned_dataset.csv"
OUTPUT_CSV = "./summary_comparison_results.csv"

# Load fine-tuned model
tokenizer_ft = AutoTokenizer.from_pretrained(MODEL_PATH)
model_ft = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Load original pre-trained model
tokenizer_orig = AutoTokenizer.from_pretrained("t5-small")
model_orig = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Load dataset
df = pd.read_csv(DATA_PATH)
samples = df.sample(5, random_state=42)

# Load ROUGE
rouge = evaluate.load("rouge")

# Prepare list to store results
results = []

for i, row in samples.iterrows():
    article = row["cleaned_article"]

    # Fine-tuned summary
    inputs_ft = tokenizer_ft(article, return_tensors="pt", truncation=True, max_length=512)
    outputs_ft = model_ft.generate(**inputs_ft, max_new_tokens=100, num_beams=4, do_sample=False, early_stopping=True)
    summary_ft = tokenizer_ft.decode(outputs_ft[0], skip_special_tokens=True)

    # Original model summary
    inputs_orig = tokenizer_orig(article, return_tensors="pt", truncation=True, max_length=512)
    outputs_orig = model_orig.generate(**inputs_orig, max_new_tokens=100, num_beams=4, do_sample=False, early_stopping=True)
    summary_orig = tokenizer_orig.decode(outputs_orig[0], skip_special_tokens=True)

    # Compute ROUGE
    rouge_ft = rouge.compute(predictions=[summary_ft], references=[row["cleaned_summary"]])
    rouge_orig = rouge.compute(predictions=[summary_orig], references=[row["cleaned_summary"]])

    # Append to results
    results.append({
        "title": row["title"],
        "reference_summary": row["cleaned_summary"],
        "fine_tuned_summary": summary_ft,
        "original_summary": summary_orig,
        "rouge1_ft": rouge_ft["rouge1"],
        "rouge2_ft": rouge_ft["rouge2"],
        "rougeL_ft": rouge_ft["rougeL"],
        "rouge1_orig": rouge_orig["rouge1"],
        "rouge2_orig": rouge_orig["rouge2"],
        "rougeL_orig": rouge_orig["rougeL"],
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved to {os.path.abspath(OUTPUT_CSV)}")
