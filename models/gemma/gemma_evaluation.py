!pip install transformers datasets evaluate accelerate pandas scikit-learn
!pip install rouge_score

from google.colab import drive
drive.mount('/content/drive')

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
import evaluate
from tqdm.auto import tqdm # for process bar

# --------------------------------------------------
# GPU check
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU")
else:
    print("CPU")


rouge = evaluate.load('rouge')

# --------------------------------------------------
# Data
# --------------------------------------------------

EVAL_DATA_PATH = "/content/drive/MyDrive/AiProject/Datasets/cleaned_eval_100.csv"

try:
    eval_df = pd.read_csv(EVAL_DATA_PATH)

    eval_df = eval_df[["cleaned_article", "cleaned_summary"]].dropna()

    eval_df = eval_df[eval_df["cleaned_article"].str.strip().str.len() > 5]
    eval_df = eval_df[eval_df["cleaned_summary"].str.strip().str.len() > 3]
    eval_df = eval_df.reset_index(drop=True)

    print(f"For evaluating {len(eval_df)} samples are loaded.")

    references = eval_df['cleaned_summary'].tolist()
    articles = eval_df['cleaned_article'].tolist()

except FileNotFoundError:
    print("err file not found")

    # --------------------------------------------------
# Model 1: BASE MODEL
# --------------------------------------------------
model_name_base = "Turkish-NLP/t5-efficient-small-MLSUM-TR-fine-tuned"
tokenizer_base = AutoTokenizer.from_pretrained(model_name_base)
model_base = AutoModelForSeq2SeqLM.from_pretrained(model_name_base).to(device)
model_base.eval() # Değerlendirme moduna al
print(f"Base Model ({model_name_base}) loaded.")

# --------------------------------------------------
# Model 2: FINE-TUNED MODEL
# --------------------------------------------------
model_path_finetuned = "/content/drive/MyDrive/AiProject/Models/t5_finetuned"
tokenizer_finetuned = AutoTokenizer.from_pretrained(model_path_finetuned)
model_finetuned = AutoModelForSeq2SeqLM.from_pretrained(model_path_finetuned).to(device)
model_finetuned.eval()
print(f"Fine-Tuned Model ({model_path_finetuned}) loaded.")

# to store generated summaries
base_predictions = []
finetuned_predictions = []

max_target_length = 64 # same as training

# Disable gradient calculations with torch.no_grad() (speeds up)
with torch.no_grad():
    for article in tqdm(articles, desc="Generating summaries..."):

        # --- Base Model Prediction ---
        inputs_base = tokenizer_base(
            article, return_tensors="pt", truncation=True, padding="max_length", max_length=256
        ).to(device)

        summary_ids_base = model_base.generate(
            **inputs_base, max_length=max_target_length, num_beams=4
        )
        summary_base = tokenizer_base.decode(summary_ids_base[0], skip_special_tokens=True)
        base_predictions.append(summary_base)

        # --- Fine-Tuned Model Prediction ---
        inputs_finetuned = tokenizer_finetuned(
            article, return_tensors="pt", truncation=True, padding="max_length", max_length=256
        ).to(device)

        summary_ids_finetuned = model_finetuned.generate(
            **inputs_finetuned, max_length=max_target_length, num_beams=4
        )
        summary_finetuned = tokenizer_finetuned.decode(summary_ids_finetuned[0], skip_special_tokens=True)
        finetuned_predictions.append(summary_finetuned)

print("All summaries generated successfully!")

print("ROUGE metrics are calculating...")

# Base model
rouge_scores_base = rouge.compute(
    predictions=base_predictions,
    references=references,
    use_stemmer=True
)

# Fine-tuned model
rouge_scores_finetuned = rouge.compute(
    predictions=finetuned_predictions,
    references=references,
    use_stemmer=True
)

# multiply by 100 to increase readability
results_data = {
    "Base Model": {k: v * 100 for k, v in rouge_scores_base.items()},
    "Fine-Tuned Model": {k: v * 100 for k, v in rouge_scores_finetuned.items()}
  }

df_results = pd.DataFrame(results_data)

df_results["Difference (Improvement)"] = df_results["Fine-Tuned Model"] - df_results["Base Model"]

pd.set_option('display.float_format', lambda x: '%.4f' % x)


results_file_path = "/content/drive/MyDrive/AiProject/Models/rouge_metrics_comparison.txt"

report_content = "--- ROUGE METRICS COMPARISON (0-100) ---\n\n"
report_content += df_results.to_string() # Convert DataFrame to string

try:
    with open(results_file_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n Metrics saved to file: {results_file_path}")
except Exception as e:
    print(f"\n err: {e}")

print("\n--- ROUGE METRICS COMPARISON (0-100) ---")
print(df_results)

print("\n\n--- FIRST EXAMPLE ---")
print(f"\n[ORIGINAL DATA]:\n{articles[0][:500]}...\n")
print(f"[REAL SUMMARY (Reference)]:\n{references[0]}\n")
print(f"[BASE MODEL SUMMARY]:\n{base_predictions[0]}\n")
print(f"[FINE-TUNED MODEL SUMARY]:\n{finetuned_predictions[0]}\n")