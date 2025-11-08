from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
from tqdm import tqdm
import evaluate
import torch

MODEL_ID = "csebuetnlp/mT5_multilingual_XLSum"

DATA_CSV_PATH = "../../data/processed/cleaned_dataset.csv"

TEXT_COL = "cleaned_article"     
SUMMARY_COL = "cleaned_summary"   
OUTPUT_FILE = "firstModelOutput.csv"

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

#Create summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

df = pd.read_csv(DATA_CSV_PATH).fillna("")
print(f"✅ Loaded {len(df)} samples from CSV")

# Summarize
results = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row[TEXT_COL]
    ref = row[SUMMARY_COL]

    if not text:
        results.append({"original": text, "reference": ref, "summary": ""})
        continue

    try:
        input_text = f"summarize: {text}"

        summary = summarizer(
            input_text,
            max_length=150,
            min_length=40,
            do_sample=False,
            truncation=True
        )[0]["summary_text"]

    except Exception as e:
        print(f"⚠️ Error on row {i}: {e}")
        summary = ""

    results.append({
        "original": text,
        "reference": ref,
        "summary": summary
    })

out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_FILE, index=False)
print(f"Results are saved to {OUTPUT_FILE}")

rouge = evaluate.load("rouge")
preds = out_df["summary"].tolist()
refs = out_df["reference"].tolist()

metrics = rouge.compute(predictions=preds, references=refs)
print("\n📊 ROUGE results (F1 scores):")
for k, v in metrics.items():
    print(f"{k}: {v:.3f}")

# 📊 ROUGE results (F1 scores):
# rouge1: 0.391
# rouge2: 0.246
# rougeL: 0.331
# rougeLsum: 0.332