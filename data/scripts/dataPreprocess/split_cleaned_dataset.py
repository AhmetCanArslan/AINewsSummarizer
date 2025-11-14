import os
import pandas as pd

INPUT_PATH = "../../processed/cleaned_dataset.csv"
OUTPUT_DIR = "../../processed/splits"
EVAL_SIZE = 100
TRAIN_SIZES = [1000, 3000, 5000]
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH).fillna("")
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

eval_df = df.iloc[:EVAL_SIZE]
train_pool = df.iloc[EVAL_SIZE:].reset_index(drop=True)

eval_path = os.path.join(OUTPUT_DIR, f"cleaned_eval_{EVAL_SIZE}.csv")
eval_df.to_csv(eval_path, index=False)
print(f"Saved eval split -> {eval_path}")

for size in sorted(set(TRAIN_SIZES)):
    if size <= 0:
        continue
    if size > len(train_pool):
        print(f"Skipping {size}: only {len(train_pool)} training rows available.")
        continue
    subset = train_pool.iloc[:size]
    subset_path = os.path.join(OUTPUT_DIR, f"cleaned_train_{size}.csv")
    subset.to_csv(subset_path, index=False)
    print(f"Saved train split ({size} rows) -> {subset_path}")
