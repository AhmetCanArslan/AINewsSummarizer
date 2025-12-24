import os
import argparse
import pandas as pd
import re


def main(args):
    base_dir = os.path.dirname(__file__)
    default_input = os.path.abspath(os.path.join(base_dir, "../../data/mergedDataset.csv"))
    default_out_dir = os.path.abspath(os.path.join(base_dir, "../../data/processed/splits"))
    default_out_file = os.path.join(default_out_dir, f"cleaned_eval_{args.nrows}.csv")
    default_alt_out = os.path.abspath(os.path.join(base_dir, f"../../data/cleaned_eval_{args.nrows}.csv"))

    input_path = args.input if args.input else default_input
    out_path = args.output if args.output else default_out_file

    if not os.path.exists(input_path):
        print(f"[ERROR] Merged dataset not found: {input_path}")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pd.read_csv(input_path).fillna("")
    # keep only rows with actual article and summary
    if "article_text" in df.columns and "summary" in df.columns:
        df = df[df["article_text"].str.strip().str.len() > 5]
        df = df[df["summary"].str.strip().str.len() > 3]
    else:
        print("[WARN] Expected columns 'article_text' and 'summary' not found, attempting to continue with any available text columns.")

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    extracted = df.iloc[: args.nrows].copy()

    # Ensure cleaned columns exist; try project's preprocess function first
    cleaned_article_col = "cleaned_article"
    cleaned_summary_col = "cleaned_summary"

    need_article_clean = cleaned_article_col not in extracted.columns or extracted[cleaned_article_col].isnull().all()
    need_summary_clean = cleaned_summary_col not in extracted.columns or extracted[cleaned_summary_col].isnull().all()

    if need_article_clean or need_summary_clean:
        try:
            # Try to import the project's preprocessing helper
            from preprocess import preprocess_turkish_text
            print("Using project preprocess.preprocess_turkish_text to create cleaned columns.")
            if need_article_clean:
                extracted[cleaned_article_col] = extracted.get("article_text", "").apply(preprocess_turkish_text)
            if need_summary_clean:
                extracted[cleaned_summary_col] = extracted.get("summary", "").apply(preprocess_turkish_text)
        except Exception:
            print("Could not import preprocess.preprocess_turkish_text; using fallback simple_clean.")
            if need_article_clean:
                extracted[cleaned_article_col] = extracted.get("article_text", "").apply(simple_clean)
            if need_summary_clean:
                extracted[cleaned_summary_col] = extracted.get("summary", "").apply(simple_clean)

    # Save to target output path and an alternative convenient path
    extracted.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(extracted)} rows to: {out_path}")

    # Save also to a top-level path for Colab usability if not the same
    if out_path != default_alt_out:
        os.makedirs(os.path.dirname(default_alt_out), exist_ok=True)
        extracted.to_csv(default_alt_out, index=False)
        print(f"[OK] Also saved to: {default_alt_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first N rows for evaluation from merged dataset")
    parser.add_argument("--nrows", type=int, default=100, help="Number of rows to extract (default: 100)")
    parser.add_argument("--input", type=str, default=None, help="Path to merged dataset CSV (default: data/final_dataset.csv)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: data/processed/splits/cleaned_eval_<nrows>.csv)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    args = parser.parse_args()
    main(args)