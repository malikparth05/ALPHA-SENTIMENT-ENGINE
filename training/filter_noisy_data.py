#!/usr/bin/env python3
# ===========================================
# Confident Learning — Data Filter Script
# ===========================================
# This script uses the "judge" model (trained on clean data)
# to filter out bad labels from StockTwits and Reddit data.
#
# It keeps only the examples where the judge AGREES with
# the human label. Disagreements are thrown out.
#
# USAGE:
#   python -m training.filter_noisy_data
# ===========================================

import torch
import numpy as np
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from training.fine_tune import (
    load_financial_phrasebank,
    load_twitter_financial,
    load_financial_tweets_combined,
    load_reddit_financial,
    clean_text,
    balance_labels,
)

# The judge model we just trained on clean data
JUDGE_MODEL = "models/my_finbert_judge"


def filter_with_judge(noisy_dataset: Dataset, dataset_name: str) -> Dataset:
    """
    Use the judge model to filter a noisy dataset.
    
    For each sentence:
      1. The judge reads it and makes a prediction
      2. If the judge's prediction MATCHES the human label → KEEP
      3. If they DISAGREE → THROW OUT (label is probably wrong)
    """
    print(f"\n🔍 Filtering {dataset_name} with the judge model...")
    print(f"   Input: {len(noisy_dataset)} examples")

    # Load the judge
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(JUDGE_MODEL)
    model.eval()

    # Use MPS (Apple GPU) if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    kept_texts = []
    kept_labels = []
    agreements = 0
    disagreements = 0

    batch_size = 32
    texts = noisy_dataset["text"]
    labels = noisy_dataset["label"]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        # Tokenize the batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get judge's predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

        # Compare judge's prediction with the human label
        for text, human_label, judge_prediction in zip(
            batch_texts, batch_labels, predictions
        ):
            if human_label == judge_prediction:
                # AGREEMENT → Keep this example!
                kept_texts.append(text)
                kept_labels.append(human_label)
                agreements += 1
            else:
                # DISAGREEMENT → Throw it out
                disagreements += 1

        # Progress indicator
        if (i // batch_size) % 50 == 0:
            total = agreements + disagreements
            if total > 0:
                pct = (agreements / total) * 100
                print(f"   ... processed {total}/{len(texts)} | kept {agreements} ({pct:.1f}%)")

    total = agreements + disagreements
    keep_rate = (agreements / total) * 100 if total > 0 else 0
    print(f"   ✅ Kept {agreements}/{total} ({keep_rate:.1f}%) | Threw out {disagreements}")

    return Dataset.from_dict({"text": kept_texts, "label": kept_labels})


def main():
    print("=" * 60)
    print("  🧹 Confident Learning: Filtering Noisy Data")
    print("=" * 60)

    # Step 1: Load the clean datasets (always kept)
    print("\n📚 Loading CLEAN datasets (always kept)...")
    d1 = load_financial_phrasebank()
    d2 = load_twitter_financial()

    # Step 2: Load the noisy datasets
    print("\n📚 Loading NOISY datasets (will be filtered)...")
    d3 = load_financial_tweets_combined()
    d4 = load_reddit_financial()

    # Step 3: Filter noisy datasets with the judge
    d3_filtered = filter_with_judge(d3, "StockTwits/Financial Tweets")
    d4_filtered = filter_with_judge(d4, "Reddit Stock Sentiment")

    # Step 4: Combine clean + filtered data
    print("\n" + "=" * 60)
    print("  📊 Combining Clean + Filtered Data")
    print("=" * 60)

    datasets_list = [d1, d2]
    if len(d3_filtered) > 0:
        datasets_list.append(d3_filtered)
    if len(d4_filtered) > 0:
        datasets_list.append(d4_filtered)

    combined = concatenate_datasets(datasets_list)
    combined = combined.shuffle(seed=42)

    labels = combined["label"]
    pos = labels.count(0)
    neg = labels.count(1)
    neu = labels.count(2)

    print(f"\n📊 TOTAL DATASET (before balancing): {len(combined)} examples")
    print(f"   🟢 Positive: {pos}")
    print(f"   🔴 Negative: {neg}")
    print(f"   🟡 Neutral:  {neu}")

    # Step 5: Balance the labels
    combined = balance_labels(combined)

    # Step 6: Save to disk so the training script can use it
    combined.save_to_disk("training/filtered_dataset")
    print(f"\n💾 Saved filtered + balanced dataset to training/filtered_dataset/")
    print(f"   Total: {len(combined)} examples")
    print("\n✅ Done! Now run the final training with this dataset.")


if __name__ == "__main__":
    main()
