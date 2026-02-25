#!/usr/bin/env python3
# ===========================================
# Alpha Sentiment Engine — Fine-Tuning Script
# ===========================================
# This script does ACTUAL ML TRAINING!
#
# What it does:
#   1. Downloads 4 financial sentiment datasets
#   2. Combines them into one big dataset
#   3. Fine-tunes FinBERT on the combined data
#   4. Compares base model vs YOUR fine-tuned model
#   5. Saves YOUR model to disk
#
# Think of it as: taking a smart finance student (FinBERT)
# and putting them through extra training with YOUR data
# so they become even smarter at YOUR specific task.
#
# USAGE:
#   python -m training.fine_tune
#
# TIME: ~1-2 hours on CPU (your Mac), ~15 min on GPU
# ===========================================

import os
import re
import json
from datetime import datetime, timezone

import torch
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ═══════════════════════════════════════════
#  STEP 1: SETTINGS (you can tweak these)
# ═══════════════════════════════════════════
# These are the "knobs" that control training.
# We picked safe defaults, but you can experiment!

MODEL_NAME: str = "ProsusAI/finbert"             # Start from FinBERT
OUTPUT_DIR: str = "models/my_finbert"              # Save the final model here
EPOCHS: int = 3                                    # How many times to read all data
BATCH_SIZE: int = 8                                # Process 8 sentences at a time (smaller to save GPU memory)
LEARNING_RATE: float = 2e-5                        # How big the adjustments are
MAX_LENGTH: int = 128                              # Max words per sentence

# Label mapping: we standardize all datasets to these 3 labels
LABEL_MAP: dict = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
}


# ═══════════════════════════════════════════
#  STEP 2: CLEAN & LOAD DATASETS
# ═══════════════════════════════════════════
# We download 4 different datasets, CLEAN the text,
# BALANCE the labels, then combine them.

def clean_text(text: str) -> str:
    """
    Clean messy text before feeding it to the AI.
    
    WHY: Reddit and Twitter data has URLs, @mentions, 
    hashtags, emojis, and junk that confuses the model.
    Cleaning this makes the AI focus on the WORDS, not noise.
    
    Example:
      Before: "$AAPL 🚀🚀 to the moon!! @elonmusk https://t.co/xyz"
      After:  "AAPL to the moon"
    """
    # Remove URLs (https://... or http://...)
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove @mentions (@elonmusk → gone)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags symbols but keep the word (#bullish → bullish)
    text = re.sub(r'#', '', text)
    
    # Keep $ for stock tickers ($AAPL stays as AAPL)
    text = re.sub(r'\$', '', text)
    
    # Remove emojis and special unicode characters
    text = re.sub(r'[^\w\s.,!?\'-]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def balance_labels(dataset: Dataset) -> Dataset:
    """
    Balance the dataset so each label has equal representation.
    
    WHY: Our dataset has ~75% neutral labels. The AI sees "neutral"
    so often that it learns to just guess "neutral" for everything.
    By making all 3 labels equal, the AI learns all of them fairly.
    
    HOW: We "undersample" the majority class (neutral) by randomly
    picking only as many neutral examples as the smallest class.
    
    Example:
      Before: Positive=8000, Negative=6000, Neutral=43000
      After:  Positive=6000, Negative=6000, Neutral=6000
      (We keep all of the smallest class, and randomly pick
       the same number from the larger classes)
    """
    print("\n⚖️ Balancing labels...")
    
    labels = dataset["label"]
    texts = dataset["text"]
    
    # Group examples by label
    groups = {0: [], 1: [], 2: []}  # 0=pos, 1=neg, 2=neutral
    for i, label in enumerate(labels):
        groups[label].append(i)
    
    # Find the size of the smallest group
    min_count = min(len(g) for g in groups.values())
    
    label_names = {0: "Positive", 1: "Negative", 2: "Neutral"}
    for label_id, indices in groups.items():
        print(f"   {label_names[label_id]}: {len(indices)} → {min_count}")
    
    # Randomly pick min_count examples from each group
    np.random.seed(42)
    balanced_indices = []
    for label_id in groups:
        chosen = np.random.choice(groups[label_id], size=min_count, replace=False)
        balanced_indices.extend(chosen.tolist())
    
    # Shuffle the balanced indices
    np.random.shuffle(balanced_indices)
    
    balanced = Dataset.from_dict({
        "text": [texts[i] for i in balanced_indices],
        "label": [labels[i] for i in balanced_indices],
    })
    
    print(f"   ✅ Balanced dataset: {len(balanced)} examples ({min_count} per class)")
    return balanced

def load_financial_phrasebank() -> Dataset:
    """
    Dataset 1: Financial PhraseBank
    - ~5,800 sentences from financial news
    - Labeled by human experts
    - Highest quality dataset we have

    Example: "Strong earnings growth reported" → positive
    """
    print("\n📚 Loading Dataset 1: Financial PhraseBank...")

    # Load from HuggingFace (modern Parquet format)
    dataset = load_dataset("mltrev23/financial-sentiment-analysis")
    data = dataset["train"]

    # Map string labels to numbers matching FinBERT's order
    label_map = {"positive": 0, "negative": 1, "neutral": 2}

    texts = []
    labels = []
    for row in data:
        text = row.get("Sentence", row.get("text", ""))
        raw_label = row.get("Sentiment", row.get("label", ""))
        if text and raw_label:
            label_str = str(raw_label).lower().strip()
            if label_str in label_map:
                cleaned = clean_text(text)
                if cleaned:  # Skip if cleaning removed everything
                    texts.append(cleaned)
                    labels.append(label_map[label_str])

    result = Dataset.from_dict({"text": texts, "label": labels})
    print(f"   ✅ Loaded {len(result)} sentences")
    return result


def load_twitter_financial() -> Dataset:
    """
    Dataset 2: Twitter Financial News Sentiment
    - ~12,000 financial tweets
    - Labeled as bearish/bullish/neutral

    Example: "$AAPL crushing it today 🚀" → positive (bullish)
    """
    print("\n📚 Loading Dataset 2: Twitter Financial News...")

    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

    # Combine train and validation splits
    all_data = []
    for split_name in dataset:
        split_data = dataset[split_name]
        for row in split_data:
            text = row["text"]
            label = row["label"]
            # Labels: 0=bearish(negative), 1=bullish(positive), 2=neutral
            # Remap to FinBERT: 0=positive, 1=negative, 2=neutral
            remap = {0: 1, 1: 0, 2: 2}
            cleaned = clean_text(text)
            if cleaned:
                all_data.append({"text": cleaned, "label": remap[label]})

    result = Dataset.from_dict({
        "text": [d["text"] for d in all_data],
        "label": [d["label"] for d in all_data],
    })
    print(f"   ✅ Loaded {len(result)} tweets")
    return result


def load_financial_tweets_combined() -> Dataset:
    """
    Dataset 3: Combined Financial Tweets (StockTwits + others)
    - ~38,000 stock market tweets from multiple sources
    - Column is 'tweet' (not 'text')
    - Sentiment: 0=bearish, 1=bullish, 2=neutral

    Example: "Just bought more $TSLA, this dip is a gift" → positive
    """
    print("\n📚 Loading Dataset 3: Financial Tweets (StockTwits + others)...")

    try:
        dataset = load_dataset("TimKoornstra/financial-tweets-sentiment")
        data = dataset["train"]

        # This dataset uses 'tweet' column and numeric 'sentiment'
        # Sentiment: 0=bearish(negative), 1=bullish(positive), 2=neutral
        # Remap to FinBERT order: 0=positive, 1=negative, 2=neutral
        remap = {0: 1, 1: 0, 2: 2}

        texts = []
        labels = []
        for row in data:
            text = row.get("tweet", "")
            raw_label = row.get("sentiment", None)

            if not text or raw_label is None:
                continue

            label = remap.get(int(raw_label), 2)
            cleaned = clean_text(text)
            if cleaned:
                texts.append(cleaned)
                labels.append(label)

        result = Dataset.from_dict({"text": texts, "label": labels})
        print(f"   ✅ Loaded {len(result)} tweets")
        return result

    except Exception as e:
        print(f"   ⚠️ Could not load this dataset: {e}")
        print("   Skipping — we still have other datasets!")
        return Dataset.from_dict({"text": [], "label": []})


def load_reddit_financial() -> Dataset:
    """
    Dataset 4: Reddit Stock Sentiment
    - ~1,300 posts from r/stocks and r/wallstreetbets
    - Real Reddit discussion with slang and informal language
    - Label: -1.0 (negative), 0.0 (neutral), 1.0 (positive)

    Example: "GME to the moon 🚀🚀" → positive
    """
    print("\n📚 Loading Dataset 4: Reddit Stock Sentiment...")

    try:
        dataset = load_dataset("johntoro/Reddit-Stock-Sentiment")

        # Remap: -1.0 → negative (1), 0.0 → neutral (2), 1.0 → positive (0)
        label_remap = {-1: 1, 0: 2, 1: 0}

        texts = []
        labels = []
        for split_name in dataset:
            for row in dataset[split_name]:
                # Use title + text for more content
                title = row.get("title", "") or ""
                text = row.get("text", "") or ""
                combined_text = f"{title} {text}".strip()

                raw_label = row.get("label", None)
                if not combined_text or raw_label is None:
                    continue

                label = label_remap.get(int(float(raw_label)), 2)
                cleaned = clean_text(combined_text)
                if cleaned:
                    texts.append(cleaned)
                    labels.append(label)

        result = Dataset.from_dict({"text": texts, "label": labels})
        print(f"   ✅ Loaded {len(result)} Reddit posts")
        return result

    except Exception as e:
        print(f"   ⚠️ Could not load this dataset: {e}")
        print("   Skipping — we still have other datasets!")
        return Dataset.from_dict({"text": [], "label": []})


def combine_all_datasets(clean_only: bool = False) -> Dataset:
    """
    Load datasets and combine them.
    
    clean_only=True:  Only PhraseBank + Twitter (high quality, for judge model)
    clean_only=False: All 4 datasets (for full training)
    """
    print("\n" + "=" * 60)
    print("  📚 STEP 1: Loading & Combining All Datasets")
    print("=" * 60)

    datasets_list = []

    # Load each dataset (if one fails, we still have the others)
    d1 = load_financial_phrasebank()
    datasets_list.append(d1)

    d2 = load_twitter_financial()
    datasets_list.append(d2)

    if not clean_only:
        d3 = load_financial_tweets_combined()
        if len(d3) > 0:
            datasets_list.append(d3)

        d4 = load_reddit_financial()
        if len(d4) > 0:
            datasets_list.append(d4)
    else:
        print("\n   ⚡ CLEAN ONLY MODE: Skipping StockTwits and Reddit datasets")

    # Combine all datasets into one
    combined = concatenate_datasets(datasets_list)

    # Shuffle the data (mix all sources together)
    combined = combined.shuffle(seed=42)

    print(f"\n📊 TOTAL COMBINED DATASET (before balancing): {len(combined)} examples")

    # Show the breakdown
    labels = combined["label"]
    positive = labels.count(0)
    negative = labels.count(1)
    neutral = labels.count(2)
    print(f"   🟢 Positive: {positive}")
    print(f"   🔴 Negative: {negative}")
    print(f"   🟡 Neutral:  {neutral}")

    # Balance the labels so the AI learns all 3 equally
    combined = balance_labels(combined)

    return combined


# ═══════════════════════════════════════════
#  STEP 3: PREPARE FOR TRAINING
# ═══════════════════════════════════════════

def tokenize_data(dataset: Dataset, tokenizer) -> Dataset:
    """
    Turn text into numbers the model can understand.
    Same tokenization as before, but for the whole dataset at once.
    """
    print("\n" + "=" * 60)
    print("  🔤 STEP 2: Tokenizing (text → numbers)")
    print("=" * 60)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize_function, batched=True)
    print(f"   ✅ Tokenized {len(tokenized)} examples")
    return tokenized


def compute_metrics(eval_pred):
    """
    Calculate how good the model is.
    Called automatically during training to track progress.
    """
    predictions, labels = eval_pred
    predicted_labels = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels, average="weighted")

    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
    }


# ═══════════════════════════════════════════
#  STEP 4: TRAIN! (the main event)
# ═══════════════════════════════════════════

def train() -> None:
    """
    The main training function. Does everything end-to-end:
    1. Load datasets
    2. Load base model
    3. Train (fine-tune)
    4. Evaluate
    5. Save
    """

    # ---- Load the filtered dataset (clean + AI-verified) ----
    print("\n" + "=" * 60)
    print("  📚 Loading filtered dataset (clean + AI-verified)")
    print("=" * 60)
    
    from datasets import load_from_disk
    combined = load_from_disk("training/filtered_dataset")
    print(f"   ✅ Loaded {len(combined)} examples")
    
    labels = combined["label"]
    print(f"   🟢 Positive: {labels.count(0)}")
    print(f"   🔴 Negative: {labels.count(1)}")
    print(f"   🟡 Neutral:  {labels.count(2)}")

    # ---- Split into train (80%) and test (20%) ----
    print("\n" + "=" * 60)
    print("  ✂️ Splitting: 80% train, 20% test")
    print("=" * 60)

    split = combined.train_test_split(test_size=0.2, seed=42)
    train_data = split["train"]
    test_data = split["test"]
    print(f"   📖 Training set: {len(train_data)} examples")
    print(f"   📝 Test set:     {len(test_data)} examples")

    # ---- Load tokenizer ----
    print("\n" + "=" * 60)
    print("  🤖 Loading FinBERT base model")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ---- Tokenize all data ----
    train_tokenized = tokenize_data(train_data, tokenizer)
    test_tokenized = tokenize_data(test_data, tokenizer)

    # ---- Load the base model ----
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,  # positive, negative, neutral
    )
    print(f"   ✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ---- Evaluate BASE model first (before training) ----
    print("\n" + "=" * 60)
    print("  📊 STEP 3: Evaluating BASE model (before training)")
    print("=" * 60)

    base_trainer = Trainer(
        model=model,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )
    base_results = base_trainer.evaluate()
    base_accuracy = base_results["eval_accuracy"]
    base_f1 = base_results["eval_f1"]
    print(f"\n   📉 Base FinBERT accuracy:  {base_accuracy:.2%}")
    print(f"   📉 Base FinBERT F1 score:  {base_f1:.4f}")

    # ---- Set up training ----
    print("\n" + "=" * 60)
    print("  🏋️ STEP 4: TRAINING (this will take a while!)")
    print("=" * 60)
    print(f"   Epochs:        {EPOCHS}")
    print(f"   Batch size:    {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Training set:  {len(train_tokenized)} examples")
    print()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,                        # Prevents overfitting
        eval_strategy="epoch",                    # Test after each epoch
        save_strategy="epoch",                    # Save after each epoch
        load_best_model_at_end=True,              # Keep the best version
        metric_for_best_model="f1",               # "Best" = highest F1 score
        logging_steps=50,                          # Print progress every 50 steps
        save_total_limit=2,                        # Only keep 2 best checkpoints
        report_to="none",                          # Don't send to any tracking service
        use_cpu=False,                             # Use GPU (MPS on Apple Silicon) for faster training!
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )

    # ---- START TRAINING! ----
    print("🚀 Training started! Sit back and watch the magic...\n")
    trainer.train()

    # ---- Evaluate FINE-TUNED model (after training) ----
    print("\n" + "=" * 60)
    print("  📊 STEP 5: Evaluating YOUR fine-tuned model")
    print("=" * 60)

    finetuned_results = trainer.evaluate()
    finetuned_accuracy = finetuned_results["eval_accuracy"]
    finetuned_f1 = finetuned_results["eval_f1"]

    # ---- Compare! ----
    print("\n" + "=" * 60)
    print("  🏆 RESULTS: Base vs Fine-Tuned")
    print("=" * 60)
    print(f"\n   {'Metric':<15} {'Base FinBERT':<18} {'YOUR FinBERT':<18} {'Change'}")
    print(f"   {'─' * 15} {'─' * 18} {'─' * 18} {'─' * 10}")
    print(f"   {'Accuracy':<15} {base_accuracy:<18.2%} {finetuned_accuracy:<18.2%} {'+' if finetuned_accuracy > base_accuracy else ''}{(finetuned_accuracy - base_accuracy):.2%}")
    print(f"   {'F1 Score':<15} {base_f1:<18.4f} {finetuned_f1:<18.4f} {'+' if finetuned_f1 > base_f1 else ''}{(finetuned_f1 - base_f1):.4f}")

    # ---- Save the model ----
    print("\n" + "=" * 60)
    print("  💾 STEP 6: Saving YOUR model")
    print("=" * 60)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"   ✅ Model saved to: {OUTPUT_DIR}/")

    # ---- Save training report ----
    report = {
        "model_name": "Alpha Sentiment Engine — Fine-Tuned FinBERT",
        "base_model": MODEL_NAME,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "datasets": [
            "Financial PhraseBank",
            "Twitter Financial News",
            "Financial Tweets (StockTwits + others)",
            "Reddit Stock Sentiment (r/stocks, r/wallstreetbets)",
        ],
        "total_training_examples": len(train_tokenized),
        "total_test_examples": len(test_tokenized),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "base_model_accuracy": round(base_accuracy, 4),
        "base_model_f1": round(base_f1, 4),
        "finetuned_accuracy": round(finetuned_accuracy, 4),
        "finetuned_f1": round(finetuned_f1, 4),
        "accuracy_improvement": round(finetuned_accuracy - base_accuracy, 4),
    }

    report_path = os.path.join(OUTPUT_DIR, "training_report.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   📄 Training report saved to: {report_path}")

    # ---- Final message ----
    print("\n" + "=" * 60)
    print("  🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n   Your fine-tuned model is saved at: {OUTPUT_DIR}/")
    print(f"   Base accuracy:       {base_accuracy:.2%}")
    print(f"   Your accuracy:       {finetuned_accuracy:.2%}")
    print(f"   Improvement:         {'+' if finetuned_accuracy > base_accuracy else ''}{(finetuned_accuracy - base_accuracy):.2%}")
    print(f"\n   To use your model in the pipeline, update sentiment_service.py")
    print(f'   Change: MODEL_NAME = "ProsusAI/finbert"')
    print(f'   To:     MODEL_NAME = "{OUTPUT_DIR}"')
    print("=" * 60)


if __name__ == "__main__":
    train()
