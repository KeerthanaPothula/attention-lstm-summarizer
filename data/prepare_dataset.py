"""
Stage 3: Dataset loading, official-split handling, and tokenization.

Lives under data/ (rather than preprocessing/) because it is CNN/DailyMail-
specific acquisition + orchestration, not generic text cleaning — that logic
stays in preprocessing/text_preprocessor.py and is reused here.

Split policy
------------
CNN/DailyMail ships with an official, fixed train/validation/test split
(via tensorflow-datasets). We use that split as-is instead of re-pooling the
data into a custom 80/10/10 (or similar) ratio, because published ROUGE
baselines are only comparable when measured on the same official test split.

USE_SUBSET selects how many examples to sample from *within* each official
split for fast development (never mixing examples across splits). Sampling
uses config.RANDOM_SEED so the subset is reproducible. Full-dataset mode
(USE_SUBSET = False) uses every example in each official split.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from datasets import load_dataset

import config
from preprocessing.text_preprocessor import TextPreprocessor, clean_text


def _load_split(split_name: str, num_samples):
    """Load one OFFICIAL split by name; optionally sample num_samples from it."""
    ds = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split=split_name)
    if num_samples is not None:
        ds = ds.shuffle(seed=config.RANDOM_SEED).select(range(min(num_samples, len(ds))))
    return list(ds["article"]), list(ds["highlights"])


def prepare():
    t0 = time.time()
    mode = "EXPERIMENTAL SUBSET" if config.USE_SUBSET else "FULL DATASET"
    print(f"Dataset mode: {mode}")
    print(f"Dataset: {config.DATASET_NAME} v{config.DATASET_VERSION} "
          f"(official predefined train/validation/test split)")

    train_n = config.TRAIN_SAMPLES if config.USE_SUBSET else None
    val_n = config.VAL_SAMPLES if config.USE_SUBSET else None
    test_n = config.TEST_SAMPLES if config.USE_SUBSET else None

    print("Loading official 'train' split...")
    train_articles, train_summaries = _load_split("train", train_n)
    print("Loading official 'validation' split...")
    val_articles, val_summaries = _load_split("validation", val_n)
    print("Loading official 'test' split...")
    test_articles, test_summaries = _load_split("test", test_n)

    print(f"Fitting tokenizer on TRAIN split only ({len(train_articles)} examples)...")
    preprocessor = TextPreprocessor()
    preprocessor.fit(train_articles, train_summaries)
    preprocessor.save()

    splits = {
        "train": (train_articles, train_summaries, config.DATA_DIR / "train"),
        "validation": (val_articles, val_summaries, config.DATA_DIR / "validation"),
        "test": (test_articles, test_summaries, config.DATA_DIR / "test"),
    }

    for name, (articles, summaries, out_dir) in splits.items():
        encoder_input = preprocessor.encode_inputs(articles)
        target_sequences = preprocessor.encode_targets(summaries)
        np.save(out_dir / "encoder_input.npy", encoder_input)
        np.save(out_dir / "target_sequences.npy", target_sequences)
        # Cleaned (but not tokenized) reference text, used later for ROUGE —
        # scored on the same normalized text the model is trained to produce.
        cleaned_refs = [clean_text(s) for s in summaries]
        with open(out_dir / "reference_summaries.json", "w", encoding="utf-8") as f:
            json.dump(cleaned_refs, f)
        print(f"  {name}: {len(articles)} examples -> "
              f"encoder_input{encoder_input.shape}, target_sequences{target_sequences.shape}")

    elapsed = time.time() - t0

    stats = {
        "dataset_name": config.DATASET_NAME,
        "dataset_version": config.DATASET_VERSION,
        "dataset_split_policy": "official predefined train/validation/test split (no custom re-split)",
        "mode": mode,
        "train_samples": len(train_articles),
        "validation_samples": len(val_articles),
        "test_samples": len(test_articles),
        "total_samples": len(train_articles) + len(val_articles) + len(test_articles),
        "vocab_size_configured": config.VOCAB_SIZE,
        "vocab_size_actual": preprocessor.actual_vocab_size,
        "max_input_length": config.MAX_INPUT_LENGTH,
        "max_target_length": config.MAX_TARGET_LENGTH,
        "random_seed": config.RANDOM_SEED,
        "preparation_time_seconds": round(elapsed, 2),
    }

    with open(config.DATASET_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    pd.DataFrame([stats]).to_csv(config.METRICS_DIR / "dataset_statistics.csv", index=False)

    print("\n=== Dataset preparation complete ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return stats


if __name__ == "__main__":
    prepare()
