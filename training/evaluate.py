"""
Test-set evaluation: decode data/test/ with the trained checkpoint and score
against the reference summaries with ROUGE-1/2/L (rouge-score, stemmed).

The only script that reads data/test/. Each decoding mode is evaluated and
saved separately so results stay comparable but are never mixed:

    greedy    plain argmax decoding
    improved  greedy + <unk> suppression + no-repeat trigram blocking

Outputs (per mode) in results/metrics/:
    test_metrics_<mode>.csv          mean precision/recall/F1 per ROUGE type
    generated_summaries_<mode>.csv   reference vs generated, per example

Usage: python training/evaluate.py [--limit N] [--batch-size B] [--weights PATH]
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

import config
from inference.summarizer import Summarizer

MODES = {
    "greedy": {"suppress_unk": False, "no_repeat_ngram": 0},
    "improved": {"suppress_unk": True, "no_repeat_ngram": 3},
}
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def evaluate(limit=None, batch_size=config.BATCH_SIZE, weights_path=config.MODEL_WEIGHTS_PATH):
    test_dir = config.DATA_DIR / "test"
    encoder_input = np.load(test_dir / "encoder_input.npy")
    with open(test_dir / "reference_summaries.json", encoding="utf-8") as f:
        references = json.load(f)
    if limit:
        encoder_input, references = encoder_input[:limit], references[:limit]

    summarizer = Summarizer(weights_path=weights_path)
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    print(f"Evaluating {len(references)} test examples with {weights_path}")

    results = {}
    for mode, decode_kwargs in MODES.items():
        t0 = time.time()
        generated = []
        for i in range(0, len(encoder_input), batch_size):
            ids = summarizer.generate_ids(encoder_input[i:i + batch_size], **decode_kwargs)
            generated += [summarizer.preprocessor.decode_sequence(seq) for seq in ids]
        elapsed = time.time() - t0

        scores = [scorer.score(ref, gen) for ref, gen in zip(references, generated)]
        rows = [
            {
                "mode": mode,
                "rouge_type": rt,
                "precision": float(np.mean([s[rt].precision for s in scores])),
                "recall": float(np.mean([s[rt].recall for s in scores])),
                "f1": float(np.mean([s[rt].fmeasure for s in scores])),
                "num_test_samples": len(references),
                "weights": str(weights_path),
                "decode_seconds": round(elapsed, 1),
            }
            for rt in ROUGE_TYPES
        ]
        pd.DataFrame(rows).to_csv(config.METRICS_DIR / f"test_metrics_{mode}.csv", index=False)
        pd.DataFrame({"reference": references, "generated": generated}).to_csv(
            config.METRICS_DIR / f"generated_summaries_{mode}.csv", index=False
        )
        results[mode] = rows
        print(f"\n[{mode}] {decode_kwargs} - {elapsed:.1f}s")
        for row in rows:
            print(f"  {row['rouge_type']:7s} P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N test examples")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--weights", type=Path, default=config.MODEL_WEIGHTS_PATH)
    args = parser.parse_args()
    evaluate(args.limit, args.batch_size, args.weights)
