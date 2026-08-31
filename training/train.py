"""
Training pipeline for the Attention-Augmented LSTM Summarizer.

Consumes ONLY the pre-tokenized train/validation arrays written by
data/prepare_dataset.py to data/train/ and data/validation/. The test split
(data/test/) is never read here — training/evaluate.py is the only script
that touches it, and only after training has finished.

Why a manual GradientTape loop instead of model.fit()
-------------------------------------------------------
AttentionLSTMSummarizer.call() drives the decoder one real token at a time
and returns (output_probs, attention_weights) — a tuple, with the second
element existing purely for visualization, not for computing loss. Making
that shape fit model.fit()'s single-output/loss-dict contract would require
overriding train_step/test_step on the approved model class for no benefit;
a plain train/validation loop is simpler, keeps model/summarizer.py
untouched, and still gets real masked loss, real gradients, checkpointing,
and early stopping. Each step is wrapped in tf.function so the unrolled
per-timestep decoder loop is compiled once per batch shape instead of
re-interpreted in Python every step — this is a performance-only change
(identical math), needed because CNN/DailyMail-scale training is otherwise
too slow in pure eager mode.
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from preprocessing.text_preprocessor import TextPreprocessor
from model.summarizer import build_model, masked_loss
from visualization.loss_curves import plot_loss_curves


def load_split_arrays(split_dir: Path):
    encoder_input = np.load(split_dir / "encoder_input.npy")
    target_sequences = np.load(split_dir / "target_sequences.npy")
    return encoder_input, target_sequences


def make_dataset(encoder_input, target_sequences, batch_size: int, shuffle: bool):
    # Teacher forcing shift: decoder sees tokens[:-1] as input, must predict
    # tokens[1:] — the standard next-token objective for seq2seq training.
    decoder_input = target_sequences[:, :-1]
    decoder_target = target_sequences[:, 1:]
    ds = tf.data.Dataset.from_tensor_slices((encoder_input, decoder_input, decoder_target))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(encoder_input), seed=config.RANDOM_SEED)
    return ds.batch(batch_size)


def train_model(
    train_encoder_input,
    train_target_sequences,
    val_encoder_input,
    val_target_sequences,
    vocab_size: int,
    epochs: int,
    batch_size: int,
    weights_path: Path,
    patience: int,
    latest_weights_path: Path = None,
    progress_path: Path = None,
):
    """
    latest_weights_path / progress_path (optional): if given, progress is
    checkpointed after EVERY epoch (not just on improvement) so a run that
    gets killed mid-training (process/host interruption, not a training
    failure) can be resumed from the last completed epoch instead of
    restarting from scratch. Real, unbuffered progress is printed so an
    interrupted run's log shows exactly how far it got.
    """
    model = build_model(vocab_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    train_ds = make_dataset(train_encoder_input, train_target_sequences, batch_size, shuffle=True)
    val_ds = make_dataset(val_encoder_input, val_target_sequences, batch_size, shuffle=False)

    # jit_compile=True (XLA) is a pure execution-speed change: same model,
    # same math, same masked_loss, same teacher forcing. No architecture,
    # config, or methodology is altered by compiling these two steps.
    @tf.function(jit_compile=True)
    def train_step(enc_in, dec_in, dec_target):
        with tf.GradientTape() as tape:
            probs, _ = model((enc_in, dec_in), training=True)
            loss = masked_loss(dec_target, probs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function(jit_compile=True)
    def val_step(enc_in, dec_in, dec_target):
        probs, _ = model((enc_in, dec_in), training=False)
        return masked_loss(dec_target, probs)

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1
    prior_training_time_seconds = 0.0
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_path and latest_weights_path and Path(progress_path).exists() and Path(latest_weights_path).exists():
        with open(progress_path) as f:
            saved = json.load(f)
        history = saved["history"]
        best_val_loss = saved["best_val_loss"]
        patience_counter = saved["patience_counter"]
        start_epoch = saved["next_epoch"]
        prior_training_time_seconds = saved.get("cumulative_training_time_seconds", 0.0)
        model.load_weights(str(latest_weights_path))
        print(f"Resuming from epoch {start_epoch} "
              f"(best_val_loss so far: {best_val_loss:.4f})", flush=True)

    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        train_losses = [float(train_step(enc_in, dec_in, dec_target))
                         for enc_in, dec_in, dec_target in train_ds]
        train_loss = float(np.mean(train_losses))

        val_losses = [float(val_step(enc_in, dec_in, dec_target))
                       for enc_in, dec_in, dec_target in val_ds]
        val_loss = float(np.mean(val_losses))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} - "
              f"val_loss: {val_loss:.4f} - {epoch_time:.1f}s", flush=True)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_weights(str(weights_path))
            print(f"  val_loss improved -> saved best weights to {weights_path}", flush=True)
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{patience})", flush=True)

        if latest_weights_path:
            model.save_weights(str(latest_weights_path))
        if progress_path:
            with open(progress_path, "w") as f:
                json.dump({
                    "history": history,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "next_epoch": epoch + 1,
                    "cumulative_training_time_seconds": prior_training_time_seconds + (time.time() - start_time),
                }, f)

        if patience_counter >= patience:
            print("  early stopping triggered.", flush=True)
            break

    training_time_seconds = prior_training_time_seconds + (time.time() - start_time)
    trainable_parameters = sum(int(tf.size(v)) for v in model.trainable_variables)

    run_info = {
        "epochs_requested": epochs,
        "epochs_completed": len(history["epoch"]),
        "best_validation_loss": best_val_loss,
        "training_time_seconds": round(training_time_seconds, 2),
        "trainable_parameters": trainable_parameters,
    }
    return model, history, run_info


def verify_reload(weights_path: Path, vocab_size: int) -> bool:
    """Load saved weights into a FRESH build_model() instance and confirm a
    forward pass runs with the expected output shapes."""
    fresh_model = build_model(vocab_size)
    fresh_model.load_weights(str(weights_path))

    dummy_encoder_input = tf.zeros((2, config.MAX_INPUT_LENGTH), dtype=tf.int32)
    dummy_decoder_input = tf.zeros((2, 5), dtype=tf.int32)
    probs, attention_weights = fresh_model((dummy_encoder_input, dummy_decoder_input))

    assert probs.shape == (2, 5, vocab_size), probs.shape
    assert attention_weights.shape == (2, 5, config.MAX_INPUT_LENGTH), attention_weights.shape
    return True


def save_config_snapshot(path: Path, run_info: dict, vocab_size: int):
    with open(config.DATASET_STATS_PATH) as f:
        dataset_stats = json.load(f)

    snapshot = {
        "dataset_mode": dataset_stats["mode"],
        "train_samples": dataset_stats["train_samples"],
        "validation_samples": dataset_stats["validation_samples"],
        "test_samples": dataset_stats["test_samples"],
        "vocab_size": vocab_size,
        "max_input_length": config.MAX_INPUT_LENGTH,
        "max_target_length": config.MAX_TARGET_LENGTH,
        "embedding_dim": config.EMBEDDING_DIM,
        "lstm_units": config.LSTM_UNITS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "optimizer": "Adam",
        "random_seed": config.RANDOM_SEED,
        **run_info,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot


def _print_summary(dataset_stats, vocab_size, run_info):
    print("\n=== Training summary ===")
    print(f"Dataset mode: {dataset_stats['mode']}")
    print(f"Training samples: {dataset_stats['train_samples']}")
    print(f"Validation samples: {dataset_stats['validation_samples']}")
    print(f"Test samples (informational only, NOT used): {dataset_stats['test_samples']}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Maximum input length: {config.MAX_INPUT_LENGTH}")
    print(f"Maximum target length: {config.MAX_TARGET_LENGTH}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs requested: {run_info['epochs_requested']}")
    print(f"Epochs completed: {run_info['epochs_completed']}")
    print(f"Trainable parameters: {run_info['trainable_parameters']:,}")
    print(f"Training time: {run_info['training_time_seconds']:.1f}s")
    print(f"Best validation loss: {run_info['best_validation_loss']:.4f}")


def run_subset_training():
    with open(config.DATASET_STATS_PATH) as f:
        dataset_stats = json.load(f)

    preprocessor = TextPreprocessor.load()
    vocab_size = preprocessor.actual_vocab_size

    train_encoder_input, train_target_sequences = load_split_arrays(config.DATA_DIR / "train")
    val_encoder_input, val_target_sequences = load_split_arrays(config.DATA_DIR / "validation")

    print(f"Loaded train arrays: {train_encoder_input.shape}, {train_target_sequences.shape}")
    print(f"Loaded validation arrays: {val_encoder_input.shape}, {val_target_sequences.shape}")

    model, history, run_info = train_model(
        train_encoder_input, train_target_sequences,
        val_encoder_input, val_target_sequences,
        vocab_size=vocab_size,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        weights_path=config.MODEL_WEIGHTS_PATH,
        patience=config.EARLY_STOPPING_PATIENCE,
        latest_weights_path=config.LATEST_WEIGHTS_PATH,
        progress_path=config.TRAINING_PROGRESS_PATH,
    )

    pd.DataFrame(history).to_csv(config.TRAINING_HISTORY_PATH, index=False)
    plot_loss_curves(history, config.PLOTS_DIR / "training_validation_loss.png")
    save_config_snapshot(config.CONFIG_SNAPSHOT_PATH, run_info, vocab_size)

    reload_ok = verify_reload(config.MODEL_WEIGHTS_PATH, vocab_size)
    print(f"\nWeight reload verification: {'PASSED' if reload_ok else 'FAILED'}")

    _print_summary(dataset_stats, vocab_size, run_info)
    return history, run_info


def smoke_test(train_n: int = 50, val_n: int = 10, epochs: int = 2, batch_size: int = 8):
    print("=== SMOKE TEST (tiny data, verifies the full pipeline mechanically) ===")

    preprocessor = TextPreprocessor.load()
    vocab_size = preprocessor.actual_vocab_size

    train_encoder_input, train_target_sequences = load_split_arrays(config.DATA_DIR / "train")
    val_encoder_input, val_target_sequences = load_split_arrays(config.DATA_DIR / "validation")

    train_encoder_input = train_encoder_input[:train_n]
    train_target_sequences = train_target_sequences[:train_n]
    val_encoder_input = val_encoder_input[:val_n]
    val_target_sequences = val_target_sequences[:val_n]

    print(f"Smoke-test train samples: {len(train_encoder_input)}")
    print(f"Smoke-test validation samples: {len(val_encoder_input)}")

    smoke_weights_path = config.MODELS_DIR / "smoke_test.weights.h5"
    model, history, run_info = train_model(
        train_encoder_input, train_target_sequences,
        val_encoder_input, val_target_sequences,
        vocab_size=vocab_size,
        epochs=epochs,
        batch_size=batch_size,
        weights_path=smoke_weights_path,
        patience=epochs,  # do not early-stop during the smoke test
    )

    pd.DataFrame(history).to_csv(config.METRICS_DIR / "smoke_test_history.csv", index=False)
    plot_loss_curves(
        history,
        config.PLOTS_DIR / "smoke_test_loss.png",
        title="Smoke Test: Training vs Validation Loss",
    )
    snapshot = save_config_snapshot(
        config.MODELS_DIR / "smoke_test_config_snapshot.json", run_info, vocab_size
    )
    snapshot["dataset_mode"] = f"SMOKE TEST (train={train_n}, val={val_n})"

    reload_ok = verify_reload(smoke_weights_path, vocab_size)

    print("\n=== Smoke test results ===")
    print(f"Epochs completed: {run_info['epochs_completed']}/{run_info['epochs_requested']}")
    print(f"Trainable parameters: {run_info['trainable_parameters']:,}")
    print(f"Training time: {run_info['training_time_seconds']:.1f}s")
    print(f"Best validation loss: {run_info['best_validation_loss']:.4f}")
    print(f"Weight reload verification: {'PASSED' if reload_ok else 'FAILED'}")
    print(f"Saved weights: {smoke_weights_path}")
    print(f"Saved history: {config.METRICS_DIR / 'smoke_test_history.csv'}")
    print(f"Saved plot: {config.PLOTS_DIR / 'smoke_test_loss.png'}")

    assert reload_ok
    assert run_info["epochs_completed"] == epochs
    assert smoke_weights_path.exists()
    print("\nSMOKE TEST PASSED.")
    return history, run_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny end-to-end pipeline check")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
    else:
        run_subset_training()
