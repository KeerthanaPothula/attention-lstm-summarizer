"""Plot Epoch vs Training/Validation Loss from an ACTUAL training history."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # headless-safe (no display needed, incl. inside Streamlit)
import matplotlib.pyplot as plt

import config


def plot_loss_curves(
    history: dict,
    output_path=config.PLOTS_DIR / "training_validation_loss.png",
    title: str = "Training vs Validation Loss",
) -> Path:
    """
    history must contain real, measured values under keys 'epoch',
    'train_loss', 'val_loss' (equal-length lists) — produced by
    training/train.py, never hand-entered numbers.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Training Loss")
    ax.plot(history["epoch"], history["val_loss"], marker="o", label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked Cross-Entropy Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _self_test():
    # Synthetic numbers here only to verify the plotting mechanism works end
    # to end (file gets written, non-empty) — never reported as a real
    # training result.
    dummy_history = {"epoch": [1, 2, 3], "train_loss": [2.5, 2.0, 1.7], "val_loss": [2.6, 2.2, 2.1]}
    out_path = config.PLOTS_DIR / "_loss_curves_selftest.png"

    result_path = plot_loss_curves(dummy_history, out_path)

    assert result_path.exists()
    assert result_path.stat().st_size > 0
    result_path.unlink()

    print("[loss_curves] self-test passed.")


if __name__ == "__main__":
    _self_test()
