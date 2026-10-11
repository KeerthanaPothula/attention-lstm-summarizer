"""
Central configuration for the Attention-Augmented LSTM Summarizer project.

Every other module reads its constants from here. Do not redefine any of
these values locally in another file.

Dataset split policy (see README.md "Dataset" section for full reasoning):
CNN/DailyMail ships with an official, fixed train/validation/test split
(via tensorflow-datasets). We use that official split directly instead of
re-pooling and re-splitting at a custom ratio (e.g. 80/10/10), because
published ROUGE baselines are only comparable when evaluated on the same
official test split. USE_SUBSET selects how many examples to sample from
*within* each official split for fast development; it never mixes examples
across splits.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_documents"

TOKENIZER_PATH = DATA_DIR / "tokenizer.pkl"
DATASET_STATS_PATH = DATA_DIR / "dataset_stats.json"
MODEL_WEIGHTS_PATH = MODELS_DIR / "summarizer.weights.h5"
LATEST_WEIGHTS_PATH = MODELS_DIR / "summarizer_latest.weights.h5"
TRAINING_PROGRESS_PATH = MODELS_DIR / "training_progress.json"
CONFIG_SNAPSHOT_PATH = MODELS_DIR / "config_snapshot.json"
TRAINING_HISTORY_PATH = METRICS_DIR / "training_history.csv"

for _dir in (
    DATA_DIR, DATA_DIR / "train", DATA_DIR / "validation", DATA_DIR / "test",
    MODELS_DIR, RESULTS_DIR, PLOTS_DIR, METRICS_DIR, SUMMARIES_DIR,
    SAMPLE_DOCS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Loaded via the Hugging Face `datasets` library. "abisee/cnn_dailymail" is
# the canonical CNN/DailyMail repo (named for the dataset's original author);
# tensorflow-datasets was evaluated first but dropped due to a broken
# `promise` transitive dependency in its download pipeline.
DATASET_NAME = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"
DATASET_VERSION = "3.0.0"
DATASET_SPLIT = "official"  # fixed train/validation/test split shipped with the dataset

# Subset mode: sample this many examples from *within* each official split.
# Set USE_SUBSET = False to train/evaluate on the full official splits.
USE_SUBSET = True
TRAIN_SAMPLES = 10000
VAL_SAMPLES = 1000
TEST_SAMPLES = 1000

# ---------------------------------------------------------------------------
# Vocabulary / sequence lengths
# ---------------------------------------------------------------------------
VOCAB_SIZE = 20000
MAX_INPUT_LENGTH = 300   # article tokens
MAX_TARGET_LENGTH = 60   # summary tokens (includes <start>/<end>)

START_TOKEN = "<start>"
END_TOKEN = "<end>"
OOV_TOKEN = "<unk>"
PAD_TOKEN = ""  # Keras pad_sequences pads with 0, which maps to this empty string

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 128
LSTM_UNITS = 256

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 3

# ---------------------------------------------------------------------------
# Inference / decoding
# ---------------------------------------------------------------------------
BEAM_WIDTH = 3
# Default decoding constraints for inference/summarizer.py.
# Set SUPPRESS_UNK = False and NO_REPEAT_NGRAM_SIZE = 0 for plain greedy decoding.
SUPPRESS_UNK = True
NO_REPEAT_NGRAM_SIZE = 3

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
