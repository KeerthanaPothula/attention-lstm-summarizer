"""
Single, shared text-preprocessing pipeline.

This module is imported by dataset preparation, training, evaluation, AND
the Streamlit app's inference path. There is no separate preprocessing logic
for uploaded PDFs/images/text — everything funnels through clean_text() and
TextPreprocessor before it reaches the model.
"""

import sys
import re
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config

# Keras' default Tokenizer filter strips '<' and '>', which would turn our
# "<start>"/"<end>" sentinel tokens into the plain words "start"/"end" and
# make them indistinguishable from those words occurring naturally in text.
# We remove '<' and '>' from the filter set so the sentinels stay intact.
_TOKENIZER_FILTERS = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'

_URL_RE = re.compile(r"http\S+|www\.\S+")
_ALLOWED_CHARS_RE = re.compile(r"[^a-z0-9\s\.,!?']")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/newlines/unsupported characters, collapse whitespace."""
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _ALLOWED_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def add_start_end(text: str) -> str:
    """Wrap a (already cleaned) target string with sentinel tokens."""
    return f"{config.START_TOKEN} {text} {config.END_TOKEN}"


class TextPreprocessor:
    """Fits one shared vocabulary over inputs+targets and encodes/decodes sequences."""

    def __init__(
        self,
        vocab_size: int = config.VOCAB_SIZE,
        max_input_length: int = config.MAX_INPUT_LENGTH,
        max_target_length: int = config.MAX_TARGET_LENGTH,
    ):
        self.vocab_size = vocab_size
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.tokenizer = Tokenizer(
            num_words=vocab_size,
            oov_token=config.OOV_TOKEN,
            filters=_TOKENIZER_FILTERS,
        )
        self._fitted = False

    def fit(self, input_texts, target_texts):
        """Fit the vocabulary. Call this ONLY on the training split's text."""
        cleaned_inputs = [clean_text(t) for t in input_texts]
        cleaned_targets = [add_start_end(clean_text(t)) for t in target_texts]
        self.tokenizer.fit_on_texts(cleaned_inputs + cleaned_targets)
        self._fitted = True
        return self

    @property
    def actual_vocab_size(self) -> int:
        """Vocabulary size actually in effect (bounded by num_words)."""
        return min(self.vocab_size, len(self.tokenizer.word_index) + 1)

    def encode_inputs(self, texts):
        cleaned = [clean_text(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned)
        return pad_sequences(
            sequences, maxlen=self.max_input_length, padding="post", truncating="post"
        )

    def encode_targets(self, texts):
        cleaned = [add_start_end(clean_text(t)) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned)
        return pad_sequences(
            sequences, maxlen=self.max_target_length, padding="post", truncating="post"
        )

    def decode_sequence(self, sequence) -> str:
        """Convert a sequence of token ids back to a string, stopping at <end>/pad."""
        index_word = self.tokenizer.index_word
        words = []
        for token_id in sequence:
            token_id = int(token_id)
            if token_id == 0:
                continue
            word = index_word.get(token_id, config.OOV_TOKEN)
            if word == config.END_TOKEN:
                break
            if word == config.START_TOKEN:
                continue
            words.append(word)
        return " ".join(words)

    def save(self, path=config.TOKENIZER_PATH):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "tokenizer": self.tokenizer,
                    "vocab_size": self.vocab_size,
                    "max_input_length": self.max_input_length,
                    "max_target_length": self.max_target_length,
                },
                f,
            )

    @classmethod
    def load(cls, path=config.TOKENIZER_PATH):
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            vocab_size=state["vocab_size"],
            max_input_length=state["max_input_length"],
            max_target_length=state["max_target_length"],
        )
        obj.tokenizer = state["tokenizer"]
        obj._fitted = True
        return obj


def _self_test():
    train_articles = [
        "The quick brown fox jumps over the lazy dog. It happened near the river!",
        "Scientists announced a new discovery about climate change today.",
    ]
    train_summaries = [
        "Fox jumps over dog.",
        "New climate discovery announced.",
    ]

    pp = TextPreprocessor(vocab_size=50, max_input_length=12, max_target_length=8)
    pp.fit(train_articles, train_summaries)

    assert pp._fitted
    assert config.START_TOKEN in pp.tokenizer.word_index
    assert config.END_TOKEN in pp.tokenizer.word_index

    enc_in = pp.encode_inputs(train_articles)
    enc_out = pp.encode_targets(train_summaries)
    assert enc_in.shape == (2, 12)
    assert enc_out.shape == (2, 8)

    decoded = pp.decode_sequence(enc_out[0])
    assert "fox" in decoded and "dog" in decoded, decoded
    assert config.START_TOKEN not in decoded and config.END_TOKEN not in decoded

    # save/load round trip
    tmp_path = config.DATA_DIR / "_tokenizer_selftest.pkl"
    pp.save(tmp_path)
    reloaded = TextPreprocessor.load(tmp_path)
    assert reloaded.decode_sequence(enc_out[0]) == decoded
    tmp_path.unlink()

    print("[text_preprocessor] self-test passed.")
    print(f"  actual_vocab_size = {pp.actual_vocab_size}")
    print(f"  sample decoded target: '{decoded}'")


if __name__ == "__main__":
    _self_test()
