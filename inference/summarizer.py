"""
Autoregressive inference for the trained Attention-Augmented LSTM.

decode_ids() is the single greedy decoding loop used everywhere (evaluation,
and later the app). Two optional constraints are applied to each step's
probability distribution before argmax — the model itself is unchanged:

    suppress_unk     never emit the <unk> token
    no_repeat_ngram  n > 0 blocks any token that would repeat an n-gram
                     already present in the generated sequence (3 = trigram)

With suppress_unk=False and no_repeat_ngram=0 it is plain greedy decoding.
decode_ids() only needs a `step` callable, so it is testable without
TensorFlow weights (see tests/test_decoding.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

import config
from preprocessing.text_preprocessor import TextPreprocessor
from model.summarizer import build_model


def _banned_tokens(seq, n):
    """Tokens that would complete an n-gram already present in seq."""
    if n <= 0 or len(seq) < n - 1:
        return set()
    prefix = tuple(seq[len(seq) - (n - 1):])
    return {
        seq[i + n - 1]
        for i in range(len(seq) - n + 1)
        if tuple(seq[i:i + n - 1]) == prefix
    }


def decode_ids(step, batch_size, start_id, end_id, max_len,
               unk_id=None, suppress_unk=False, no_repeat_ngram=0):
    """
    step(prev_tokens: (batch,) int array) -> (batch, vocab) probabilities.
    The step callable owns the decoder state between calls.

    Returns one list of generated token ids per row, excluding <start>/<end>.
    A row stops at <end>; every row stops after max_len decoding steps.
    """
    if suppress_unk and unk_id is None:
        raise ValueError("suppress_unk=True requires unk_id")
    outputs = [[] for _ in range(batch_size)]
    done = np.zeros(batch_size, dtype=bool)
    prev = np.full(batch_size, start_id, dtype=np.int32)

    for _ in range(max_len):
        probs = np.array(step(prev), dtype=np.float64)  # copy: we mask in place
        if suppress_unk:
            probs[:, unk_id] = -np.inf
        for row in range(batch_size):
            if done[row]:
                continue
            for token in _banned_tokens(outputs[row], no_repeat_ngram):
                probs[row, token] = -np.inf
            token = int(np.argmax(probs[row]))
            if token == end_id:
                done[row] = True
            else:
                outputs[row].append(token)
            prev[row] = token
        if done.all():
            break
    return outputs


class Summarizer:
    """Loads the saved tokenizer + weights once; generates summaries."""

    def __init__(self, weights_path=config.MODEL_WEIGHTS_PATH, tokenizer_path=config.TOKENIZER_PATH):
        self.preprocessor = TextPreprocessor.load(tokenizer_path)
        self.model = build_model(self.preprocessor.actual_vocab_size)
        self.model.load_weights(str(weights_path))
        word_index = self.preprocessor.tokenizer.word_index
        self.start_id = word_index[config.START_TOKEN]
        self.end_id = word_index[config.END_TOKEN]
        self.unk_id = word_index[config.OOV_TOKEN]

    def generate_ids(self, encoder_input,
                     suppress_unk=config.SUPPRESS_UNK,
                     no_repeat_ngram=config.NO_REPEAT_NGRAM_SIZE,
                     max_len=config.MAX_TARGET_LENGTH - 1):
        """encoder_input: (batch, MAX_INPUT_LENGTH) ids from preprocessor.encode_inputs()."""
        encoder_outputs, hidden, cell = self.model.encoder(tf.constant(encoder_input, dtype=tf.int32))
        state = [hidden, cell]

        def step(prev):
            probs, state[0], state[1], _ = self.model.decoder(
                tf.constant(prev[:, None], dtype=tf.int32), state[0], state[1], encoder_outputs
            )
            return probs.numpy()

        return decode_ids(
            step, len(encoder_input), self.start_id, self.end_id, max_len,
            unk_id=self.unk_id, suppress_unk=suppress_unk, no_repeat_ngram=no_repeat_ngram,
        )

    def summarize(self, texts, **decode_kwargs):
        """Raw texts -> summaries, through the same preprocessing used in training."""
        ids = self.generate_ids(self.preprocessor.encode_inputs(texts), **decode_kwargs)
        return [self.preprocessor.decode_sequence(seq) for seq in ids]
