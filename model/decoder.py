"""
LSTM Decoder with Bahdanau attention: Embedding -> concat(context) -> LSTM -> Dense -> Softmax.

Designed as a SINGLE-TIMESTEP cell: one call produces one output token's
probability distribution, given the previous token, the previous decoder
state, and the full encoder output sequence. Both training (teacher forcing,
looping over ground-truth target tokens) and inference (greedy/beam search,
looping over generated tokens) drive the decoder through this same call(),
so the attention weights captured at each step are always real — produced
during an actual forward pass, never fabricated after the fact.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf

import config
from model.attention import BahdanauAttention


class Decoder(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        lstm_units: int = config.LSTM_UNITS,
    ):
        super().__init__()
        self.lstm_units = lstm_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True
        )
        self.attention = BahdanauAttention(lstm_units)
        self.lstm = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=False,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, input_token, hidden_state, cell_state, encoder_outputs, training=False):
        """
        input_token:     (batch, 1) token id of the previous output step
        hidden_state:    (batch, lstm_units) decoder hidden state from the previous step
        cell_state:      (batch, lstm_units) decoder cell state from the previous step
        encoder_outputs: (batch, input_seq_len, lstm_units)

        Returns:
            output_probs:      (batch, vocab_size) — softmax distribution over the vocabulary
            hidden_state:      (batch, lstm_units) — updated decoder hidden state
            cell_state:        (batch, lstm_units) — updated decoder cell state
            attention_weights: (batch, input_seq_len) — real weights from this step
        """
        context_vector, attention_weights = self.attention(hidden_state, encoder_outputs)

        x = self.embedding(input_token)  # (batch, 1, embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=-1)

        lstm_output, hidden_state, cell_state = self.lstm(
            x, initial_state=[hidden_state, cell_state], training=training
        )
        output_probs = self.fc(lstm_output)

        return output_probs, hidden_state, cell_state, attention_weights


def _self_test():
    batch_size, input_seq_len = 4, 10
    vocab_size, embedding_dim, lstm_units = 50, 16, 32

    decoder = Decoder(vocab_size, embedding_dim, lstm_units)

    encoder_outputs = tf.random.normal((batch_size, input_seq_len, lstm_units))
    hidden_state = tf.zeros((batch_size, lstm_units))
    cell_state = tf.zeros((batch_size, lstm_units))
    input_token = tf.random.uniform((batch_size, 1), maxval=vocab_size, dtype=tf.int32)

    probs, new_hidden, new_cell, attention_weights = decoder(
        input_token, hidden_state, cell_state, encoder_outputs
    )

    assert probs.shape == (batch_size, vocab_size), probs.shape
    assert new_hidden.shape == (batch_size, lstm_units), new_hidden.shape
    assert new_cell.shape == (batch_size, lstm_units), new_cell.shape
    assert attention_weights.shape == (batch_size, input_seq_len), attention_weights.shape

    prob_sums = tf.reduce_sum(probs, axis=1).numpy()
    assert (abs(prob_sums - 1.0) < 1e-3).all(), f"softmax output not normalized: {prob_sums}"
    assert (probs.numpy() >= 0).all()

    attn_sums = tf.reduce_sum(attention_weights, axis=1).numpy()
    assert (abs(attn_sums - 1.0) < 1e-4).all(), f"attention weights not normalized: {attn_sums}"

    # Multi-step autoregressive loop must be able to chain state correctly.
    h, c = hidden_state, cell_state
    token = input_token
    attention_trace = []
    for _ in range(5):
        probs, h, c, attn = decoder(token, h, c, encoder_outputs)
        attention_trace.append(attn.numpy())
        token = tf.argmax(probs, axis=-1, output_type=tf.int32)[:, tf.newaxis]
    assert token.shape == (batch_size, 1)
    # Attention distribution should differ across steps as hidden state evolves.
    assert not (attention_trace[0] == attention_trace[-1]).all()

    # Batch-size independence.
    other_outputs = tf.random.normal((6, input_seq_len, lstm_units))
    other_h = tf.zeros((6, lstm_units))
    other_c = tf.zeros((6, lstm_units))
    other_token = tf.zeros((6, 1), dtype=tf.int32)
    probs2, h2, c2, attn2 = decoder(other_token, other_h, other_c, other_outputs)
    assert probs2.shape == (6, vocab_size)
    assert attn2.shape == (6, input_seq_len)

    print("[decoder] self-test passed.")
    print(f"  output_probs: {probs.shape}, hidden_state: {new_hidden.shape}, attention_weights: {attention_weights.shape}")
    print(f"  softmax row sums (should all be ~1.0): {prob_sums}")
    print(f"  5-step autoregressive chaining verified; attention shifts across steps")


if __name__ == "__main__":
    _self_test()
