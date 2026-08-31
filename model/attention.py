"""
Real Bahdanau (additive) attention.

    score(s_t, h_i)     = V^T * tanh(W1(h_i) + W2(s_t))
    attention_weights    = softmax_i(score(s_t, h_i))
    context_vector        = sum_i(attention_weights_i * h_i)

where h_i is the encoder output at input position i, and s_t is the
decoder's hidden state at output step t. There is no randomness in these
weights: they are a deterministic function of the current encoder outputs
and decoder hidden state, produced by the same W1/W2/V Dense layers that
are trained jointly with the rest of the model. Both the context vector
(fed to the decoder) and the attention weights (used for visualization)
are returned from a single call, so the weights displayed in the heatmap
are exactly the ones that drove that decoding step.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units: int):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)  # applied to encoder outputs
        self.W2 = tf.keras.layers.Dense(units)  # applied to decoder hidden state
        self.V = tf.keras.layers.Dense(1)

    def call(self, decoder_hidden_state, encoder_outputs):
        """
        decoder_hidden_state: (batch, decoder_units)
        encoder_outputs:      (batch, seq_len, encoder_units)

        Returns:
            context_vector:    (batch, encoder_units)
            attention_weights: (batch, seq_len) — sums to 1.0 across seq_len per example
        """
        hidden_with_time_axis = tf.expand_dims(decoder_hidden_state, axis=1)  # (batch, 1, dec_units)

        score = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(hidden_with_time_axis)))
        # score: (batch, seq_len, 1)

        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, seq_len, 1)

        context_vector = tf.reduce_sum(attention_weights * encoder_outputs, axis=1)  # (batch, enc_units)

        return context_vector, tf.squeeze(attention_weights, axis=-1)


def _self_test():
    batch_size, seq_len = 4, 10
    encoder_units, decoder_units, attention_units = 32, 24, 16

    attention = BahdanauAttention(attention_units)
    encoder_outputs = tf.random.normal((batch_size, seq_len, encoder_units))
    decoder_hidden = tf.random.normal((batch_size, decoder_units))

    context_vector, attention_weights = attention(decoder_hidden, encoder_outputs)

    assert context_vector.shape == (batch_size, encoder_units), context_vector.shape
    assert attention_weights.shape == (batch_size, seq_len), attention_weights.shape

    sums = tf.reduce_sum(attention_weights, axis=1).numpy()
    assert (abs(sums - 1.0) < 1e-4).all(), f"attention weights not normalized: {sums}"
    assert (attention_weights.numpy() >= 0).all(), "softmax output must be non-negative"

    # Different decoder hidden states must yield different attention distributions
    # (i.e. weights genuinely depend on the query, not a fixed/fake pattern).
    other_hidden = tf.random.normal((batch_size, decoder_units))
    _, other_weights = attention(other_hidden, encoder_outputs)
    assert not tf.reduce_all(tf.abs(attention_weights - other_weights) < 1e-6), (
        "attention weights did not change with a different decoder hidden state"
    )

    # Batch-size independence.
    other_batch_outputs = tf.random.normal((9, seq_len, encoder_units))
    other_batch_hidden = tf.random.normal((9, decoder_units))
    context2, weights2 = attention(other_batch_hidden, other_batch_outputs)
    assert context2.shape == (9, encoder_units)
    assert weights2.shape == (9, seq_len)

    print("[attention] self-test passed.")
    print(f"  context_vector: {context_vector.shape}, attention_weights: {attention_weights.shape}")
    print(f"  attention weight row sums (should all be ~1.0): {sums}")


if __name__ == "__main__":
    _self_test()
