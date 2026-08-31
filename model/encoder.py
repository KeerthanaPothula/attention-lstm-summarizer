"""
LSTM Encoder: Embedding -> LSTM.

Returns the COMPLETE sequence of encoder hidden states (not just the final
one) because Bahdanau attention (model/attention.py) needs to attend over
every input position, plus the final hidden/cell state used to seed the
decoder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf

import config


class Encoder(tf.keras.Model):
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
        self.lstm = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, input_sequence, training=False):
        """
        input_sequence: (batch, seq_len) token ids

        Returns:
            encoder_outputs: (batch, seq_len, lstm_units) — full hidden-state sequence
            hidden_state:    (batch, lstm_units)
            cell_state:      (batch, lstm_units)
        """
        x = self.embedding(input_sequence)
        encoder_outputs, hidden_state, cell_state = self.lstm(x, training=training)
        return encoder_outputs, hidden_state, cell_state


def _self_test():
    batch_size, seq_len = 4, 10
    vocab_size, embedding_dim, lstm_units = 50, 16, 32

    encoder = Encoder(vocab_size, embedding_dim, lstm_units)
    dummy_input = tf.random.uniform((batch_size, seq_len), maxval=vocab_size, dtype=tf.int32)

    encoder_outputs, hidden_state, cell_state = encoder(dummy_input)

    assert encoder_outputs.shape == (batch_size, seq_len, lstm_units), encoder_outputs.shape
    assert hidden_state.shape == (batch_size, lstm_units), hidden_state.shape
    assert cell_state.shape == (batch_size, lstm_units), cell_state.shape

    # A second, differently-sized batch must work too (no hardcoded batch dim).
    other_batch = tf.random.uniform((7, seq_len), maxval=vocab_size, dtype=tf.int32)
    outputs2, h2, c2 = encoder(other_batch)
    assert outputs2.shape == (7, seq_len, lstm_units)
    assert h2.shape == (7, lstm_units)

    print("[encoder] self-test passed.")
    print(f"  encoder_outputs: {encoder_outputs.shape}")
    print(f"  hidden_state:    {hidden_state.shape}")
    print(f"  cell_state:      {cell_state.shape}")
    print(f"  batch-size independence verified (batch=4 and batch=7 both work)")


if __name__ == "__main__":
    _self_test()
