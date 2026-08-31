"""
Complete trainable Attention-Augmented LSTM Encoder-Decoder.

Wires together model/encoder.py (Encoder), model/decoder.py (Decoder, which
already owns a BahdanauAttention instance internally — model/attention.py is
not re-instantiated or reimplemented here). No Transformer component is used
anywhere in this file.

Training-time flow (teacher forcing):
    encoder_input  -> Encoder -> encoder_outputs, hidden_state, cell_state
    for each decoder timestep t (one real ground-truth token at a time):
        Decoder(decoder_input[:, t], hidden_state, cell_state, encoder_outputs)
            -> probs_t, hidden_state, cell_state, attention_weights_t
    output_probs      = stack(probs_t)               over t -> (batch, target_len, vocab_size)
    attention_weights = stack(attention_weights_t)    over t -> (batch, target_len, input_len)

The decoder is driven one token at a time in both branches of the project:
here (teacher forcing) and later in inference/beam_search.py and
inference/summarizer.py (autoregressive generation). That is what guarantees
the attention weights are always real values produced by an actual forward
pass through the trained decoder/attention layers — never fabricated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf

import config
from model.encoder import Encoder
from model.decoder import Decoder


class AttentionLSTMSummarizer(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        lstm_units: int = config.LSTM_UNITS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.lstm_units = lstm_units
        self.encoder = Encoder(vocab_size, embedding_dim, lstm_units)
        self.decoder = Decoder(vocab_size, embedding_dim, lstm_units)

    def call(self, inputs, training=False):
        """
        inputs: (encoder_input, decoder_input)
            encoder_input: (batch, input_seq_len) token ids
            decoder_input: (batch, target_len) token ids, teacher-forced
                            (already shifted right — see training/train.py,
                            which slices target_sequences[:, :-1] for this).

        Returns:
            output_probs:      (batch, target_len, vocab_size) softmax probabilities
            attention_weights: (batch, target_len, input_seq_len) — one real
                                attention distribution per decoder timestep
        """
        encoder_input, decoder_input = inputs
        encoder_outputs, hidden_state, cell_state = self.encoder(encoder_input, training=training)

        target_len = decoder_input.shape[1]
        probs_per_step = []
        attention_per_step = []

        for t in range(target_len):
            token = decoder_input[:, t : t + 1]
            probs, hidden_state, cell_state, attention_weights = self.decoder(
                token, hidden_state, cell_state, encoder_outputs, training=training
            )
            probs_per_step.append(probs)
            attention_per_step.append(attention_weights)

        output_probs = tf.stack(probs_per_step, axis=1)
        attention_weights = tf.stack(attention_per_step, axis=1)
        return output_probs, attention_weights


def masked_loss(y_true, y_pred):
    """
    SparseCategoricalCrossentropy that ignores padding positions (token id 0),
    so the decoder is not penalized/rewarded for predicting past <end>.

    y_true: (batch, target_len) token ids
    y_pred: (batch, target_len, vocab_size) softmax probabilities
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction="none"
    )
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    per_token_loss = loss_object(y_true, y_pred) * mask
    return tf.reduce_sum(per_token_loss) / tf.reduce_sum(mask)


def build_model(
    vocab_size: int,
    embedding_dim: int = config.EMBEDDING_DIM,
    lstm_units: int = config.LSTM_UNITS,
    max_input_length: int = config.MAX_INPUT_LENGTH,
) -> AttentionLSTMSummarizer:
    """
    Factory that both constructs the architecture AND forces weight creation
    (via one dummy forward pass), so the returned model's variables always
    exist with shapes matching config — required before model.load_weights()
    can be called reliably on a freshly-constructed instance.
    """
    model = AttentionLSTMSummarizer(vocab_size, embedding_dim, lstm_units)
    dummy_encoder_input = tf.zeros((1, max_input_length), dtype=tf.int32)
    dummy_decoder_input = tf.zeros((1, 2), dtype=tf.int32)
    model((dummy_encoder_input, dummy_decoder_input))
    return model


def _self_test():
    vocab_size = 30
    embedding_dim, lstm_units = 16, 32
    batch_size, input_len, target_len = 4, 12, 7

    model = build_model(vocab_size, embedding_dim, lstm_units, max_input_length=input_len)

    encoder_input = tf.random.uniform((batch_size, input_len), maxval=vocab_size, dtype=tf.int32)
    decoder_input = tf.random.uniform((batch_size, target_len), maxval=vocab_size, dtype=tf.int32)
    decoder_target = tf.random.uniform((batch_size, target_len), minval=1, maxval=vocab_size, dtype=tf.int32)

    output_probs, attention_weights = model((encoder_input, decoder_input), training=True)

    # 4. Output tensor dimensions.
    assert output_probs.shape == (batch_size, target_len, vocab_size), output_probs.shape
    # 5. Attention weights returned for every decoder timestep.
    assert attention_weights.shape == (batch_size, target_len, input_len), attention_weights.shape
    # 6. Attention weights normalized (sum to 1 over input positions, at every timestep).
    attn_sums = tf.reduce_sum(attention_weights, axis=-1).numpy()
    assert (abs(attn_sums - 1.0) < 1e-4).all(), f"attention weights not normalized: {attn_sums}"
    # softmax outputs are also a valid probability distribution over the vocabulary
    prob_sums = tf.reduce_sum(output_probs, axis=-1).numpy()
    assert (abs(prob_sums - 1.0) < 1e-3).all(), f"output probs not normalized: {prob_sums}"

    # 7. Gradients can be calculated through the model (real masked loss, not a stub).
    with tf.GradientTape() as tape:
        probs, _ = model((encoder_input, decoder_input), training=True)
        loss = masked_loss(decoder_target, probs)
    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in gradients), "some trainable variables received no gradient"
    assert any(tf.reduce_any(tf.not_equal(g, 0)).numpy() for g in gradients), "all gradients were zero"

    # 8. Model has trainable parameters.
    assert len(model.trainable_variables) > 0
    param_count = sum(int(tf.size(v)) for v in model.trainable_variables)
    assert param_count > 0

    print("[summarizer] self-test passed.")
    print(f"  output_probs shape:      {output_probs.shape}")
    print(f"  attention_weights shape: {attention_weights.shape}")
    print(f"  masked_loss value:       {float(loss):.4f}")
    print(f"  trainable variables:     {len(model.trainable_variables)}")
    print(f"  trainable parameters:    {param_count:,}")


if __name__ == "__main__":
    _self_test()
