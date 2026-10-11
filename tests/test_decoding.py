"""
Unit tests for inference.summarizer.decode_ids, using stub step functions
(no trained weights needed), plus a cheap smoke test against the real
checkpoint that is skipped when the git-ignored weights/data are absent.
Run: python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
from inference.summarizer import Summarizer, decode_ids

VOCAB, START, END, UNK = 20, 2, 3, 1


def make_step(preferred):
    """Stub decoder: preferred(prev_token) is the argmax; other tokens get a
    fixed, distinct, lower probability so the runner-up is deterministic."""
    calls = []

    def step(prev):
        calls.append(prev.copy())
        probs = np.tile(np.linspace(0.01, 0.02, VOCAB), (len(prev), 1))
        for row, token in enumerate(prev):
            probs[row, preferred(int(token))] = 1.0
        return probs
    step.calls = calls
    return step


def trigrams(seq):
    return [tuple(seq[i:i + 3]) for i in range(len(seq) - 2)]


class DecodeIdsTest(unittest.TestCase):
    def test_unk_suppression(self):
        always_unk = lambda prev: UNK
        plain = decode_ids(make_step(always_unk), 1, START, END, 10, unk_id=UNK)[0]
        self.assertIn(UNK, plain)
        suppressed = decode_ids(make_step(always_unk), 1, START, END, 10,
                                unk_id=UNK, suppress_unk=True)[0]
        self.assertNotIn(UNK, suppressed)
        self.assertEqual(len(suppressed), 10)

    def test_suppress_unk_requires_unk_id(self):
        with self.assertRaises(ValueError):
            decode_ids(make_step(lambda p: 5), 1, START, END, 5, suppress_unk=True)

    def test_trigram_blocking(self):
        # Model wants to loop 5 -> 6 -> 7 -> 5 ... forever.
        cycle = {START: 5, 5: 6, 6: 7, 7: 5}
        preferred = lambda prev: cycle.get(prev, 5)
        plain = decode_ids(make_step(preferred), 1, START, END, 30)[0]
        self.assertLess(len(set(trigrams(plain))), len(trigrams(plain)))  # repeats without blocking

        blocked = decode_ids(make_step(preferred), 1, START, END, 30, no_repeat_ngram=3)[0]
        self.assertEqual(len(set(trigrams(blocked))), len(trigrams(blocked)))
        self.assertEqual(blocked[:3], [5, 6, 7])  # first occurrence is still allowed

    def test_end_token_stops_row_and_is_excluded(self):
        seq = {START: 5, 5: 6, 6: END}
        step = make_step(lambda prev: seq.get(prev, 9))
        out = decode_ids(step, 1, START, END, 50)[0]
        self.assertEqual(out, [5, 6])
        self.assertEqual(len(step.calls), 3)  # stopped right after <end>

    def test_end_in_batch_does_not_extend_finished_row(self):
        # Row 0 starts from START and ends after one token; row 1 never ends.
        def step(prev):
            probs = np.full((2, VOCAB), 0.01)
            probs[0, END if prev[0] == 5 else 5] = 1.0
            probs[1, 8] = 1.0
            return probs
        out = decode_ids(step, 2, START, END, 6)
        self.assertEqual(out[0], [5])
        self.assertEqual(out[1], [8] * 6)

    def test_max_length(self):
        out = decode_ids(make_step(lambda prev: 5), 3, START, END, 7)
        self.assertEqual([len(seq) for seq in out], [7, 7, 7])
        self.assertEqual(decode_ids(make_step(lambda prev: 5), 1, START, END, 0), [[]])


_CHECKPOINT_FILES = (config.MODEL_WEIGHTS_PATH, config.TOKENIZER_PATH, config.DATA_DIR / "test" / "encoder_input.npy")


@unittest.skipUnless(all(p.exists() for p in _CHECKPOINT_FILES), "checkpoint/tokenizer/test data not present")
class CheckpointSmokeTest(unittest.TestCase):
    """Loads the real checkpoint and decodes 2 test articles for 10 steps (a few seconds)."""

    @classmethod
    def setUpClass(cls):
        cls.s = Summarizer()
        cls.enc = np.load(config.DATA_DIR / "test" / "encoder_input.npy")[:2]

    def test_decoding_is_valid(self):
        for kwargs in ({"suppress_unk": False, "no_repeat_ngram": 0}, {"suppress_unk": True, "no_repeat_ngram": 3}):
            ids = self.s.generate_ids(self.enc, max_len=10, **kwargs)
            self.assertEqual(len(ids), 2)
            for seq in ids:
                self.assertLessEqual(len(seq), 10)
                self.assertTrue(all(0 <= t < self.s.preprocessor.actual_vocab_size for t in seq))
                self.assertNotIn(self.s.end_id, seq)
                self.assertIsInstance(self.s.preprocessor.decode_sequence(seq), str)
                if kwargs["suppress_unk"]:
                    self.assertNotIn(self.s.unk_id, seq)
                    self.assertEqual(len(set(trigrams(seq))), len(trigrams(seq)))

    def test_stepwise_inference_matches_training_forward_pass(self):
        # Guards the inference wiring: feeding the generated tokens back through the
        # teacher-forced model.call() must reproduce the same per-step distributions.
        seqs = self.s.generate_ids(self.enc, max_len=10, suppress_unk=True, no_repeat_ngram=3)
        n = min(len(seq) for seq in seqs)
        dec_in = np.array([[self.s.start_id] + seq[:n - 1] for seq in seqs], dtype=np.int32)
        forward, _ = self.s.model((self.enc, dec_in))

        enc_out, h, c = self.s.model.encoder(self.enc)
        for t in range(n):
            probs, h, c, _ = self.s.model.decoder(dec_in[:, t:t + 1], h, c, enc_out)
            np.testing.assert_allclose(probs.numpy(), forward.numpy()[:, t], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
