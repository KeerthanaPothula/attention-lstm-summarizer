# Attention-Augmented LSTM for Abstractive Text Summarization

An LSTM encoder-decoder with Bahdanau attention, trained on CNN/DailyMail
in TensorFlow/Keras. No Transformer components.

## Architecture

```
article tokens -> Embedding(128) -> LSTM encoder(256, full sequence + final h/c)
                                          |
             previous token -> Embedding(128)
                                          v
     Bahdanau attention(decoder h, encoder outputs) -> context vector
                                          v
     concat(context, embedding) -> LSTM decoder(256) -> Dense(vocab) -> softmax
```

| File | Role |
|---|---|
| `model/encoder.py` | Embedding + LSTM; returns every encoder hidden state plus the final h/c |
| `model/attention.py` | Bahdanau (additive) attention: `score = V·tanh(W1·s_t + W2·h_i)`, `weights = softmax(score)`, `context = Σ weights_i·h_i` |
| `model/decoder.py` | Single-timestep decoder used by both training and inference |
| `model/summarizer.py` | Teacher-forced training model + masked cross-entropy loss (padding ignored) |
| `inference/summarizer.py` | Autoregressive decoding (see [Decoding](#decoding)) |
| `training/train.py` | Training loop (Adam, early stopping, best-checkpoint saving, resumable) |
| `training/evaluate.py` | Test-set ROUGE evaluation |

The checked-in `config.py` describes the GitHub checkout: vocabulary limit 20,000, input length 300 tokens, target length 60 tokens (including `<start>`/`<end>`), embedding dimension 128, LSTM units 256, batch size 32, learning rate 1e-3, and early-stopping patience 3. The separately recovered Colab checkpoint uses vocabulary limit 10,000, embedding dimension 256, and LSTM units 512. Its evaluation results below belong to that separate experiment, not the checked-in configuration.

## Dataset and split

Data comes from `abisee/cnn_dailymail` v3.0.0 via Hugging Face `datasets`.
The project uses the dataset's **official** train/validation/test split, not
a custom re-split, so test scores come from the official test split. Subset
mode samples from within each official split with seed 42. The tokenizer is
fitted on the training split only, and the test split is read only by
`training/evaluate.py`.

There are two experiments, and they used different data sizes:

| | Colab experiment | This checkout |
|---|---:|---:|
| Train examples | 100,000 | 10,000 |
| Validation examples | 10,000 | 1,000 |
| Test examples | 2,000 | 1,000 |
| Data / checkpoints | Google Drive (not in this repo) | `data/`, `models/` (git-ignored) |

## Training summary

**Colab experiment:** trained for 12 epochs. The best checkpoint (lowest
validation loss) was from epoch 9. That checkpoint and its training history
are on Google Drive and are not part of this repository.

**This checkout:** 2 completed epochs on the 10k subset. Train loss went from
7.47 to 7.07 and validation loss from 7.08 to 6.86, over about 69 minutes of
CPU time (`models/training_progress.json`). This checkpoint is far from
converged.

## Decoding

`inference/summarizer.py` has one greedy decoding loop (`decode_ids`). It
starts from `<start>`, stops a row when `<end>` is generated (the `<end>` id
is not included in the output), and caps every row at `MAX_TARGET_LENGTH - 1`
steps. It can apply two constraints at inference time; neither one changes
the model:

- **`<unk>` suppression** (`suppress_unk`): the `<unk>` token is masked out
  before argmax.
- **No-repeat n-gram blocking** (`no_repeat_ngram`, 3 = trigram): any token
  that would recreate an n-gram already in the output is masked out.

The defaults in `config.py` are `SUPPRESS_UNK = True` and
`NO_REPEAT_NGRAM_SIZE = 3`. For plain greedy decoding, pass
`suppress_unk=False, no_repeat_ngram=0`.

```python
from inference.summarizer import Summarizer
s = Summarizer()                       # loads data/tokenizer.pkl + models/summarizer.weights.h5
s.summarize(["article text ..."])      # improved decoding (config defaults)
s.summarize(["..."], suppress_unk=False, no_repeat_ngram=0)   # plain greedy
```

Beam search (`BEAM_WIDTH` in `config.py`) is not implemented yet.

## Evaluation methodology

`python training/evaluate.py [--limit N] [--weights PATH]`

1. Load `data/test/encoder_input.npy` and the cleaned reference summaries.
2. Decode every test article twice, once with plain greedy decoding and once
   with improved decoding, using the same checkpoint.
3. Score each generated summary against its reference with `rouge-score`
   (ROUGE-1, ROUGE-2, ROUGE-L, Porter stemming) and average precision,
   recall and F1 over the test set.
4. Write each mode to its own files:
   `results/metrics/test_metrics_{greedy,improved}.csv` and
   `results/metrics/generated_summaries_{greedy,improved}.csv`.

References are lowercased and cleaned with the same `clean_text()` used in
training, so the scores are measured on normalized text.

## ROUGE results

These results come from two different experiments: different checkpoints,
different data sizes and different test sets. **Do not compare numbers across
the two tables.**

### Colab experiment (epoch-9 checkpoint, 2,000 test examples)

These numbers were regenerated in Colab using the recovered epoch-9
checkpoint and the 2,000 saved test examples on Google Drive. The evaluation
generated predictions with plain greedy decoding and with `<unk>` suppression
plus trigram blocking, then calculated mean per-example ROUGE F1 using
`rouge-score` with Porter stemming. The checkpoint and test artifacts remain
in the Drive backup and are not committed to this repository.

| Metric | Greedy | Greedy + `<unk>` suppression + trigram blocking |
|---|---:|---:|
| ROUGE-1 F1 | 0.1857 | 0.2650 |
| ROUGE-2 F1 | 0.0373 | 0.0550 |
| ROUGE-L F1 | 0.1342 | 0.1758 |

### This checkout (epoch-2 checkpoint, 1,000 test examples)

These numbers come from `python training/evaluate.py` run on this checkout's
epoch-2 checkpoint. The script writes them to
`results/metrics/test_metrics_{greedy,improved}.csv`. `results/` is
git-ignored, so those CSVs (and the generated-summary CSVs) are **not
committed**. Re-run the script to regenerate them.

| Metric | Greedy | Greedy + `<unk>` suppression + trigram blocking |
|---|---:|---:|
| ROUGE-1 F1 | 0.0000 | 0.1434 |
| ROUGE-2 F1 | 0.0000 | 0.0179 |
| ROUGE-L F1 | 0.0000 | 0.1162 |

At epoch 2, plain greedy decoding outputs only `<unk>` for every article, so
every score is zero. With improved decoding there are only 6 distinct outputs
across all 1,000 articles: generic, frequent-word text that ignores the
input. The non-zero ROUGE here comes from overlap on common words, not from
real summarization. This checkpoint needs more training before its scores mean
anything.

Checks showing this comes from the undertrained checkpoint, not a decoding bug:

- The checkpoint loads correctly. Teacher-forced loss on the test set is 6.885,
  close to the training-time validation loss of 6.86. A freshly initialized
  model scores 9.90, which is about ln(20,000).
- Step-by-step inference reproduces the training forward pass:
  `CheckpointSmokeTest` checks that per-step probabilities match
  `model.call()` within 1e-5.
- `<unk>` is the most frequent token in the training targets (5.2%, followed
  by "the" at 3.7%). At this loss the model has mostly learned token
  frequencies, so plain argmax picks `<unk>` at every step and never reaches
  `<end>` before the 59-step cap.
- The first-step distribution barely depends on the article: across 200 test
  articles, the per-token standard deviation is about 7e-8 against a mean
  probability of 5e-5. All 200 have the same argmax. With `<unk>` masked and
  trigrams blocked, decoding follows the same frequent-word path, and the six
  variants come from near-tied probabilities near the end of the sequence.

The recovered Colab model's generated examples also show repetition and
unsupported details. For example, a reference about medical marijuana produced
an unrelated summary about government topics. A successful inference call means
the checkpoint and decoding code run; it does not establish factual reliability.
The improved decoding scores are lexical-overlap measurements, not correctness
scores.

## Limitations

- **ROUGE measures n-gram overlap, not correctness.** A higher ROUGE score
  does not mean a summary is factually accurate, faithful to the source, or
  free of hallucinations. The decoding constraints remove `<unk>` tokens and
  repeated phrases, which raises overlap with the reference, but they do
  nothing to make the content more correct.
- The Colab evaluation depends on the recovered checkpoint, tokenizer and
  test artifacts stored in Google Drive; these artifacts are not committed to
  this repository. The reported metrics were regenerated in Colab from those
  artifacts, but a clean end-to-end rerun using only this repository has not
  been verified.
- The checkpoint in this checkout is undertrained (2 epochs on 10k
  examples) and does not produce input-dependent summaries.
- Trigram blocking is a heuristic. It can also block legitimate repeats, such
  as a name that appears twice.
- Decoding is greedy only; beam search is not implemented.
- The current checkout's vocabulary limit is 20,000 and its model has no
  copy mechanism, so
  rare names and numbers become `<unk>`. Suppressing `<unk>` replaces them
  with some other word; it does not recover the missing word.
- Articles are truncated to 300 tokens, so content later in long articles is
  never seen.
- The Streamlit app, PDF/OCR input, attention heatmaps and error analysis
  from `PROJECT_REQUIREMENTS.md` are not implemented in this checkout.

## Usage

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python data/prepare_dataset.py        # download + tokenize (subset mode by default)
python training/train.py              # train / resume
python training/evaluate.py           # test-set ROUGE, greedy vs improved
python -m unittest discover tests     # decoding unit tests
```
