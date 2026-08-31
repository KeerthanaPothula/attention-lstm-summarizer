# Attention-Augmented LSTM for Abstractive Text Summarization with PDF and Image Input

I want you to act as a senior Machine Learning Engineer, NLP researcher, TensorFlow developer, and full-stack Python developer.

Build a COMPLETE, RUNNABLE, FINAL-YEAR PROJECT titled:

"Attention-Augmented LSTM for Abstractive Text Summarization with PDF and Image Input"

IMPORTANT:
I do NOT want a simple text summarization demo.
I want a complete end-to-end academic project with dataset training, testing, evaluation, visualization, PDF processing, image OCR, and a professional Streamlit application.

The project must be implemented in:
- Python
- TensorFlow 2.x / Keras
- Streamlit

The main summarization model MUST be an:
"Attention-Augmented LSTM Encoder-Decoder"

Do NOT replace the main model with a Transformer.

========================================================
1. PROJECT OBJECTIVE
========================================================

The final system must allow a user to upload:

1. PDF
2. Image (PNG/JPG/JPEG/WEBP)
3. Optional TXT file
4. Optional manually entered text

The system must:

PDF:
PDF → text extraction → preprocessing → summarization

IMAGE:
Image → OCR → extracted text → preprocessing → summarization

TEXT:
Text → preprocessing → summarization

The final workflow must be:

User Input
   ↓
PDF Extraction / OCR
   ↓
Text Preprocessing
   ↓
Tokenization
   ↓
Embedding
   ↓
LSTM Encoder
   ↓
Bahdanau Attention
   ↓
LSTM Decoder
   ↓
Abstractive Summary
   ↓
Evaluation
   ↓
Attention Visualization
   ↓
Final Dashboard


========================================================
2. DATASET
========================================================

Use the CNN/DailyMail dataset for supervised abstractive text summarization.

The dataset MUST be divided into separate:

TRAIN
VALIDATION
TEST

sets.

Use a clearly documented split.

Prefer:

72% Training
8% Validation
20% Testing

OR use:

80% Training
10% Validation
10% Testing

Choose one and explain why.

The test set MUST NEVER be used during training.

The code must clearly print:

Total samples
Training samples
Validation samples
Testing samples

Also print:

Vocabulary size
Maximum input sequence length
Maximum target sequence length


IMPORTANT:
Do not accidentally train on the test set.

Make the split reproducible using a fixed random seed.

========================================================
3. DATASET SUBSET SUPPORT
========================================================

CNN/DailyMail is computationally large.

Therefore, create configuration options such as:

USE_SUBSET = True

TRAIN_SAMPLES = 10000
VALIDATION_SAMPLES = 1000
TEST_SAMPLES = 1000

The user must be able to switch to full-dataset training later.

Clearly display whether the experiment uses:

FULL DATASET

or

EXPERIMENTAL SUBSET

Never present subset results as full-dataset results.

========================================================
4. TEXT PREPROCESSING
========================================================

Implement a proper preprocessing pipeline.

Include:

- text cleaning
- whitespace normalization
- appropriate punctuation handling
- lowercasing where appropriate
- tokenization
- vocabulary generation
- unknown token handling
- padding
- truncation
- sequence length control
- START token
- END token

Use consistent preprocessing during:

training
validation
testing
inference

Do NOT use different preprocessing logic for uploaded files.

========================================================
5. PDF INPUT
========================================================

The Streamlit application must allow PDF uploads.

Use a reliable Python PDF extraction library such as:

PyMuPDF

The PDF pipeline must be:

PDF
 ↓
Read all pages
 ↓
Extract text from every page
 ↓
Combine text
 ↓
Clean text
 ↓
Chunk if necessary
 ↓
Summarize
 ↓
Display results

The application must support multi-page PDFs.

Display:

- filename
- file size
- page count
- extracted text
- original word count
- processing time
- summary

For extremely long PDFs, implement sensible chunking.

Do NOT summarize only the first page.

========================================================
6. IMAGE INPUT / OCR
========================================================

The application must support:

PNG
JPG
JPEG
WEBP

Use OCR such as:

Tesseract OCR / pytesseract

Pipeline:

IMAGE
 ↓
Image preprocessing
 ↓
OCR
 ↓
Extracted text
 ↓
Text preprocessing
 ↓
Attention-LSTM
 ↓
Summary

Display:

- uploaded image
- extracted OCR text
- OCR processing time
- word count
- generated summary

Clearly explain in the code and README:

OCR:
Image → Text

Summarization:
Text → Summary

Do not confuse OCR with the summarization model.

If Tesseract is not installed, show a clear error message explaining how to install it.

========================================================
7. MAIN MODEL
========================================================

The main model MUST be:

Attention-Augmented LSTM Encoder-Decoder.

Architecture:

Input Tokens
     ↓
Embedding
     ↓
LSTM Encoder
     ↓
Encoder Hidden States
     ↓
Bahdanau Attention
     ↓
Context Vector
     ↓
LSTM Decoder
     ↓
Dense Layer
     ↓
Softmax
     ↓
Generated Summary

The encoder MUST return the complete sequence of hidden states because attention requires them.

========================================================
8. LSTM ENCODER
========================================================

Implement a proper TensorFlow/Keras LSTM encoder.

Include:

- embedding layer
- LSTM
- hidden state
- cell state
- encoder sequence outputs

Make dimensions configurable.

For example:

EMBEDDING_DIM
LSTM_UNITS

Do not hard-code dimensions in multiple files.

========================================================
9. BAHDANAU ATTENTION
========================================================

Implement a REAL Bahdanau attention mechanism.

Do NOT use random attention weights.

The attention process should be:

Encoder Outputs
       +
Decoder Hidden State
       ↓
Attention Scores
       ↓
Softmax
       ↓
Attention Weights
       ↓
Context Vector

Conceptually:

score(s_t, h_i)

attention_weights = softmax(scores)

context_vector =
sum(attention_weights_i * encoder_output_i)

The attention layer MUST return:

1. context vector
2. attention weights

because the attention weights will be used for visualization.

Explain the mathematics in comments and README.

========================================================
10. DECODER
========================================================

Implement an LSTM decoder.

During training use teacher forcing appropriately.

During inference generate words autoregressively.

Generation must start with:

<START>

and terminate with:

<END>

Also enforce:

MAX_TARGET_LENGTH

to avoid infinite generation.

========================================================
11. DECODING
========================================================

Implement:

1. Greedy decoding

Preferably also implement:

2. Beam search

Make beam width configurable.

Explain:

Greedy Search
vs
Beam Search

If beam search is implemented, use it for final inference if it improves results.

Do not make fake claims about improvement.

========================================================
12. TRAINING
========================================================

Implement a complete training pipeline.

Include:

- training
- validation
- loss calculation
- optimizer
- checkpointing
- early stopping
- best-model saving
- training history

Use:

Adam optimizer

and:

SparseCategoricalCrossentropy

with appropriate masking/handling of padding tokens.

Save:

trained model
tokenizer
vocabulary
configuration
training history

so that the Streamlit app can load the trained model without retraining.

========================================================
13. CONFIGURATION
========================================================

Create one central configuration file.

Include:

DATASET
DATASET_SPLIT
USE_SUBSET
TRAIN_SAMPLES
VAL_SAMPLES
TEST_SAMPLES

VOCAB_SIZE
EMBEDDING_DIM
LSTM_UNITS

MAX_INPUT_LENGTH
MAX_TARGET_LENGTH

BATCH_SIZE
EPOCHS
LEARNING_RATE

BEAM_WIDTH

RANDOM_SEED

Do not scatter these parameters throughout the project.

========================================================
14. TRAINING RESULTS
========================================================

Generate and save:

Training Loss Curve
Validation Loss Curve

Plot:

Epoch vs Training Loss
Epoch vs Validation Loss

Also report:

- total training time
- number of epochs
- best validation loss
- trainable parameters
- batch size
- learning rate
- optimizer

========================================================
15. TESTING
========================================================

After training, evaluate ONLY on the unseen TEST SET.

Pipeline:

Test Article
 ↓
Trained Model
 ↓
Generated Summary
 ↓
Reference Summary
 ↓
ROUGE Evaluation

Do NOT calculate final test performance using training data.

Clearly report:

Number of test samples evaluated.

========================================================
16. ROUGE
========================================================

Implement:

ROUGE-1
ROUGE-2
ROUGE-L

These are mandatory.

Optionally implement:

BLEU

Report appropriate precision, recall and F1 values where available.

Use actual predictions.

NEVER fabricate ROUGE values.

Save evaluation results to CSV/JSON.

========================================================
17. USER UPLOAD VS TEST DATA
========================================================

This distinction is VERY IMPORTANT.

For CNN/DailyMail test samples:

Generated Summary
       +
Reference Summary
       ↓
ROUGE-1
ROUGE-2
ROUGE-L

For an arbitrary PDF/image uploaded by a user:

There may be no reference summary.

Therefore DO NOT fabricate ROUGE scores.

Instead display:

Original Word Count
Summary Word Count
Compression Ratio
Inference Time
Generated Summary
Attention Visualization

If the user optionally provides a reference summary, then calculate:

ROUGE-1
ROUGE-2
ROUGE-L

========================================================
18. COMPRESSION RATIO
========================================================

Calculate:

Compression Ratio =
(summary word count / original word count) × 100

Display:

Original Words
Summary Words
Compression Ratio

========================================================
19. ATTENTION HEATMAP
========================================================

Create a REAL attention visualization using the actual attention weights from the trained model.

The heatmap must show:

Input words
versus
Generated summary words

Do NOT generate fake attention values.

Save heatmaps to:

results/plots/

Also display them in Streamlit.

========================================================
20. ERROR ANALYSIS
========================================================

Implement qualitative error analysis.

Analyze:

1. Repetition
2. Omission
3. Hallucination
4. Incorrect information
5. Long input documents

Show example:

SOURCE
REFERENCE SUMMARY
GENERATED SUMMARY
ERROR TYPE

Do not claim automatic factuality detection is perfect.

If hallucination detection is heuristic, clearly label it as heuristic.

========================================================
21. PDF PAGE-WISE SUMMARY
========================================================

For multi-page PDFs, provide:

1. Overall summary

and optionally:

2. Page-wise summaries

For very long documents, use chunking and hierarchical summarization if required.

Clearly explain this approach.

========================================================
22. STREAMLIT APPLICATION
========================================================

Create a professional Streamlit application.

Run using:

streamlit run app.py

The interface should contain:

HOME
DOCUMENT UPLOAD
EXTRACTED TEXT
SUMMARY
MODEL PERFORMANCE
ATTENTION VISUALIZATION
ERROR ANALYSIS
DATASET INFORMATION

========================================================
23-30. UI PAGES DETAIL
========================================================

HOME: Title, description, upload entry point.

UPLOAD: Support PDF, PNG, JPG, JPEG, WEBP, TXT. Show filename, file size, file type.

EXTRACTED TEXT: Expandable area. For PDF: pages, word count. For image: OCR text, OCR processing time, word count.

SUMMARY PAGE: Generated summary, then cards for Original Words, Summary Words, Compression Ratio, Inference Time.

EVALUATION DASHBOARD: ROUGE-1/2/L, BLEU if implemented. Only display ROUGE when a valid reference summary is available.

MODEL INFORMATION: Model type, Encoder, Decoder, Attention type, Dataset, Dataset Split, Vocabulary Size (actual), Max Input Length (actual), Max Summary Length (actual), Embedding Dimension (actual), LSTM Units (actual), Optimizer, Learning Rate (actual), Trainable Parameters (actual).

DATASET DASHBOARD: Total/Train/Val/Test samples, Vocabulary Size, Max Input/Target Length, whether FULL DATASET or SUBSET was used.

MODEL PERFORMANCE DASHBOARD: Training Loss, Validation Loss, ROUGE-1/2/L, BLEU, Training Time, Inference Time, with charts.

========================================================
31. PROJECT STRUCTURE
========================================================

attention_lstm_summarizer/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── preprocessing/
│   ├── __init__.py
│   ├── text_preprocessor.py
│   ├── pdf_processor.py
│   └── ocr_processor.py
│
├── model/
│   ├── __init__.py
│   ├── encoder.py
│   ├── attention.py
│   ├── decoder.py
│   └── summarizer.py
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   └── evaluate.py
│
├── inference/
│   ├── __init__.py
│   ├── summarizer.py
│   └── beam_search.py
│
├── evaluation/
│   ├── __init__.py
│   ├── rouge.py
│   ├── metrics.py
│   └── error_analysis.py
│
├── visualization/
│   ├── __init__.py
│   ├── loss_curves.py
│   └── attention_heatmap.py
│
├── models/
│
├── results/
│   ├── plots/
│   ├── metrics/
│   └── summaries/
│
└── sample_documents/

You may modify this structure if technically necessary, but explain why.

========================================================
32. CODE QUALITY
========================================================

Modular, readable, maintainable, properly commented, OOP where appropriate, error-handled, validated, reproducible.

Do not create: TODO, pass, placeholder_function(), fake_output(), random_metrics(). No undefined variables.

========================================================
33. ERROR HANDLING
========================================================

Handle: missing model files, missing tokenizer, invalid PDF, scanned PDF, image without readable text, empty extracted text, very long documents, unsupported file types, Tesseract not installed, model generation failure, memory errors where practical.

Display useful Streamlit error messages.

========================================================
34. LONG DOCUMENT HANDLING
========================================================

PDF → Extract Text → Split into chunks → Summarize each chunk → Combine chunk summaries → Final summarization → Final Summary.

Do not simply truncate a 50-page document without informing the user.

========================================================
35. RESEARCH PAPER COMPARISON
========================================================

Suitable for comparison against at least TWO research papers published between 2024 and 2026. DO NOT INVENT PAPERS.

Comparison table: Paper, Year, Dataset, Model, Attention, ROUGE-1/2/L, Training characteristics, Inference characteristics, Limitations. Compare Paper 1, Paper 2, Our Proposed Model.

========================================================
36. NOVELTY
========================================================

CORE MODEL: Attention-Augmented LSTM.
APPLICATION CONTRIBUTIONS: PDF support, Image OCR support, Multi-page document handling, Attention visualization, Beam search, Comprehensive evaluation, Interactive Streamlit interface.

Do NOT falsely claim UI is a novel architecture.

========================================================
37. BONUS FEATURES
========================================================

If practical: Bahdanau vs Luong Attention comparison, Beam Search, Interactive Streamlit interface, Transformer baseline comparison. Keep main model as Attention-LSTM. Optional features must not break core project.

========================================================
38. REPORT-READY OUTPUT
========================================================

Automatically save: dataset_statistics.csv, training_history.csv, test_metrics.csv, generated_summaries.csv, and plots: training_validation_loss.png, attention_heatmap.png, rouge_comparison.png.

========================================================
39. README
========================================================

Detailed README: Title, Description, Problem Statement, Objectives, Features, Architecture, Dataset, Dataset Split, Preprocessing, LSTM Encoder, Bahdanau Attention, LSTM Decoder, Training, Testing, ROUGE Evaluation, PDF Processing, OCR, Attention Visualization, Error Analysis, Installation, Usage, Streamlit Instructions, Limitations, Future Scope.

========================================================
40. REQUIREMENTS.TXT
========================================================

Only packages actually used: tensorflow, numpy, pandas, streamlit, PyMuPDF, Pillow, pytesseract, rouge-score, nltk, matplotlib, scikit-learn (verify every import is included).

========================================================
41. INSTALLATION
========================================================

venv setup, pip install -r requirements.txt, dataset setup instructions, Tesseract installation instructions separately (external executable).

========================================================
42. EXECUTION
========================================================

Dataset preparation, training (python training/train.py), evaluation (python training/evaluate.py), inference, streamlit run app.py.

========================================================
43. CODE DELIVERY FORMAT
========================================================

Provide code FILE-BY-FILE, each complete and runnable. After all files: Installation, Dataset setup, Training, Evaluation, Streamlit execution, Example workflow, Troubleshooting, Expected outputs.

========================================================
44. COMPUTATIONAL REQUIREMENT
========================================================

Must support SMALL SUBSET MODE for development/testing on limited GPU/CPU, and FULL DATASET MODE for final experiments. Clearly distinguish the two.

========================================================
45. NO FABRICATED RESULTS
========================================================

Never write hardcoded ROUGE/accuracy/timing/parameter values unless actually calculated from the trained model. Use "Actual result generated after training" placeholders in docs where a real run hasn't happened yet.

========================================================
46. FINAL VALIDATION CHECKLIST
========================================================

Dataset loads; train/val/test split works; test data not used in training; preprocessing works; tokenization works; padding works; encoder works; Bahdanau attention works; decoder works; training works; validation works; model saves/loads; test evaluation works; ROUGE works; attention weights are real; heatmap works; PDF extraction works (multi-page); image OCR works; text summarization works; long-document handling works; compression ratio works; Streamlit works; error handling works; requirements.txt correct; README complete.

========================================================
47. FINAL RESPONSE FORMAT
========================================================

1. Complete architecture
2. Complete folder structure
3. Explanation of how modules interact
4. Every source file completely
5. Installation commands
6. Dataset preparation
7. Training command
8. Evaluation command
9. Streamlit command
10. Example PDF workflow
11. Example image workflow
12. Expected outputs
13. Troubleshooting

Do NOT stop after giving the architecture. Do NOT ask to implement missing parts. Do NOT give pseudocode. Generate the complete project code.
