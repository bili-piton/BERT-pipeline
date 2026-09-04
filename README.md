# BERT-pipeline

A Python pipeline for extracting contextual word embeddings from scientific PDFs using SciBERT ([Beltagy et al., 2019](https://aclanthology.org/D19-1371/)). Designed for analysing stratigraphic terminology in Palaeolithic archaeology, but adaptable to any domain-specific vocabulary analysis in academic literature.

This repository accompanies:

Galfi, J. & Cascalheira, J. (forthcoming). Clarifying Stratigraphic Terminology in Palaeolithic Archaeology Using Natural Language Processing. *Advances in Archaeological Practice*.

## What it does

The pipeline reads a corpus of scientific papers (as PDFs), locates occurrences of user-defined target terms, and extracts SciBERT contextual embeddings for each occurrence. The output is a set of embedding vectors (one per term occurrence) that capture how each term is used in context, suitable for downstream clustering, dimensionality reduction, or semantic analysis.

### Pipeline steps

1. Extract text from PDFs (stops at the references section)
2. Preprocess: remove citations, lowercase, lemmatize, filter stopwords
3. Tokenize with SciBERT's WordPiece tokenizer
4. For each target term occurrence, extract a context window and pass it through SciBERT
5. Output the mean of the last hidden state as a 768-dimensional embedding vector
6. Export results to CSV (one file per window size)

## Requirements

- Python 3.10–3.11 (the range supported by the pinned dependency set; verified on 3.11.5)
- PyTorch
- Transformers (HuggingFace)
- PyMuPDF
- NLTK
- NumPy
- pandas

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` is pinned to the versions the published results were produced with — see [Reproducibility](#reproducibility).

The first run additionally downloads the SciBERT weights (~440 MB) from HuggingFace and three NLTK corpora, so it requires network access.

## Usage

1. Place your PDF files in a folder called `Papers/` (or edit `folder_paths` in the script)

2. Edit the target terms in the script if needed:
   ```python
   terms = ['occupation', 'layer', 'level', 'horizon', 'lithic', 'bone', 'flake', 'charcoal']
   ```

3. Run the pipeline:
   ```bash
   python bert_pipeline.py
   ```

4. Output files will be generated in the working directory:
   - `final_all_20.csv`
   - `final_all_30.csv`
   - `final_all_40.csv`
   - `final_all_50.csv`

Each CSV contains one row per term occurrence, with 768 columns (SciBERT hidden dimensions) plus a `Term` label column. The number in the filename refers to the context window size (in tokens).

## Configuration

All options are set by editing variables in the script:

| Variable | Default | Description |
|---|---|---|
| `folder_paths` | `['Papers']` | Folders containing input PDFs |
| `terms` | 8 archaeological terms | Target terms to extract embeddings for |
| `max_length` | `512` | BERT maximum token length per chunk |
| Window sizes | `20, 30, 40, 50` | Context window sizes (tokens right of target term) |

## Reproducibility

The analysis code in this repository is **frozen as published** — the text extraction, preprocessing, tokenisation, windowing and embedding steps are exactly those used for the accompanying paper. This section records what was verified so that the results can be reproduced, notes the few packaging changes made after acceptance, and documents the behaviours a replicator needs to know about in order to interpret the output correctly.

### Changes made after acceptance

Three changes were made to the repository after the paper was accepted. All are verified result-neutral: re-running the pipeline before and after produces **byte-identical** output.

1. Removed the unused `umap` and `matplotlib` imports (and their entries in `requirements.txt`). Neither was used by the analysis, but `import umap` was a module-level import, so the script would not start without a package it never called.
2. Pinned `requirements.txt` to the versions used for the published results.
3. Pinned the SciBERT model revision in `from_pretrained`.

No part of the analysis — text extraction, preprocessing, tokenisation, windowing or embedding — was modified.

### Verified environments

The pipeline was re-run and checked in September 2026 under two dependency stacks:

| Package | Development stack (now pinned) | Latest versions (compatibility-tested) |
|---|---|---|
| Python | 3.11.5 | 3.11.5 |
| torch | 2.11.0 | 2.11.0 |
| transformers | 4.32.1 | 5.16.1 |
| pymupdf | 1.27.2 | 1.28.2 |
| nltk | 3.8.1 | 3.10.3 |
| numpy | 1.24.4 | 2.4.6 * |
| pandas | 2.2.0 | 3.0.5 * |

\* The numpy 2.x / pandas 3.x combination was verified against the array-assembly and CSV-export path specifically; the SciBERT forward pass was verified on torch 2.11.

End-to-end runs were performed on the current stack; on the development stack the model load, tokenisation, preprocessing and forward pass were verified component by component. Embeddings produced under transformers 4.32.1 and 5.16.1 agree to a maximum absolute difference of **2.1e-6** (cosine similarity 1.0), i.e. float32 rounding noise only. The transformers 5.x load emits an `UNEXPECTED` key report for the `cls.*` masked-LM heads; this is expected when loading a pretraining checkpoint into `BertModel` and does not affect the encoder weights.

`requirements.txt` was originally unpinned, which meant a fresh install resolved to the right-hand column rather than the versions the code was written against. It is now pinned to the left-hand column, so `pip install -r requirements.txt` reproduces the development environment directly. The right-hand column is recorded because the pipeline was checked against it too: the code still runs there, so it is a viable fallback if the pinned versions become hard to install on future hardware.

Note that `import fitz` is deprecated as of PyMuPDF 1.28 (`import pymupdf` is the replacement) and will eventually stop working; the pinned 1.27.2 is unaffected.

### Model version

The pipeline loads `allenai/scibert_scivocab_uncased` from the HuggingFace Hub, pinned to the revision used for the published results:

```
revision 24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1
```

That repository has been unchanged upstream since 2022-10-03, so the pin selects the same weights the analysis originally used and protects against any future change to the model card.

The repository ships `pytorch_model.bin` only (no safetensors), so loading requires a transformers/torch pair that still accepts `.bin` checkpoints.

### NLTK data

The script calls `nltk.download()` for `stopwords`, `wordnet` and `words` at import time. These are fetched to `~/nltk_data` on first run and require network access. Depending on the NLTK version, `wordnet` may also need `omw-1.4`.

### Determinism

The model runs under `model.eval()` with no dropout, sampling, or random initialisation, and the script sets no seeds because it needs none. Two independent end-to-end runs on the same corpus and the same machine produced **byte-identical** CSV output, and repeated forward passes on the same input are bitwise identical.

One caveat affects **row order rather than row content**. Input PDFs are iterated in `os.listdir()` order, which is filesystem-dependent and is not sorted. The set of embeddings is therefore identical across machines, but the order of rows within each term block may differ. This matters if the CSVs are fed to an order-sensitive downstream method (UMAP included). For a byte-exact match across machines, sort the file list or sort the rows before downstream analysis.

### Runtime

Inference runs on CPU only — the script never moves the model or inputs to a GPU/MPS device — and performs one forward pass per term occurrence with a batch size of 1. The four window sizes are produced by four independent passes, each of which re-opens every PDF and recomputes text extraction, preprocessing and tokenisation from scratch, so the full run costs roughly four times a single-window run.

As a scale reference, a test corpus yielding ~4,800 term occurrences per window (~19,000 forward passes across all four window sizes) took about 7.5 minutes on an Apple M5 Max CPU, i.e. on the order of 40 windows per second. Runtime scales linearly with the number of term occurrences, so a full paper corpus should be budgeted in hours rather than minutes.

## Known behaviour and limitations

These are properties of the published code, verified empirically. They are documented here rather than changed, so that the released results remain reproducible.

**Plural forms are not counted.** In `preprocess_text`, the `w in english_words` dictionary filter is applied to the *raw* token, while lemmatisation happens afterwards. NLTK's `words` corpus does not list the plural forms of the target terms, so `layers`, `levels`, `occupations`, `flakes`, `bones`, `horizons`, `lithics` and `charcoals` are all discarded before they can be lemmatised to their singular forms. Verified on a test corpus containing `flakes` 120 times: the `flake` term yields **zero** embeddings. Term counts therefore reflect singular usages only.

**Only dictionary words survive preprocessing.** The same filter removes proper nouns, site names, most domain-specific compounds, and the fragments produced when a hyphenated word is split across a line break (`occupa-\ntion` becomes two non-words and is dropped).

**Context windows are right-sided.** Each window starts at the target term and extends up to `window_size` tokens to its right; no left-hand context is included.

**Windows do not cross 512-token chunk boundaries.** The token stream is cut into fixed 512-token chunks and terms are matched within each chunk, so a window near a chunk end is silently truncated. Measured on a test corpus, the proportion of occurrences receiving a shorter window than requested is approximately `window_size / 512`:

| Window size | Occurrences with a shortened window |
|---|---|
| 20 | 4.3% |
| 30 | 6.5% |
| 40 | 8.8% |
| 50 | 11.1% |

Terms that SciBERT splits into multiple WordPieces — `lithic` (`lith` + `##ic`), `flake` (`fl` + `##ake`) and `charcoal` (`char` + `##coal`) — additionally lose roughly one occurrence per chunk boundary crossed, because the pieces fall on either side of the cut.

**Windows are encoded without `[CLS]`/`[SEP]`.** Special tokens are added once around the whole document and the windows are sliced out of the interior, so almost every window reaches the model as a bare token sequence.

**Subword matching has no boundary guard.** A term's WordPiece sequence can match inside a longer word (`level` inside `levelling`). Checked exhaustively against the full post-filter vocabulary, this affects at most 22 word types per term (`bone`: 22, `level`: 10, `horizon`: 5, `layer`/`flake`: 2, `occupation`/`charcoal`: 1, `lithic`: 0) — items such as `bonedog`, `layerage` and `horizonward` that do not occur in this literature. The practical impact is negligible.

**Output carries no provenance.** Each CSV row is 768 embedding dimensions plus a `Term` label. There is no column identifying the source PDF or the position of the occurrence, so rows cannot be traced back to individual papers. (`paper_indices` is populated internally but never written out, and records the folder name rather than the filename.)

**Reference-section detection is coarse.** A document is cut off as soon as `references`, `bibliography` or `works cited` appears anywhere in the current 5-page block — including in running text. When this triggers, the whole block is discarded, so body text preceding the heading in that block is not analysed.

## Replication

To replicate the results from Galfi & Cascalheira (forthcoming), download the paper corpus from OSF: [https://doi.org/10.17605/OSF.IO/SQCRN](https://doi.org/10.17605/OSF.IO/SQCRN)

## Citation

If you use this pipeline, please cite:

```bibtex
@article{galfi_cascalheira_forthcoming,
  author = {Galfi, Jovan and Cascalheira, Jo\~{a}o},
  title = {Clarifying Stratigraphic Terminology in {Palaeolithic} Archaeology Using Natural Language Processing},
  journal = {Advances in Archaeological Practice},
  year = {forthcoming}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
