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

- Python 3.8+
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
