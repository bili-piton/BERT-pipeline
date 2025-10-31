###install the necessary lybraries first
import os
import re
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import fitz  # PyMuPDF
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
import umap 




## ---------------- Setup ----------------
nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('words')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
english_words = set(words.words())

model_name = 'allenai/scibert_scivocab_uncased'
print(f"Loading model '{model_name}'...")
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()
print("Model loaded and set to evaluation mode.")

max_length = 512
folder_paths = ['Papers'] ###set up your own folder path 
terms = ['occupation', 'layer', 'level', 'horizon', 'lithic', 'bone', 'flake', 'charcoal']

# ----------- Helper functions ----------
def preprocess_text(text):
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    toks = text.split()
    toks = [lemmatizer.lemmatize(w) for w in toks
            if len(w) > 2 and w not in stop_words and w in english_words]
    return ' '.join(toks)

def process_text(text, term, window_size):
    # tokenize the *whole* preprocessed text
    tokens = tokenizer.encode(text, add_special_tokens=True)
    out_vecs = []

    # term as WordPieces/ids
    term_tokens = tokenizer.tokenize(term)
    term_ids = tokenizer.convert_tokens_to_ids(term_tokens)
    term_length = len(term_ids)

    # ensure window can at least contain the whole term
    eff_window = max(window_size, term_length)

    # slide through the document in 512-token chunks
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i+max_length]
        if not chunk:
            continue

        # search for the term *within this chunk*
        for start in range(0, len(chunk) - term_length + 1):
            if chunk[start:start + term_length] == term_ids:
                # term at the *beginning* of the window, followed by right context
                win_start = start
                win_end = min(len(chunk), start + eff_window)

                window_input_ids = torch.tensor([chunk[win_start:win_end]])
                with torch.no_grad():
                    window_outputs = model(window_input_ids)
                    window_emb = window_outputs.last_hidden_state.squeeze(0).mean(dim=0).numpy()
                    out_vecs.append(window_emb)

    return out_vecs

def is_references_section(text):
    refs = ['references', 'bibliography', 'works cited']
    tl = text.lower()
    return any(k in tl for k in refs)

# --------------- One full run for a given window size ---------------
def build_dataset_for_window(window_size):
    print(f"\n=== Building dataset (window_size={window_size}) ===")

    term_embeddings = {term: [] for term in terms}
    paper_indices   = {term: [] for term in terms}
    term_counts     = {term: 0 for term in terms}

    for folder_path in folder_paths:
        label = os.path.basename(folder_path)
        pdf_names = [fn for fn in os.listdir(folder_path) if fn.endswith('.pdf')]
        total_pdfs = len(pdf_names)
        processed_pdfs = 0

        for filename in pdf_names:
            processed_pdfs += 1
            sys.stdout.write(
                f"\rProcessing {processed_pdfs}/{total_pdfs} PDFs | " +
                " | ".join([f"{t}:{term_counts[t]}" for t in terms])
            )
            sys.stdout.flush()

            pdf_path = os.path.join(folder_path, filename)
            try:
                doc = fitz.open(pdf_path)
                num_pages = len(doc)

                stop_doc = False
                for start_page in range(0, num_pages, 5):
                    end_page = min(start_page + 5, num_pages)
                    text = ""

                    for page_num in range(start_page, end_page):
                        page = doc.load_page(page_num)
                        text += page.get_text()
                        if is_references_section(text):
                            stop_doc = True
                            break

                    if stop_doc:
                        break

                    preprocessed_text = preprocess_text(text)

                    for term in terms:
                        vecs = process_text(preprocessed_text, term, window_size)
                        if vecs:
                            term_embeddings[term].extend(vecs)
                            paper_indices[term].extend([label] * len(vecs))
                            term_counts[term] += len(vecs)

            except Exception as e:
                print(f"\nError processing {pdf_path}: {e}")
                continue

    print("\n✅ Finished processing all PDFs.")
    print("Final term counts:", term_counts)

    # Combine into arrays
    for term in terms:
        term_embeddings[term] = np.array(term_embeddings[term])
        print(f"{term.capitalize()} embeddings shape: {term_embeddings[term].shape}")

    if not any(term_embeddings[t].size > 0 for t in terms):
        print("⚠️ No embeddings found for any terms. Skipping dataset build.")
        return

    all_embeddings = np.vstack([term_embeddings[t] for t in terms if term_embeddings[t].size > 0])
    term_labels = []
    for t in terms:
        term_labels.extend([t] * len(term_embeddings[t]))

    embeddings_df = pd.DataFrame(all_embeddings)
    embeddings_df['Term'] = term_labels
    out_csv = f'final_all_{window_size}.csv'
    embeddings_df.to_csv(out_csv, index=False)
    print(f"Saved dataset → {out_csv}")

# ---------------- Run for selected windows ----------------
for ws in [20, 30, 40, 50]:
    build_dataset_for_window(ws)
