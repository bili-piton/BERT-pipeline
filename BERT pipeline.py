import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
import fitz  # PyMuPDF
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
english_words = set(words.words())

# Initialize BERT tokenizer and model
model_name = 'allenai/scibert_scivocab_uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# Parameters
window_size = 20  # Sliding window size
folder_paths = ['test']  # List of folders with PDFs to process

# Terms to analyze
terms = ['occupation', 'layer', 'level', 'horizon', 'lithic', 'bone', 'charcoal', 'flake']

# Containers for embeddings and paper labels
term_embeddings = {term: [] for term in terms}
paper_indices = {term: [] for term in terms}

# Store tokenized windows for inspection (optional)
tokenized_windows = []

def process_text(text, term, filename):
    """
    For a given text and a target term, find all windows where the term occurs,
    get BERT embeddings for the window, and collect the embeddings.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    term_tokens = tokenizer.tokenize(term)
    term_ids = tokenizer.convert_tokens_to_ids(term_tokens)
    term_length = len(term_ids)
    
    embeddings_list = []
    
    for start in range(0, len(tokens) - term_length + 1):
        if tokens[start:start + term_length] == term_ids:
            # Define window bounds, making sure not to exceed tokens length
            window_end = min(start + window_size, len(tokens))
            window_input_ids = torch.tensor([tokens[start:window_end]])
            
            with torch.no_grad():
                outputs = model(window_input_ids)
                # outputs.last_hidden_state shape: (batch, seq_len, hidden_size)
                # Average across sequence tokens (dim=1)
                window_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
                embeddings_list.append(window_embedding)
            
            # Save tokenized window for debugging/inspection
            tokenized_window = tokenizer.convert_ids_to_tokens(tokens[start:window_end])
            tokenized_words = [tokenizer.convert_tokens_to_string([token]) for token in tokenized_window]
            tokenized_windows.append({
                'document': filename,
                'window_start_index': start,
                'tokenized_window': ' '.join(tokenized_words)
            })
    
    return embeddings_list


# Process PDFs in each folder
for folder_path in folder_paths:
    print(f"Processing PDF files in folder: {folder_path}")
    label = os.path.basename(folder_path)
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    total_files = len(pdf_files)

    for i, filename in enumerate(pdf_files, 1):
        pdf_path = os.path.join(folder_path, filename)
        print(f"Processing file {i}/{total_files} in folder '{label}': {filename}")
        
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            references_keywords = ['references', 'bibliography', 'works cited']

            for start_page in range(0, num_pages, 5):
                end_page = min(start_page + 5, num_pages)
                text = ""
                stop_extraction = False

                for page_num in range(start_page, end_page):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    text += page_text

                    if any(keyword in page_text.lower() for keyword in references_keywords):
                        print(f"References section found in {filename} (page {page_num}). Stopping extraction.")
                        stop_extraction = True
                        break

                if stop_extraction:
                    break

                # Clean and preprocess text
                cleaned_text = re.sub(r'\(.*?\)', '', text)  # Remove parentheses content
                cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)  # Remove special chars
                cleaned_text = cleaned_text.lower()
                tokens = cleaned_text.split()
                tokens = [
                    lemmatizer.lemmatize(word)
                    for word in tokens
                    if len(word) > 2 and word not in stop_words and word in english_words
                ]
                preprocessed_text = ' '.join(tokens)

                # Extract embeddings for each term
                for term in terms:
                    embeddings = process_text(preprocessed_text, term, filename)
                    if embeddings:
                        term_embeddings[term].extend(embeddings)
                        paper_indices[term].extend([label] * len(embeddings))

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue


# Convert lists to numpy arrays
for term in terms:
    term_embeddings[term] = np.array(term_embeddings[term])
    print(f"{term.capitalize()} embeddings shape: {term_embeddings[term].shape}")

# Flatten embeddings into a DataFrame
embedding_rows = []
for term, embeddings in term_embeddings.items():
    for emb in embeddings:
        embedding_rows.append({'term': term, **{f'dim_{i}': emb[i] for i in range(len(emb))}})

embeddings_df = pd.DataFrame(embedding_rows)


# === t-SNE visualization ===

# Group definitions
groups = {
    "LFBC": ['lithic',  'flake', 'bone', 'charcoal'],
    "OLLH": ['occupation', 'layer', 'level', 'horizon']
}

# Colors for plotting
colors = {
    'occupation': '#377eb8', 'layer': '#ff7f00', 'level': '#4daf4a', 'horizon': '#f781bf',
    'lithic': '#984ea3', 'bone': '#dede00', 'charcoal': '#999999',
    'flake': '#e41a1c',
}

default_color = 'blue'
default_marker = 'o'  # Use same marker for all (no special_markers defined)

fontsize = 10
perplexities = [50, 75, 100, 200]

subplot_labels = [chr(i) for i in range(ord('a'), ord('a') + len(perplexities) * len(groups))]

fig, axes = plt.subplots(len(perplexities), len(groups), figsize=(14, 5 * len(perplexities)))
axes = axes.flatten()

plot_idx = 0

for perplexity in perplexities:
    print(f"\n=== Running t-SNE with perplexity = {perplexity} ===")

    for group_name, group_terms in groups.items():
        print(f"Processing group: {group_name}...")

        group_data = embeddings_df[embeddings_df['term'].isin(group_terms)].reset_index(drop=True)
        embeddings = group_data.loc[:, group_data.columns.str.startswith('dim_')].values
        terms_array = group_data['term'].values

        # Encode terms as numeric labels for silhouette_score
        le = LabelEncoder()
        numeric_labels = le.fit_transform(terms_array)

        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)

        ax = axes[plot_idx]

        # Plot each term in group
        for term in group_terms:
            mask = terms_array == term
            ax.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                label=term,
                color=colors.get(term, default_color),
                alpha=0.6,
                edgecolor='black',
                s=50,
                marker=default_marker
            )

        ax.set_title(f'{group_name} (Perplexity {perplexity})', fontsize=16)
        ax.set_xlabel('Component 1', fontsize=fontsize)
        ax.set_ylabel('Component 2', fontsize=fontsize)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=fontsize, framealpha=0.7)

        # Compute and display silhouette score
        try:
            silhouette = silhouette_score(reduced_embeddings, numeric_labels, metric='euclidean')
            score_text = f'Silhouette: {silhouette:.2f}'
        except Exception as e:
            score_text = f'Silhouette: N/A'

        ax.text(
            0.95, 0.05,
            score_text,
            transform=ax.transAxes,
            fontsize=fontsize + 4,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

        ax.text(
            0.05, 0.95,
            subplot_labels[plot_idx],
            transform=ax.transAxes,
            fontsize=fontsize + 14,
            verticalalignment='top',
            horizontalalignment='left',
            weight='bold',
            color='gray',
            alpha=0.4
        )

        plot_idx += 1

plt.tight_layout()
plt.show()

