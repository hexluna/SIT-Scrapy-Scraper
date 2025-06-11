import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from hashlib import md5
import torch
import time

# ========================== Initialize Embedding Model ========================== #
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')  # RAG-optimized model
model = model.to('cuda')
print("Using GPU:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback")
print("Model device:", next(model.parameters()).device)

# ========================== Text Cleaning and Chunking ========================== #
def clean_for_embedding(text):
    return text.replace('\n', ' ').strip()


def chunk_text(text, max_words=50, overlap=10):
    step = max_words - overlap
    if step <= 0:
        raise ValueError("Overlap must be less than max_words")
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words - overlap)
    ]

def infer_tags(text):
    tags = []
    if any(word in text.lower() for word in ["admission", "apply", "requirements", "enrollment"]):
        tags.append("admissions")
    if any(word in text.lower() for word in ["campus", "hostel", "library", "canteen"]):
        tags.append("facilities")
    if any(word in text.lower() for word in ["course", "degree", "programme", "module"]):
        tags.append("courses")
    if any(word in text.lower() for word in ["event", "orientation", "cca", "club"]):
        tags.append("student_life")
    return tags


# def chunk_text(text, max_words=150, overlap=20):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), max_words - overlap):
#         chunk = " ".join(words[i:i + max_words])
#         if chunk:
#             chunks.append(chunk)
#     return chunks

# ========================== Prep Embedding Inputs ========================== #
output_folder = 'output'
chunks_to_embed = []
metadata = []
seen_hashes = set()
count_skipped = 0
id_counter = 0

print(" Scanning and chunking files...")

for filename in os.listdir(output_folder):
    if filename.endswith('.json'):
        filepath = os.path.join(output_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                raw_text = " ".join(data.get('article_texts', []) or data.get('text_lines', []) or data.get('content', []))
                clean_text = clean_for_embedding(raw_text)
                if not clean_text.strip():
                    count_skipped += 1
                    continue
                chunks = chunk_text(clean_text, max_words=150)
                # chunks = chunk_text(clean_text, max_words=150, overlap=20)
                for chunk in chunks:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    chunk_hash = md5(chunk.encode()).hexdigest()
                    if chunk_hash in seen_hashes:
                        continue
                    seen_hashes.add(chunk_hash)

                    chunks_to_embed.append(chunk)
                    metadata.append({
                        'id': id_counter,
                        'file': filename,
                        'url': data.get('url'),
                        'title': data.get('title'),
                        'meta_description': data.get('meta', {}).get('description'),
                        'chunk_text': chunk,
                        'tags': infer_tags(chunk)
                    })
                    id_counter += 1

            except Exception as e:
                print(f" Error processing {filename}: {e}")
                count_skipped += 1

print(f" Prepared {len(chunks_to_embed)} unique text chunks from {output_folder}.")

# ========================== Batched Embedding ========================== #
print(" Embedding text chunks...")
print(f"Total chunks to embed: {len(chunks_to_embed)}")
start = time.time()
embeddings = model.encode(
    chunks_to_embed,
    batch_size=512,
    normalize_embeddings=True,
    show_progress_bar=True
)
print(f"Embedding only took: {time.time() - start:.2f} seconds")
# ========================== FAISS Vector Index ========================== #
print(" Creating FAISS index...")

dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)  # For cosine similarity (with normalized embeddings)
index = faiss.IndexIDMap(faiss_index)

ids = np.array([m['id'] for m in metadata])
index.add_with_ids(np.array(embeddings).astype('float32'), ids)

# ========================== Save Index and Metadata ========================== #
print(" Saving index and metadata...")
os.makedirs('vector_index', exist_ok=True)
faiss.write_index(index, 'vector_index/faiss_index.idx')

with open('vector_index/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\n Done! {len(embeddings)} vectors saved.")
print(f" Skipped {count_skipped} files or empty entries.")
