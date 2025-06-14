import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from hashlib import md5
import torch
import time
import nltk
from transformers import AutoTokenizer
# nltk.download('punkt')
from tqdm import tqdm


class IncrementalEmbedder:
    def __init__(self,
                 model_name='sentence-transformers/all-MiniLM-L6-v2',
                 data_dir='output',
                 index_dir='vector_index',
                 batch_size=512,
                 chunk_size=450,
                 chunk_overlap=50,
                 chunk_token_limit=512):

        self.model_name = model_name
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.metadata_path = os.path.join(index_dir, 'metadata.json')
        self.index_path = os.path.join(index_dir, 'faiss_index.idx')
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_token_limit = chunk_token_limit
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        os.makedirs(self.index_dir, exist_ok=True)
        self.model = SentenceTransformer(self.model_name, device='cuda')
        self.model.to('cuda')

        self.metadata = []
        self.hashes = set()
        self.id_counter = 0
        self.faiss_index = None

        self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                for m in self.metadata:
                    h = md5(m['chunk_text'].encode()).hexdigest()
                    self.hashes.add(h)
                if self.metadata:
                    self.id_counter = max(m['id'] for m in self.metadata) + 1
        if os.path.exists(self.index_path):
            self.faiss_index = faiss.read_index(self.index_path)

        print(f"Loaded {len(self.metadata)} existing chunks.")

    @staticmethod
    def clean_text(text):
        return text.replace('\n', ' ').strip()

    def chunk_text(self, text):
        max_tokens = self.chunk_token_limit
        #overlap = self.chunk_overlap
        sentences = nltk.sent_tokenize(text)
        chunks, current_chunk = [], []
        current_len = 0

        for sent in sentences:
            sent_len = len(self.tokenizer.encode(sent, add_special_tokens=False, max_length=max_tokens, truncation=True))
            if current_len + sent_len <= self.chunk_token_limit:
                current_chunk.append(sent)
                current_len += sent_len
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_len = sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    @staticmethod
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

    def process_files(self):
        new_chunks = []
        new_metadata = []
        count_skipped = 0
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        for filename in tqdm(all_files, desc="Processing files"):
            filepath = os.path.join(self.data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    raw_text = " ".join(data.get('article_texts', []) or data.get('text_lines', []) or data.get('content', []))
                    clean = self.clean_text(raw_text)
                    if not clean:
                        count_skipped += 1
                        continue
                    chunks = self.chunk_text(clean)
                    for chunk in chunks:
                        chunk = chunk.strip()
                        if not chunk:
                            continue
                        chunk_hash = md5(chunk.encode()).hexdigest()
                        if chunk_hash in self.hashes:
                            continue
                        self.hashes.add(chunk_hash)
                        new_chunks.append(chunk)
                        meta = {
                            'id': self.id_counter,
                            'file': filename,
                            'url': data.get('url'),
                            'title': data.get('title'),
                            'meta_description': data.get('meta', {}).get('description'),
                            'chunk_text': chunk,
                            'tags': self.infer_tags(chunk)
                        }
                        new_metadata.append(meta)
                        self.id_counter += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                count_skipped += 1

        print(f"Prepared {len(new_chunks)} new chunks. Skipped {count_skipped} files.")
        return new_chunks, new_metadata

    def embed_and_update(self, new_chunks, new_metadata):
        if not new_chunks:
            print("No new data to embed.")
            return

        print("Embedding...")
        start = time.time()
        embeddings = self.model.encode(
            new_chunks,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        print(f"Embedding took {time.time() - start:.2f}s")

        if self.faiss_index is None:
            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            print("Created new FAISS index.")

        ids = np.array([m['id'] for m in new_metadata])
        self.faiss_index.add_with_ids(np.array(embeddings).astype('float32'), ids)

        self.metadata += new_metadata
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        faiss.write_index(self.faiss_index, self.index_path)
        print(f"Added {len(new_chunks)} new vectors.")

    def run(self):
        new_chunks, new_metadata = self.process_files()
        self.embed_and_update(new_chunks, new_metadata)


if __name__ == '__main__':
    embedder = IncrementalEmbedder()
    embedder.run()