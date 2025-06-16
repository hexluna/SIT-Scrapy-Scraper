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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


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
        self.lock = Lock()

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
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        final_chunks = []

        for para in paragraphs:
            token_len = len(
                self.tokenizer.encode(para, add_special_tokens=False, max_length=max_tokens, truncation=True))

            if token_len <= max_tokens:
                final_chunks.append(para)
            else:
                sentences = nltk.sent_tokenize(para)
                if not sentences:
                    continue

                sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)

                current_chunk = []
                current_tokens = 0
                current_embedding = None

                for i, sent in enumerate(sentences):
                    sent_embedding = sentence_embeddings[i]
                    sent_token_len = len(
                        self.tokenizer.encode(sent, add_special_tokens=False, max_length=max_tokens, truncation=True))

                    if not current_chunk:
                        current_chunk.append(sent)
                        current_tokens = sent_token_len
                        current_embedding = sent_embedding
                        continue

                    similarity = torch.nn.functional.cosine_similarity(current_embedding, sent_embedding, dim=0).item()

                    if similarity >= 0.6 and current_tokens + sent_token_len <= max_tokens:
                        current_chunk.append(sent)
                        current_tokens += sent_token_len
                        current_embedding = (current_embedding + sent_embedding) / 2
                    else:
                        final_chunks.append(" ".join(current_chunk))
                        current_chunk = [sent]
                        current_tokens = sent_token_len
                        current_embedding = sent_embedding

                if current_chunk:
                    final_chunks.append(" ".join(current_chunk))

        return final_chunks

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

    @staticmethod
    def remove_boilerplate(text):
        boilerplate_phrases = [
            "all rights reserved", "copyright", "disclaimer", "terms and conditions",
            "please note", "contact us", "follow us", "privacy policy"
        ]
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            if any(phrase in line_lower for phrase in boilerplate_phrases):
                continue  # skip boilerplate line
            if len(line.strip()) == 0:
                continue  # skip empty lines
            filtered_lines.append(line)
        return "\n".join(filtered_lines)

    @staticmethod
    def remove_consecutive_duplicates(text):
        lines = text.split('\n')
        new_lines = []
        prev_line = None
        for line in lines:
            line_strip = line.strip()
            if line_strip != prev_line:
                new_lines.append(line)
            prev_line = line_strip
        return "\n".join(new_lines)

    def single_process_files(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        new_chunks = []
        new_metadata = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                raw_text = " ".join(
                    data.get('article_texts', []) or data.get('text_lines', []) or data.get('content', []))
                raw_text_no_boilerplate = self.remove_boilerplate(raw_text)
                text_no_dupes = self.remove_consecutive_duplicates(raw_text_no_boilerplate)
                clean = self.clean_text(text_no_dupes)
                if not clean:
                    return [], []
                chunks = self.chunk_text(clean)
                for chunk in chunks:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    meta = {
                        'file': filename,
                        'url': data.get('url'),
                        'title': data.get('title'),
                        'meta_description': data.get('meta', {}).get('description'),
                        'chunk_text': chunk,
                        'tags': self.infer_tags(chunk)
                    }
                    new_chunks.append(chunk)
                    new_metadata.append(meta)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        print(f"Prepared {len(new_chunks)} new chunks from {filename}.")
        return new_chunks, new_metadata

    def process_files(self):
        new_chunks = []
        new_metadata = []
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.single_process_files, f): f for f in all_files}
            for future in tqdm(as_completed(futures), total=len(all_files), desc="Processing files"):
                chunks, metadata = future.result()
                with self.lock:
                    for chunk, meta in zip(chunks, metadata):
                        chunk_hash = md5(chunk.encode()).hexdigest()
                        if chunk_hash in self.hashes:
                            continue
                        self.hashes.add(chunk_hash)
                        meta['id'] = self.id_counter
                        self.id_counter += 1
                        new_chunks.append(chunk)
                        new_metadata.append(meta)

        print(f"Prepared {len(new_chunks)} new chunks.")
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