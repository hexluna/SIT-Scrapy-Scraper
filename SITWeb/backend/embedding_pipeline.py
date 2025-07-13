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
                raw_metadata = json.load(f)

            self.metadata = raw_metadata

            def compute_hash_and_id(meta):
                chunk = meta.get('chunk_text', '')
                return md5(chunk.encode()).hexdigest(), meta.get('id', -1)

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(tqdm(executor.map(compute_hash_and_id, raw_metadata),
                                    total=len(raw_metadata), desc="Loading metadata"))

            self.hashes = {h for h, _ in results}
            ids = [i for _, i in results if i is not None]
            if ids:
                self.id_counter = max(ids) + 1

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
                self.tokenizer.encode(para, add_special_tokens=False, truncation=True))

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
            "please note", "privacy policy","©", "license",
            "open access", "peer-review", "CC BY-NC-ND"
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

    @staticmethod
    def split_text_into_token_chunks(text, tokenizer, max_tokens=512, min_chunk_tokens=100):
        tokens = tokenizer.encode(text, add_special_tokens=False)

        print(f"[Chunking] Tokenizing... Total tokens: {len(tokens)}")
        chunks = []
        buffer = []

        # Chunk based on token limit
        for token in tqdm(tokens, desc="Splitting into max_token chunks"):
            buffer.append(token)
            if len(buffer) >= max_tokens:
                chunks.append(buffer)
                buffer = []

        if buffer:
            chunks.append(buffer)

        print(f"[Chunking] Initial chunks: {len(chunks)}")

        # Merge small chunks if under min_chunk_tokens
        merged_chunks = []
        for chunk in tqdm(chunks, desc="Merging small chunks"):
            if len(chunk) < min_chunk_tokens and merged_chunks:
                merged_chunks[-1].extend(chunk)
            else:
                merged_chunks.append(chunk)

        print(f"[Chunking] Final merged chunks: {len(merged_chunks)}")

        decoded_chunks = []
        for chunk in tqdm(merged_chunks, desc="Decoding token chunks"):
            decoded_chunks.append(tokenizer.decode(chunk))

        return decoded_chunks

    def _get_summary_for_file(self, filename):
        for entry in self.metadata:
            if entry["file"] == filename and entry["chunk_text"].startswith("[SUMMARY]"):
                return entry["chunk_text"].replace("[SUMMARY] ", "")
        return "Summary not found."

    def _generate_summary(self, pdf_path):
        from chatbot_caching import llm
        from transformers import AutoTokenizer
        from PyPDF2 import PdfReader
        import os
        import re
        from hashlib import md5

        filename = os.path.basename(pdf_path)

        # Check if already processed
        already_processed = any(meta["file"] == filename for meta in self.metadata)
        if already_processed:
            print(f"[Ingest] Skipping: {filename} already in metadata.")
            summary = self._get_summary_for_file(filename)
            yield summary
            return False, summary

        reader = PdfReader(pdf_path)
        raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        raw_text = self.remove_boilerplate(raw_text)
        raw_text = self.remove_consecutive_duplicates(raw_text)
        clean = self.clean_text(raw_text)

        # ===================== Extract Abstract / conclusion ===================== #
        abstract = ""
        conclusion = ""
        abstract_match = re.search(r"\babstract\b[:\n]*([\s\S]{200,2000})", clean, re.IGNORECASE)
        if abstract_match:
            abstract = abstract_match.group(1).strip()

        conclusion_match = re.search(r"\bconclusion[s]?\b[:\n]*([\s\S]{200,2000})", clean, re.IGNORECASE)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()

        summary_text = f"{abstract}\n\n{conclusion}" if abstract and conclusion else clean
        if not summary_text:
            yield "[ERROR]: PDF text is empty after cleaning."
            return

        chunks = self.chunk_text(clean)
        new_chunks, new_metadata = [], []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            chunk_hash = md5(chunk.encode()).hexdigest()
            if chunk_hash in self.hashes:
                continue
            self.hashes.add(chunk_hash)
            meta = {
                "file": filename,
                "url": None,
                "title": filename,
                "meta_description": "User uploaded document",
                "chunk_text": chunk,
                "tags": self.infer_tags(chunk),
                "id": self.id_counter
            }
            self.id_counter += 1
            new_chunks.append(chunk)
            new_metadata.append(meta)

        self.embed_and_update(new_chunks, new_metadata)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        token_chunks = self.split_text_into_token_chunks(summary_text, tokenizer, max_tokens=512)

        print(f"[Summary] Splitting into {len(token_chunks)} optimized chunks...")

        os.makedirs("summaries", exist_ok=True)

        chunk_summaries = []
        yield "[Generating summary...]\n"
        for i, chunk in enumerate(token_chunks):
            try:
                prompt = f"You are a helpful assistant. Summarize the following section:\n\n{chunk}\n\nSummary:"
                partial_summary = ""
                for token in llm.create_completion(prompt=prompt, max_tokens=256, stream=True):
                    text = token["choices"][0]["text"]
                    partial_summary += text
                chunk_summaries.append(partial_summary.strip())
            except Exception as e:
                yield f"[WARN] Skipping chunk {i} due to error: {str(e)}\n"
                continue

        combined = " ".join(chunk_summaries)
        final_summary = ""
        for token in llm.create_completion(
            prompt=f"You are a summarization expert. Summarize the following {len(chunk_summaries)} section summaries "
                    f"into 3–5 plain English sentences:\n\n{combined}\n\nFinal Summary:",
            max_tokens=512,
            stream=True
        ):
            text = token["choices"][0]["text"]
            final_summary += text
            yield text

        summary_meta = {
            "file": filename,
            "chunk_text": f"[SUMMARY] {final_summary.strip()}",
            "title": filename,
            "id": self.id_counter,
            "tags": ["summary"],
        }
        self.metadata.append(summary_meta)
        self.faiss_index.add_with_ids(
            np.array(self.model.encode([final_summary], normalize_embeddings=True)).astype("float32"),
            np.array([self.id_counter])
        )
        self.id_counter += 1

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        faiss.write_index(self.faiss_index, self.index_path)

if __name__ == '__main__':
    embedder = IncrementalEmbedder()
    embedder.run()