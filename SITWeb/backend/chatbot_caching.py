# chatbot_with_cache_and_timing.py
import os
import json
import sys
import time
import re
import faiss
import numpy as np
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
# ========================== CrossEncoder ========================== #
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ========================== Config ========================== #
EMBED_CACHE_PATH = "embedding_cache.json"
MAX_CACHE_SIZE = 1000  # Set None for unlimited
embedding_cache = {}

# ========================== Load Cache ========================== #
if os.path.exists(EMBED_CACHE_PATH):
    try:
        with open(EMBED_CACHE_PATH, "r", encoding="utf-8") as f:
            raw_cache = json.load(f)
            embedding_cache = {k: np.array(v, dtype="float32") for k, v in raw_cache.items()}
            print(f"[Cache] Loaded {len(embedding_cache)} entries.")
    except Exception as e:
        print(f"[Cache] Failed to load cache: {e}")

# ========================== Load Vector Store ========================== #
print("Loading vector index and metadata...")
index = faiss.read_index("vector_index/faiss_index.idx")

with open("vector_index/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ========================== Embedding Model ========================== #
print("Loading embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
# ========================== MMR Reranking ========================== #
def mmr(query_embedding, doc_embeddings, mmr_k=8, lambda_param=0.5):
    selected = []
    doc_indices = list(range(len(doc_embeddings)))
    sim_to_query = cosine_similarity([query_embedding], doc_embeddings)[0]
    sim_matrix = cosine_similarity(doc_embeddings)

    for _ in range(mmr_k):
        if not doc_indices:
            break
        if not selected:
            idx = np.argmax(sim_to_query)
            selected.append(idx)
            doc_indices.remove(idx)
            continue

        mmr_scores = []
        for i in doc_indices:
            relevance = sim_to_query[i]
            diversity = max([sim_matrix[i][j] for j in selected])
            score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((i, score))

        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_idx)
        doc_indices.remove(best_idx)

    return selected
# ========================== Cohere Reranking ========================== #
def crossencoder_rerank(query, passages, top_n=4):
    if isinstance(passages[0], dict):
        passages_text = [p["text"] for p in passages]
    else:
        passages_text = passages

    inputs = tokenizer(
        [query] * len(passages),
        passages,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[:, 0].tolist()

    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [p for p, _ in ranked[:top_n]]

# ========================== Retriever ========================== #
def deduplicate_chunks(chunks, threshold=0.92):
    if len(chunks) <= 1:
        return chunks

    embeddings = embedding_model.encode(chunks, normalize_embeddings=True)
    keep = []
    seen = set()

    for i in range(len(chunks)):
        if i in seen:
            continue
        keep.append(chunks[i])
        for j in range(i + 1, len(chunks)):
            if j in seen:
                continue
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim > threshold:
                seen.add(j)
    return keep

def retrieve_relevant_chunks(query, k=15, mmr_k=8, lambda_param=0.5, rerank_top=4):
    embed_start = time.time()
    if query in embedding_cache:
        query_vec = embedding_cache[query]
        print(f"[Cache] Hit for: \"{query}\"")
    else:
        print(f"[Cache] Miss for: \"{query}\" → encoding")
        query_vec = embedding_model.encode([query], normalize_embeddings=True)[0].astype("float32")
        if MAX_CACHE_SIZE is None or len(embedding_cache) < MAX_CACHE_SIZE:
            embedding_cache[query] = query_vec
    print(f"[Timing] Embedding: {time.time() - embed_start:.2f}s")

    print("Searching FAISS index...")
    search_start = time.time()
    D, I = index.search(np.array([query_vec]), k)
    print(f"[Timing] FAISS search: {time.time() - search_start:.2f}s")
    doc_embeddings = []
    valid_metadata = []
    for idx in I[0]:
        if idx == -1:
            continue
        # entry = metadata[idx]
        # url = entry.get("url", "")
        # tag = entry.get("tag", [])
        # title = entry.get("url", "")
        chunk = metadata[idx]["chunk_text"]
        doc_embed = embedding_model.encode(chunk, normalize_embeddings=True)
        doc_embeddings.append(doc_embed)
        valid_metadata.append(chunk)
        # valid_metadata.append({"text": chunk, "url": url, "tags": tag, "title": title})


    if not doc_embeddings:
        return []
    doc_embeddings = np.vstack(doc_embeddings)
    selected_indices = mmr(query_vec, doc_embeddings, mmr_k, lambda_param=lambda_param)
    mmr_passages = [valid_metadata[i] for i in selected_indices]
    final_chunks = crossencoder_rerank(query, mmr_passages, top_n=rerank_top)
    final_results = []
    # for chunk in final_chunks:
    #     for entry in mmr_passages:
    #         if entry["text"] == chunk:
    #             final_results.append(entry)
    #             break
    return final_chunks

# ========================== llama.cpp Model ========================== #
model_name = "capybarahermes-2.5-mistral-7b.Q4_K_M.gguf"
model_url = f"https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/resolve/main/{model_name}"
model_path = "."
model_full_path = os.path.join(model_path, model_name)

if not os.path.exists(model_full_path):
    print("Model file not found. Downloading model...")
    response = requests.get(model_url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    with open(model_full_path, 'wb') as f, tqdm(
        desc=model_name, total=total_size, unit='B', unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print("Download complete.")

llm = Llama(
    model_path=model_full_path,
    allow_download=False,
    n_gpu_layers=-1,
    n_ctx=4096,
    use_mlock=True,
    use_mmap=True,
    verbose=False,
)

# ========================== Predefined Intents ========================== #
INTENT_RESPONSES = {
    "greeting": {
        "phrases": ["hello", "hi", "hey", "good morning", "good afternoon"],
        "response": "Hello! How can I assist you with information about SIT today?",
    },
    "identity": {
        "phrases": ["who are you", "what is your name", "identify yourself"],
        "response": "I'm a virtual assistant here to help you with questions about the Singapore Institute of Technology (SIT).",
    },
    "capabilities": {
        "phrases": ["what can you do", "how can you help", "what are your functions"],
        "response": "I can help answer questions about SIT, including courses, admissions, student life, and campus facilities.",
    },
    "help": {
        "phrases": ["help", "i need help", "assist me"],
        "response": "Feel free to ask me anything related to SIT — courses, campus life, admissions, and more.",
    }
}

# ========================== Chat Function ========================== #
def chunk_text(text, max_chars=3000):
    for i in range(0, len(text), max_chars):
        yield text[i:i+max_chars]

def ask_chatbot(query):
    cleaned_query = query.lower().strip()

    for intent in INTENT_RESPONSES.values():
        for phrase in intent["phrases"]:
            if phrase in cleaned_query:
                yield intent["response"]
                return

    def clean_qa_format(text):
        lines = text.splitlines()
        return " ".join([
            line for line in lines
            if not re.match(r"^(Question:|Answer:)", line.strip())
            and not re.match(r"^What .*[\?？]$", line.strip())
        ]).strip()

    print("Retrieving relevant context...")
    retrieval_start = time.time()
    context_chunks = retrieve_relevant_chunks(query)
    deduped_chunks = deduplicate_chunks(context_chunks)
    filtered_chunks = [clean_qa_format(chunk) for chunk in deduped_chunks]
    full_context = "\n\n".join(filtered_chunks)
    print(f"[Timing] Retrieval: {time.time() - retrieval_start:.2f}s")
    max_prompt_tokens = 4096 - 1024
    full_context = full_context[:max_prompt_tokens * 4]
    full_output = ""
    # Prompt prep
    prompt_start = time.time()
    prompt = f"""You are an intelligent virtual assistant stationed at the SIT (Singapore Institute of Technology) Information Center. 
Your job is to assist users by answering any questions they have about SIT. This includes topics like courses, admissions, campus facilities, events, student life, and academic programs. 
Always speak in plain, friendly English. Never mimic a Q&A format.
If the user asks about your role, you can respond that you are an SIT chatbot here to help with information about the university.
If the answer to a question is not in the context or not related to SIT, respond with "I'm sorry, I can only answer questions about SIT.
If providing a website link, always use the full URL format (e.g., https://www.singaporetech.edu.sg/) so it can be clicked and ensure the URL domain name is correct (e.g., singaporetech.edu.sg).

Context:
{full_context}


The user asked: "{query}"
Respond with a helpful, plain-sentence explanation below:
"""
    print(f"[Timing] Prompt prep: {time.time() - prompt_start:.2f}s")
    print("Generating answer...")

    print("Bot: ", end="", flush=True)
    full_output = ""

    for token in llm.create_completion(
            prompt=prompt,
            max_tokens=1024,
            stream=True,
            stop=["\nYou:", "\nThe user asked:", "</s>", "###"]
    ):
        curr_text = token['choices'][0]['text']
        if curr_text:
            print(curr_text, end='', flush=True)  # still see it stream in terminal
            full_output += curr_text
            yield curr_text

    print("\n[DEBUG] Streaming complete.")

# ========================== Chat Loop ========================== #
if __name__ == "__main__":
    print("\nChatbot ready! Type 'exit' to quit.\n")
    print("Bot: Hello! How can I assist you with information about SIT today?\n")

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            response = ask_chatbot(user_input)
            if response is not None:
                print(f"Bot: {response}\n")
    finally:
        with open(EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({k: v.tolist() for k, v in embedding_cache.items()}, f, indent=2)
            print(f"[Cache] Saved {len(embedding_cache)} embeddings.")
        del llm