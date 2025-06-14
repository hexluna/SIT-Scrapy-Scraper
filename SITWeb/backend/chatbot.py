# chatbot_with_cache_and_timing.py
import os
import json
import time
import re
import faiss
import numpy as np
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

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

# ========================== Retriever ========================== #
def retrieve_relevant_chunks(query, k=10):
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

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        score = float(score)
        results.append((score, metadata[idx]))
    results.sort(reverse=True, key=lambda x: x[0])
    return [entry[1]["chunk_text"] for entry in results[:4]]

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

# llm = None  # Internal lazy-initialized model

# def get_llm():
#     global llm
#     if llm is None:
#         print("[LLM] Lazy-loading model...")
#         llm = Llama(
#             model_path=model_full_path,
#             allow_download=False,
#             n_gpu_layers=-1,
#             n_ctx=4096,
#             use_mlock=True,
#             use_mmap=True,
#             verbose=False,
#         )
#         print("[LLM] Model loaded.")
#     return llm

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
def ask_chatbot(query):
    total_start = time.time()
    cleaned_query = query.lower().strip()

    def clean_qa_format(text):
        lines = text.splitlines()
        return " ".join([
            line for line in lines
            if not re.match(r"^(Question:|Answer:)", line.strip())
            and not re.match(r"^What .*[\?？]$", line.strip())
        ]).strip()

    # Intent detection
    intent_start = time.time()
    for intent in INTENT_RESPONSES.values():
        for phrase in intent["phrases"]:
            if phrase in cleaned_query:
                print(f"[Timing] Intent check: {time.time() - intent_start:.2f}s")
                print(f"[Timing] Total: {time.time() - total_start:.2f}s")
                return intent["response"]

    print("Retrieving relevant context...")
    retrieval_start = time.time()
    context_chunks = retrieve_relevant_chunks(query)
    filtered_chunks = [clean_qa_format(chunk) for chunk in context_chunks]
    context = "\n\n".join(filtered_chunks)
    print(f"[Timing] Retrieval: {time.time() - retrieval_start:.2f}s")

    # Prompt prep
    prompt_start = time.time()
    prompt = f"""You are an intelligent virtual assistant stationed at the SIT (Singapore Institute of Technology) Information Center. 
Your job is to assist users by answering any questions they have about SIT. This includes topics like courses, admissions, campus facilities, events, student life, and academic programs. 
Always speak in plain, friendly English. Never mimic a Q&A format.
If the user asks about your role, you can respond that you are an SIT chatbot here to help with information about the university.
If the answer to a question is not in the context or not related to SIT, respond with "I'm sorry, I can only answer questions about SIT.
If providing a website link, always use the full URL format (e.g., https://www.sitlearn.singaporetech.edu.sg) so it can be clicked.

Context:
{context}

The user asked: "{query}"
Respond with a helpful, plain-sentence explanation below:
"""
    print(f"[Timing] Prompt prep: {time.time() - prompt_start:.2f}s")

    # LLM response
    print("Generating answer...")
    llm_start = time.time()
    response_text = "" 
    for token in llm.create_completion(prompt=prompt, max_tokens=1024, stream=True,stop=["\nYou:", "\nThe user asked:", "</s>", "###"]):
        chunk = token['choices'][0]['text']
        print(token['choices'][0]['text'], end='', flush=True)
        response_text += chunk
    print(f"\n[Timing] LLM generation: {time.time() - llm_start:.2f}s")
    print(f"[Timing] Total: {time.time() - total_start:.2f}s")
    return response_text

# ========================== Chat Loop ========================== #
# print("\nChatbot ready! Type 'exit' to quit.\n")
# print("Bot: Hello! How can I assist you with information about SIT today?\n")

# try:
#     while True:
#         user_input = input("\nYou: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Goodbye!")
#             break
#         response = ask_chatbot(user_input)
#         if response is not None:
#             print(f"Bot: {response}\n")
# finally:
#     with open(EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
#         json.dump({k: v.tolist() for k, v in embedding_cache.items()}, f, indent=2)
#         print(f"[Cache] Saved {len(embedding_cache)} embeddings.")



def run_cli():
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

if __name__ == "__main__":
    run_cli()