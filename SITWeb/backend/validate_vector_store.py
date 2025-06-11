import json
import faiss
import numpy as np

metadata_path = 'vector_index/metadata.json'
index_path = 'vector_index/faiss_index.idx'

# ========================== Load Metadata ========================== #
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

ids_meta = set(m['id'] for m in metadata)

# ========================== Load Faiss ========================== #
index = faiss.read_index(index_path)
ids_faiss = set(index.id_map.keys())

# ========================== Compare Metadata ========================== #
print(f"Metadata entries: {len(ids_meta)}")
print(f"FAISS entries: {len(ids_faiss)}")

missing_in_faiss = ids_meta - ids_faiss
missing_in_meta = ids_faiss - ids_meta

if not missing_in_faiss and not missing_in_meta:
    print("FAISS index and metadata are fully consistent.")
else:
    print("Inconsistency detected!")
    print("Missing in FAISS:", missing_in_faiss)
    print("Missing in metadata:", missing_in_meta)