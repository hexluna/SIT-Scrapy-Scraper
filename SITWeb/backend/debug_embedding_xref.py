import os
import json

# Paths
summary_folder = 'output'
metadata_file = 'vector_index/metadata.json'

# Load indexed filenames from metadata
with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)
indexed_files = set(entry['file'] for entry in metadata)

# List all original .json files
all_files = set(filename for filename in os.listdir(summary_folder) if filename.endswith('.json'))

# Find files that were NOT included in metadata / FAISS index
missing_files = sorted(all_files - indexed_files)

# Report
if missing_files:
    print("❌ These files were not embedded or stored in FAISS:")
    for filename in missing_files:
        print(f" - {filename}")
else:
    print("✅ All files were successfully embedded and stored in the FAISS index.")
