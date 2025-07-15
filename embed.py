import faiss
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer
import json
import os

# Initialize tokenizer and embedding model
enc = tiktoken.encoding_for_model("gpt-4")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
index = faiss.IndexFlatL2(dimension)
metadatas = []  # To store document metadata

def chunk(text, max_tokens=350):
    tokens = enc.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield enc.decode(tokens[i:i+max_tokens])

def embed(text):
    # Generate embeddings using Hugging Face model
    # Ensure input is a single string and output is a 2D array
    embedding = model.encode([text], convert_to_numpy=True)
    return embedding  # Shape: (1, dimension)

def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    for idx, part in enumerate(chunk(raw)):
        yield idx, part

def save_to_faiss(data_folder="data", index_file="faiss_index.bin", meta_file="faiss_metadata.json"):
    # Process all files in the data folder
    for doc in os.listdir(data_folder):
        doc_path = os.path.join(data_folder, doc)
        if os.path.isfile(doc_path):
            for idx, chunk_txt in load_file(doc_path):
                emb = embed(chunk_txt)
                # Add to FAISS index (emb is already (1, dimension))
                index.add(emb.astype(np.float32))
                # Store metadata
                metadatas.append({
                    'doc_id': doc,
                    'chunk_index': idx,
                    'content': chunk_txt
                })
    
    # Save FAISS index
    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, os.path.join("vectorstore", index_file))
    
    # Save metadata
    with open(os.path.join("vectorstore", meta_file), 'w', encoding='utf-8') as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Process all files in the data folder
    data_folder = "data"
    index_path = "faiss_index.bin"
    meta_path = "faiss_metadata.json"
    save_to_faiss(data_folder, index_path, meta_path)