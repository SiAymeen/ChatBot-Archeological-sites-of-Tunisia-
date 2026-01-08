import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os

# Step 1: Load the CSV file
file_path = 'data/Merged_Archaeological_Sites_Data.csv'
data = pd.read_csv(file_path)

# Step 2: Prepare texts by combining relevant columns for richer context
texts = []
metadatas = []
for _, row in data.iterrows():
    text = f"{row['Site']} located in {row['Location']}, {row['Delegation']}. Historical Context: {row['Historical Context']}. Key Features: {row['Key Features']}."
    texts.append(text)
    metadatas.append({
        "site": row['Site'],
        "delegation": row['Delegation'],
        "source": row['Source'],
    })

# Step 3: Initialize the model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to chunk text into smaller parts (with overlap)
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

# Step 4: Apply chunking to the texts
all_chunks = []
all_embeddings = []
all_chunk_metadatas = []
all_ids = []
for idx, text in enumerate(texts):
    site_chunks = chunk_text(text)
    site_embeddings = model.encode(site_chunks)
    for j, chunk in enumerate(site_chunks):
        all_chunks.append(chunk)
        all_embeddings.append(site_embeddings[j])
        all_chunk_metadatas.append(metadatas[idx])  # Same metadata for all chunks of a site
        all_ids.append(f"doc_{idx}_{j}")

# Step 5: Initialize ChromaDB with persistence (this creates ./db)
client = chromadb.PersistentClient(path="./db")

# Delete existing collection if needed
collection_name = "archaeological_sites"
if collection_name in [c.name for c in client.list_collections()]:
    print(f"Collection '{collection_name}' already exists. Deleting it...")
    client.delete_collection(name=collection_name)

# Create a new collection with cosine similarity
collection = client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

# Step 6: Add the chunks, embeddings, and metadatas to ChromaDB
collection.add(
    documents=all_chunks,
    metadatas=all_chunk_metadatas,
    ids=all_ids,
    embeddings=all_embeddings
)

print("Data has been indexed in ChromaDB!")
print("db folder created at:", "./db")
if os.path.exists("./db"):
    print("Contents:", os.listdir("./db"))