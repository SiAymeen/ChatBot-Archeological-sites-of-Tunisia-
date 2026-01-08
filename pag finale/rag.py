from sentence_transformers import SentenceTransformer
import chromadb
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load ChromaDB client
client = chromadb.PersistentClient(path="./db")
collection = client.get_collection(name="archaeological_sites")

# Prompt template
template = """
You are an expert in archaeology, particularly focusing on the historical and archaeological sites in Tunisia.

Use the following context to answer the question. If no relevant information is in the context, say "Je ne dispose pas d'information fiable sur ce point."

Context:
{context}

Question: {question}

Generate a response in this structured format:
**Réponse** : [texte clair et précis]
**Sources** :
- [Site] ([Delegation]): [source]
- ...
"""
prompt = PromptTemplate.from_template(template)
llm = OllamaLLM(model="llama3")

def query_rag(user_query, top_k=3, similarity_threshold=0.5):
    # Embed the query
    query_embedding = model.encode([user_query])[0]

    # Retrieve from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Filter by similarity (cosine distance < 1 - threshold)
    filtered_docs = []
    filtered_metadatas = []
    for dist, doc, meta in zip(results['distances'][0], results['documents'][0], results['metadatas'][0]):
        if dist < (1 - similarity_threshold):
            filtered_docs.append(doc)
            filtered_metadatas.append(meta)

    if not filtered_docs:
        return "Je ne dispose pas d'information fiable sur ce point.", []

    # Build context
    retrieved_context = "\n\n".join(filtered_docs)

    # Generate response
    response = llm.invoke(prompt.format(context=retrieved_context, question=user_query))
    response_text = response if isinstance(response, str) else response.content

    return response_text, filtered_metadatas