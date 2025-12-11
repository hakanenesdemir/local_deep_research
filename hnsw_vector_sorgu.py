import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List

DB_PATH = "/home/ugo/Documents/Python/bitirememe projesi/DB/chorame/yerel_veritabani"
COLLECTION_NAME = "dokumanlarim"
MODEL_NAME = "all-MiniLM-L6-v2"

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

def hnsw_vector_search(queries: List[str], top_k: int = 5):
    model = SentenceTransformer(MODEL_NAME)
    collection = get_collection()
    print(f"✓ Koleksiyon: {collection.name} | Kayıt sayısı: {collection.count()}")
    for q in queries:
        q_vec = model.encode(q).tolist()
        results = collection.query(query_embeddings=[q_vec], n_results=top_k, include=["documents", "metadatas", "distances"])
        print(f"\nSorgu: '{q}'")
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ), 1):
            sim = 1 - dist
            print(f"  {i}. Benzerlik: {sim:.4f} | Doc ID: {meta.get('doc_id')} | Chunk: {meta.get('chunk_id')} | Dosya: {meta.get('filename')}")
            print(f"     Metin: {doc[:180]}...")

if __name__ == "__main__":
    sample_queries = [
        "artificial intelligence healthcare",
        "machine learning medical diagnosis",
        "deep learning neural networks",
    ]
    hnsw_vector_search(sample_queries, top_k=5)