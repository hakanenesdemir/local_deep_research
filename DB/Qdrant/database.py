from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Arama sorgusu
sorgu = "artificial intelligence in healthcare"
sorgu_vector = model.encode(sorgu).tolist()

# Benzer sonuçları bul (güncel API)
results = client.query_points(
    collection_name="test_collection",
    query=sorgu_vector,
    limit=5,
    with_payload=True
)

for result in results.points:
    print(f"\n--- Skor: {result.score:.4f} ---")
    print(f"Filename: {result.payload.get('filename')}")
    print(f"Text: {result.payload.get('text', '')[:300]}...")