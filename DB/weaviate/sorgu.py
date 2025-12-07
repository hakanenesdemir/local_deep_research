import weaviate
from sentence_transformers import SentenceTransformer
import time

# Weaviate'e bağlan
client = weaviate.connect_to_local()

print("✓ Weaviate'e bağlanıldı")

# Collection'ı al
collection = client.collections.get("Documents")

# Embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

# Kayıt sayısını kontrol et
def check_record_count():
    agg = collection.aggregate.over_all(total_count=True)
    return agg.total_count

# Vector arama
def search_near_vector(query_text, limit=5):
    # Query embedding'i oluştur
    query_vector = model.encode(query_text).tolist()
    
    try:
        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
        )
        return results.objects
    except Exception as e:
        print(f"✗ Arama hatası: {e}")
        return []

# Text tabanlı arama (BM25)
def search_bm25(query_text, limit=5):
    try:
        results = collection.query.bm25(
            query=query_text,
            limit=limit,
            return_metadata=weaviate.classes.query.MetadataQuery(score=True)
        )
        return results.objects
    except Exception as e:
        print(f"✗ Arama hatası: {e}")
        return []

# Hybrid arama (Vector + BM25)
def search_hybrid(query_text, limit=5):
    # Query embedding'i oluştur
    query_vector = model.encode(query_text).tolist()
    
    try:
        results = collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            limit=limit,
            return_metadata=weaviate.classes.query.MetadataQuery(score=True)
        )
        return results.objects
    except Exception as e:
        print(f"✗ Arama hatası: {e}")
        return []

# İlk N kaydı göster
def show_first_documents(limit=5):
    try:
        results = collection.query.fetch_objects(limit=limit)
        return results.objects
    except Exception as e:
        print(f"✗ Veri gösterme hatası: {e}")
        return []

# Ana fonksiyon
def main():
    try:
        # Kayıt sayısını kontrol et
        count = check_record_count()
        print(f"✓ Veritabanında {count} adet kayıt bulunmaktadır\n")
        
        if count == 0:
            print("⚠ Veritabanında veri yok! Önce write_vector_database.py çalıştırın.")
            return
        
        # Test sorguları
        queries = [
            "artificial intelligence healthcare",
            "machine learning medical diagnosis",
            "deep learning neural networks"
        ]
        
        for query in queries:
            print("="*60)
            print(f"SORGU: '{query}'")
            print("="*60)
            
            # Vector Search
            print("\n📊 VECTOR SEARCH (Semantic):")
            print("-"*60)
            start = time.time()
            results = search_near_vector(query, limit=3)
            search_time = time.time() - start
            print(f"Arama zamanı: {search_time:.4f}s\n")
            
            for idx, result in enumerate(results, 1):
                distance = result.metadata.distance if hasattr(result.metadata, 'distance') else 0
                similarity = 1 - distance
                print(f"{idx}. Sonuç (Benzerlik: {similarity:.4f})")
                print(f"   Doc ID: {result.properties['doc_id']} | Chunk: {result.properties['chunk_id']}")
                print(f"   Filename: {result.properties['filename']}")
                print(f"   Metin: {result.properties['metin'][:150]}...\n")
            
            # BM25 Search
            print("📊 BM25 SEARCH (Keyword):")
            print("-"*60)
            start = time.time()
            results = search_bm25(query, limit=3)
            search_time = time.time() - start
            print(f"Arama zamanı: {search_time:.4f}s\n")
            
            for idx, result in enumerate(results, 1):
                score = result.metadata.score if hasattr(result.metadata, 'score') else 0
                print(f"{idx}. Sonuç (Score: {score:.4f})")
                print(f"   Doc ID: {result.properties['doc_id']} | Chunk: {result.properties['chunk_id']}")
                print(f"   Filename: {result.properties['filename']}")
                print(f"   Metin: {result.properties['metin'][:150]}...\n")
            
            # Hybrid Search
            print("📊 HYBRID SEARCH (Vector + BM25):")
            print("-"*60)
            start = time.time()
            results = search_hybrid(query, limit=3)
            search_time = time.time() - start
            print(f"Arama zamanı: {search_time:.4f}s\n")
            
            for idx, result in enumerate(results, 1):
                score = result.metadata.score if hasattr(result.metadata, 'score') else 0
                print(f"{idx}. Sonuç (Hybrid Score: {score:.4f})")
                print(f"   Doc ID: {result.properties['doc_id']} | Chunk: {result.properties['chunk_id']}")
                print(f"   Filename: {result.properties['filename']}")
                print(f"   Metin: {result.properties['metin'][:150]}...\n")
            
            print("\n")
        
        # İlk 5 kaydı göster
        print("="*60)
        print("İLK 5 KAYIT")
        print("="*60)
        first_records = show_first_documents(limit=5)
        for idx, result in enumerate(first_records, 1):
            print(f"\n{idx}. Doc ID: {result.properties['doc_id']} | Chunk: {result.properties['chunk_id']}")
            print(f"   Filename: {result.properties['filename']}")
            print(f"   Metin: {result.properties['metin'][:200]}...")
            
    finally:
        client.close()
        print("\n✓ Weaviate bağlantısı kapatıldı")

if __name__ == "__main__":
    main()