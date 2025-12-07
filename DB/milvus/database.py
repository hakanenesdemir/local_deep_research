from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import time

# Milvus client oluştur
client = MilvusClient("milvus_demo.db")

print("✓ Milvus client oluşturuldu")

# Embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

# Kayıt sayısını kontrol et
def check_record_count():
    try:
        stats = client.get_collection_stats(collection_name="documents")
        return stats['row_count']
    except Exception as e:
        print(f"✗ Kayıt sayısı kontrol hatası: {e}")
        return 0

# Vector arama
def search_vectors(query_text, limit=5):
    # Query embedding'i oluştur
    query_vector = model.encode(query_text).tolist()
    
    try:
        results = client.search(
            collection_name="documents",
            data=[query_vector],
            limit=limit,
            output_fields=["metin", "chunk_id", "doc_id", "filename", "filepath"]
        )
        return results[0] if results else []
    except Exception as e:
        print(f"✗ Arama hatası: {e}")
        return []

# Metadata ile filtrelenmiş arama
def search_with_filter(query_text, doc_id=None, limit=5):
    # Query embedding'i oluştur
    query_vector = model.encode(query_text).tolist()
    
    try:
        filter_expr = f"doc_id == {doc_id}" if doc_id else ""
        
        results = client.search(
            collection_name="documents",
            data=[query_vector],
            limit=limit,
            filter=filter_expr,
            output_fields=["metin", "chunk_id", "doc_id", "filename", "filepath"]
        )
        return results[0] if results else []
    except Exception as e:
        print(f"✗ Filtrelenmiş arama hatası: {e}")
        return []

# Tüm kayıtları sorgula (query)
def query_documents(limit=5):
    try:
        results = client.query(
            collection_name="documents",
            filter="",
            output_fields=["metin", "chunk_id", "doc_id", "filename", "filepath"],
            limit=limit
        )
        return results
    except Exception as e:
        print(f"✗ Query hatası: {e}")
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
            print("\n📊 VECTOR SEARCH (Cosine Similarity):")
            print("-"*60)
            start = time.time()
            results = search_vectors(query, limit=5)
            search_time = time.time() - start
            print(f"Arama zamanı: {search_time:.4f}s\n")
            
            for idx, result in enumerate(results, 1):
                distance = result.get('distance', 0)
                similarity = 1 - distance  # Cosine similarity
                print(f"{idx}. Sonuç (Benzerlik: {similarity:.4f})")
                print(f"   Doc ID: {result['entity']['doc_id']} | Chunk: {result['entity']['chunk_id']}")
                print(f"   Filename: {result['entity']['filename']}")
                print(f"   Metin: {result['entity']['metin'][:150]}...\n")
            
            print("\n")
        
        # Belirli bir döküman içinde arama
        print("="*60)
        print("BELİRLİ DÖKÜMANDA ARAMA (Doc ID: 0)")
        print("="*60)
        query = "artificial intelligence"
        start = time.time()
        results = search_with_filter(query, doc_id=0, limit=3)
        search_time = time.time() - start
        print(f"Sorgu: '{query}'")
        print(f"Arama zamanı: {search_time:.4f}s\n")
        
        for idx, result in enumerate(results, 1):
            distance = result.get('distance', 0)
            similarity = 1 - distance
            print(f"{idx}. Sonuç (Benzerlik: {similarity:.4f})")
            print(f"   Doc ID: {result['entity']['doc_id']} | Chunk: {result['entity']['chunk_id']}")
            print(f"   Filename: {result['entity']['filename']}")
            print(f"   Metin: {result['entity']['metin'][:150]}...\n")
        
        # İlk 5 kaydı göster
        print("="*60)
        print("İLK 5 KAYIT")
        print("="*60)
        first_records = query_documents(limit=5)
        for idx, result in enumerate(first_records, 1):
            print(f"\n{idx}. Doc ID: {result['doc_id']} | Chunk: {result['chunk_id']}")
            print(f"   Filename: {result['filename']}")
            print(f"   Metin: {result['metin'][:200]}...")
            
    except Exception as e:
        print(f"✗ Hata: {e}")
    finally:
        print("\n✓ İşlem tamamlandı")

if __name__ == "__main__":
    main()