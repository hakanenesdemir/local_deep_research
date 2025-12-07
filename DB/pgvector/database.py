import psycopg2
from sentence_transformers import SentenceTransformer
import time

# PostgreSQL bağlantısı
def connect_db():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="vector_db",
            user="postgres",
            password="yeni_sifre",
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"Bağlantı hatası: {e}")
        return None

# L2 Distance ile arama
def search_l2(conn, query_text, limit=5):
    cursor = conn.cursor()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Query embedding'i oluştur
    query_embedding = model.encode(query_text)
    embedding_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
    
    try:
        cursor.execute(f"""
            SELECT id, chunk_id, kaynak, metin, 
                   embedding <-> '{embedding_str}'::vector AS distance
            FROM documents
            ORDER BY embedding <-> '{embedding_str}'::vector
            LIMIT %s;
        """, (limit,))
        
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"✗ Arama hatası: {e}")
        return []
    finally:
        cursor.close()

# Cosine Distance ile arama
def search_cosine(conn, query_text, limit=5):
    cursor = conn.cursor()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Query embedding'i oluştur
    query_embedding = model.encode(query_text)
    embedding_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
    
    try:
        cursor.execute(f"""
            SELECT id, chunk_id, kaynak, metin, 
                   embedding <=> '{embedding_str}'::vector AS cosine_distance
            FROM documents
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT %s;
        """, (limit,))
        
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"✗ Arama hatası: {e}")
        return []
    finally:
        cursor.close()

# Inner Product ile arama
def search_inner_product(conn, query_text, limit=5):
    cursor = conn.cursor()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Query embedding'i oluştur
    query_embedding = model.encode(query_text)
    embedding_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
    
    try:
        cursor.execute(f"""
            SELECT id, chunk_id, kaynak, metin, 
                   (embedding <#> '{embedding_str}'::vector) * -1 AS inner_product
            FROM documents
            ORDER BY embedding <#> '{embedding_str}'::vector
            LIMIT %s;
        """, (limit,))
        
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"✗ Arama hatası: {e}")
        return []
    finally:
        cursor.close()

# Veritabanında kaç kayıt olduğunu kontrol et
def check_record_count(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM documents;")
        count = cursor.fetchone()[0]
        return count
    except Exception as e:
        print(f"✗ Kayıt sayısı kontrol hatası: {e}")
        return 0
    finally:
        cursor.close()

# Tüm verileri göster
def show_all_documents(conn, limit=10):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, chunk_id, kaynak, metin
            FROM documents
            LIMIT %s;
        """, (limit,))
        
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"✗ Veri gösterme hatası: {e}")
        return []
    finally:
        cursor.close()

# Ana fonksiyon
def main():
    conn = connect_db()
    if not conn:
        return
    
    try:
        # Kayıt sayısını kontrol et
        count = check_record_count(conn)
        print(f"✓ Veritabanında {count} adet kayıt bulunmaktadır\n")
        
        if count == 0:
            print("⚠ Veritabanında veri yok! Önce write_vector_database.py çalıştırın.")
            return
        
        # Test sorguları
        queries = [
            "artificial intelligence healthcare",
            "machine learning medical diagnosis",
            "deep learning neural networks",
            "data science analytics"
        ]
        
        for query in queries:
            print("="*60)
            print(f"SORGU: '{query}'")
            print("="*60)
            
            # L2 Distance
            print("\n📊 L2 DISTANCE (Euclidean) İLE ARAMA:")
            print("-"*60)
            start = time.time()
            results = search_l2(conn, query, limit=3)
            search_time = time.time() - start
            print(f"Arama zamanı: {search_time:.4f}s\n")
            
            for idx, (id, chunk_id, kaynak, metin, distance) in enumerate(results, 1):
                print(f"{idx}. Sonuç (Distance: {distance:.4f})")
                print(f"   ID: {id} | Chunk: {chunk_id}")
                print(f"   Metin: {metin[:150]}...\n")
            
            # Cosine Distance
            print("📊 COSINE DISTANCE İLE ARAMA:")
            print("-"*60)
            start = time.time()
            results = search_cosine(conn, query, limit=3)
            search_time = time.time() - start
            print(f"Arama zamanı: {search_time:.4f}s\n")
            
            for idx, (id, chunk_id, kaynak, metin, cosine_dist) in enumerate(results, 1):
                similarity = 1 - cosine_dist
                print(f"{idx}. Sonuç (Benzerlik: {similarity:.4f})")
                print(f"   ID: {id} | Chunk: {chunk_id}")
                print(f"   Metin: {metin[:150]}...\n")
            
            # Inner Product
            print("📊 INNER PRODUCT İLE ARAMA:")
            print("-"*60)
            start = time.time()
            results = search_inner_product(conn, query, limit=3)
            search_time = time.time() - start
            print(f"Arama zamanı: {search_time:.4f}s\n")
            
            for idx, (id, chunk_id, kaynak, metin, inner_prod) in enumerate(results, 1):
                print(f"{idx}. Sonuç (Inner Product: {inner_prod:.4f})")
                print(f"   ID: {id} | Chunk: {chunk_id}")
                print(f"   Metin: {metin[:150]}...\n")
            
            print("\n")
        
        # İlk 5 kaydı göster
        print("="*60)
        print("İLK 5 KAYIT")
        print("="*60)
        first_records = show_all_documents(conn, limit=5)
        for idx, (id, chunk_id, kaynak, metin) in enumerate(first_records, 1):
            print(f"\n{idx}. ID: {id} | Chunk: {chunk_id}")
            print(f"   Kaynak: {kaynak}")
            print(f"   Metin: {metin[:200]}...")
            
    finally:
        conn.close()
        print("\n✓ Veritabanı bağlantısı kapatıldı")

if __name__ == "__main__":
    main()