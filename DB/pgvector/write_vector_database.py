import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import json
import os
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

# Veritabanı ve tablo oluştur
def create_table(conn):
    cursor = conn.cursor()
    try:
        # pgvector extension'ını etkinleştir
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Eski tabloyu sil (temiz başlangıç için)
        cursor.execute("DROP TABLE IF EXISTS documents;")
        
        # Tablo oluştur
        cursor.execute("""
            CREATE TABLE documents (
                id BIGSERIAL PRIMARY KEY,
                chunk_id INT,
                kaynak VARCHAR(255),
                metin TEXT,
                embedding vector(384)
            );
        """)
        
        conn.commit()
        print("✓ Tablo başarıyla oluşturuldu")
    except Exception as e:
        print(f"✗ Tablo oluşturma hatası: {e}")
        conn.rollback()
    finally:
        cursor.close()

# Metni parçalara böl
def metni_parcala(metin, chunk_size=500, overlap=100):
    """Metni çakışmalı parçalara böl"""
    parcalar = []
    kelimeler = metin.split()
    
    for i in range(0, len(kelimeler), chunk_size - overlap):
        parca = ' '.join(kelimeler[i:i + chunk_size])
        if len(parca.strip()) > 50:
            parcalar.append(parca.strip())
    
    return parcalar

# Veri ekle
def insert_documents(conn, metinler):
    cursor = conn.cursor()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    batch_size = 100
    total_inserted = 0
    
    for i in range(0, len(metinler), batch_size):
        batch_metinler = metinler[i:i+batch_size]
        
        # Embedding oluştur
        embeddings = model.encode(batch_metinler)
        
        try:
            for j, (metin, embedding) in enumerate(zip(batch_metinler, embeddings)):
                # Embedding'i string formatına çevir: [val1, val2, ...]
                embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
                
                cursor.execute("""
                    INSERT INTO documents (chunk_id, kaynak, metin, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                """, (i + j, "metin_dosyasi.txt", metin, embedding_str))
            
            conn.commit()
            total_inserted += len(batch_metinler)
            print(f"✓ {total_inserted}/{len(metinler)} kayıt eklendi...")
        except Exception as e:
            print(f"✗ Veri ekleme hatası: {e}")
            conn.rollback()
    
    cursor.close()
    print(f"\n✓ Toplam {total_inserted} kayıt başarıyla eklendi")

# Ana fonksiyon
def main():
    # JSON dosyasını oku
    json_dosya_yolu = '/home/ugo/Documents/Python/bitirememe projesi/metin_dosyasi.json'
    
    if not os.path.exists(json_dosya_yolu):
        print(f"✗ HATA: {json_dosya_yolu} dosyası bulunamadı!")
        return
    
    with open(json_dosya_yolu, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Dökümanları çıkar
    documents = data.get("documents", [])
    print(f"✓ Toplam {len(documents)} adet döküman bulundu")
    
    # Tüm metinleri topla
    tum_metinler = []
    for doc in documents:
        full_text = doc.get("full_text", "")
        if full_text and len(full_text.strip()) > 50:
            tum_metinler.append(full_text)
    
    # Metinleri parçala
    metinler = []
    for metin in tum_metinler:
        parcalar = metni_parcala(metin, chunk_size=300, overlap=50)
        metinler.extend(parcalar)
    
    print(f"✓ Toplam {len(metinler)} adet metin parçası oluşturuldu\n")
    
    # Veritabanına bağlan
    conn = connect_db()
    if not conn:
        return
    
    try:
        # Tablo oluştur
        create_table(conn)
        
        # Veri ekle
        print("📝 Veri ekleniyor...\n")
        start_time = time.time()
        insert_documents(conn, metinler)
        insert_time = time.time() - start_time
        print(f"✓ Toplam veri ekleme zamanı: {insert_time:.2f}s")
            
    finally:
        conn.close()
        print("✓ Veritabanı bağlantısı kapatıldı")

if __name__ == "__main__":
    main()