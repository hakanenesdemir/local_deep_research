from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import json
import os
import time

# Milvus client oluştur (yerel veritabanı)
client = MilvusClient("milvus_demo.db")

print("✓ Milvus client oluşturuldu")

# Eski collection'ı sil (varsa)
try:
    client.drop_collection(collection_name="documents")
    print("✓ Eski collection silindi")
except:
    pass

# Collection oluştur
client.create_collection(
    collection_name="documents",
    dimension=384,  # all-MiniLM-L6-v2 embedding boyutu
    metric_type="COSINE",  # Cosine similarity kullan
    auto_id=True
)

print("✓ Collection oluşturuldu")

# JSON dosyasını oku
json_dosya_yolu = '/home/ugo/Documents/Python/bitirememe projesi/metin_dosyasi.json'

if not os.path.exists(json_dosya_yolu):
    print(f"HATA: {json_dosya_yolu} dosyası bulunamadı!")
    exit()

with open(json_dosya_yolu, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Dökümanları çıkar
documents = data.get("documents", [])
print(f"✓ Toplam {len(documents)} adet döküman bulundu")


def metni_parcala(metin, chunk_size=500, overlap=100):
    """Metni çakışmalı parçalara böl"""
    parcalar = []
    kelimeler = metin.split()
    
    for i in range(0, len(kelimeler), chunk_size - overlap):
        parca = ' '.join(kelimeler[i:i + chunk_size])
        if len(parca.strip()) > 50:
            parcalar.append(parca.strip())
    
    return parcalar


# Embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

# Tüm parçaları topla
tum_veriler = []

for doc in documents:
    doc_id = doc.get("id", 0)
    filename = doc.get("filename", "")
    filepath = doc.get("filepath", "")
    full_text = doc.get("full_text", "")
    
    if not full_text or len(full_text.strip()) < 50:
        print(f"⚠ Atlanan döküman (boş veya çok kısa): {filename}")
        continue
    
    # Metni parçala
    parcalar = metni_parcala(full_text, chunk_size=300, overlap=50)
    
    for chunk_idx, parca in enumerate(parcalar):
        tum_veriler.append({
            "metin": parca,
            "chunk_id": chunk_idx,
            "doc_id": doc_id,
            "filename": filename,
            "filepath": filepath
        })

print(f"✓ Toplam {len(tum_veriler)} adet metin parçası oluşturuldu\n")

# Milvus'a batch halinde ekle
batch_size = 100
total_added = 0

print("📝 Veri ekleniyor...")
start_time = time.time()

for i in range(0, len(tum_veriler), batch_size):
    batch_veriler = tum_veriler[i:i+batch_size]
    
    # Batch için embedding'leri oluştur
    batch_metinler = [v["metin"] for v in batch_veriler]
    embeddings = model.encode(batch_metinler)
    
    # Milvus için veri hazırla
    milvus_data = []
    for j, veri in enumerate(batch_veriler):
        milvus_data.append({
            "vector": embeddings[j].tolist(),
            "metin": veri["metin"],
            "chunk_id": veri["chunk_id"],
            "doc_id": veri["doc_id"],
            "filename": veri["filename"],
            "filepath": veri["filepath"]
        })
    
    # Milvus'a ekle
    res = client.insert(
        collection_name="documents",
        data=milvus_data
    )
    
    total_added += len(batch_veriler)
    print(f"✓ {total_added}/{len(tum_veriler)} parça eklendi...")

insert_time = time.time() - start_time

print(f"\n{'='*60}")
print(f"✓ {len(tum_veriler)} adet veri başarıyla Milvus'a kaydedildi!")
print(f"✓ Toplam ekleme zamanı: {insert_time:.2f}s")
print(f"{'='*60}")

# Collection bilgisi
stats = client.get_collection_stats(collection_name="documents")
print(f"\n📊 Collection Bilgisi:")
print(f"   Collection adı: documents")
print(f"   Toplam kayıt: {stats['row_count']}")

print("✓ İşlem tamamlandı")