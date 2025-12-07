import chromadb
from chromadb.utils import embedding_functions
import os
import json
import time

client = chromadb.PersistentClient(path="./yerel_veritabani")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="dokumanlarim",
    embedding_function=ef
)

json_dosya_yolu = '/home/ugo/Documents/Python/bitirememe projesi/metin_dosyasi.json'

if not os.path.exists(json_dosya_yolu):
    print(f"HATA: {json_dosya_yolu} dosyası bulunamadı!")
    exit()

# JSON dosyasını oku
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


# Tüm parçaları ve metadata'ları topla
tum_metinler = []
tum_idler = []
tum_metadatalar = []

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
        tum_metinler.append(parca)
        tum_idler.append(f"doc_{doc_id}_chunk_{chunk_idx}")
        tum_metadatalar.append({
            "doc_id": doc_id,
            "filename": filename,
            "filepath": filepath,
            "chunk_id": chunk_idx
        })

print(f"✓ Toplam {len(tum_metinler)} adet metin parçası oluşturuldu\n")

# ChromaDB'ye batch halinde ekle
batch_size = 100
total_added = 0

print("📝 Veri ekleniyor...")
start_time = time.time()

for i in range(0, len(tum_metinler), batch_size):
    batch_metinler = tum_metinler[i:i+batch_size]
    batch_idler = tum_idler[i:i+batch_size]
    batch_metadatalar = tum_metadatalar[i:i+batch_size]
    
    collection.add(
        documents=batch_metinler,
        metadatas=batch_metadatalar,
        ids=batch_idler
    )
    total_added += len(batch_metinler)
    print(f"✓ {total_added}/{len(tum_metinler)} parça eklendi...")

insert_time = time.time() - start_time

print(f"\n{'='*60}")
print(f"✓ {len(tum_metinler)} adet veri başarıyla ChromaDB'ye kaydedildi!")
print(f"✓ Toplam ekleme zamanı: {insert_time:.2f}s")
print(f"{'='*60}")

# Koleksiyon bilgisi
print(f"\n📊 Koleksiyon Bilgisi:")
print(f"   Koleksiyon adı: {collection.name}")
print(f"   Toplam kayıt: {collection.count()}")