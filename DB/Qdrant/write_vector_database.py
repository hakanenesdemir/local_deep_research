from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import json
import os

# Qdrant client oluştur
client = QdrantClient(host="localhost", port=6333)

# Embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

# Collection oluştur
client.recreate_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# JSON dosyasını oku
json_dosya_yolu = '/home/ugo/Documents/Python/bitirememe projesi/metin_dosyasi.json'

if not os.path.exists(json_dosya_yolu):
    print(f"HATA: {json_dosya_yolu} dosyası bulunamadı!")
    exit()

with open(json_dosya_yolu, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Dökümanları çıkar
documents = data.get("documents", [])
print(f"Toplam {len(documents)} adet döküman bulundu.")


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
tum_parcalar = []
tum_metadatalar = []

for doc in documents:
    doc_id = doc.get("id", 0)
    filename = doc.get("filename", "")
    filepath = doc.get("filepath", "")
    full_text = doc.get("full_text", "")
    
    if not full_text or len(full_text.strip()) < 50:
        print(f"Atlanan döküman (boş veya çok kısa): {filename}")
        continue
    
    # Metni parçala
    parcalar = metni_parcala(full_text, chunk_size=300, overlap=50)
    
    for chunk_idx, parca in enumerate(parcalar):
        tum_parcalar.append(parca)
        tum_metadatalar.append({
            "doc_id": doc_id,
            "filename": filename,
            "filepath": filepath,
            "chunk_id": chunk_idx,
            "text": parca
        })

print(f"Toplam {len(tum_parcalar)} adet metin parçası oluşturuldu.")

# Embedding oluştur
print("Embedding'ler oluşturuluyor...")
embeddings = model.encode(tum_parcalar)

# Qdrant'a ekle
batch_size = 100

for i in range(0, len(tum_parcalar), batch_size):
    batch_metinler = tum_parcalar[i:i+batch_size]
    batch_embeddings = embeddings[i:i+batch_size]
    batch_metadatalar = tum_metadatalar[i:i+batch_size]
    
    points = [
        PointStruct(
            id=i + j,
            vector=batch_embeddings[j].tolist(),
            payload=batch_metadatalar[j]
        )
        for j in range(len(batch_metinler))
    ]
    
    client.upsert(
        collection_name="test_collection",
        points=points
    )
    print(f"{i+len(batch_metinler)}/{len(tum_parcalar)} parça eklendi...")

print(f"\n{len(tum_parcalar)} adet veri başarıyla Qdrant veritabanına kaydedildi!")