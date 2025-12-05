import lancedb
import os
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

db = lancedb.connect("/home/ugo/Documents/Python/bitirememe projesi/DB/lanceDatabase/db")
func = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2", device="cuda")

txt_dosya_yolu = '/home/ugo/Documents/Python/bitirememe projesi/metin_dosyasi.txt'

if not os.path.exists(txt_dosya_yolu):
    print(f"HATA: {txt_dosya_yolu} dosyası bulunamadı!")
    exit()

with open(txt_dosya_yolu, 'r', encoding='utf-8') as f:
    icerik = f.read()


def metni_parcala(metin, chunk_size=500, overlap=100):
    """Metni çakışmalı parçalara böl"""
    parcalar = []
    kelimeler = metin.split()
    
    for i in range(0, len(kelimeler), chunk_size - overlap):
        parca = ' '.join(kelimeler[i:i + chunk_size])
        if len(parca.strip()) > 50:
            parcalar.append(parca.strip())
    
    return parcalar


class Document(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()
    chunk_id: int
    kaynak: str


# Metni parçala
metinler = metni_parcala(icerik, chunk_size=300, overlap=50)
print(f"Toplam {len(metinler)} adet metin parçası oluşturuldu.")

# Tablo oluştur
table = db.create_table("documents", schema=Document, mode="overwrite")

# Verileri hazırla ve ekle
batch_size = 1000

for i in range(0, len(metinler), batch_size):
    batch_metinler = metinler[i:i+batch_size]
    
    batch_data = [
        {
            "text": metin,
            "chunk_id": i + j,
            "kaynak": "metin_dosyasi.txt"
        }
        for j, metin in enumerate(batch_metinler)
    ]
    
    table.add(batch_data)
    print(f"{i+len(batch_metinler)}/{len(metinler)} parça eklendi...")

print(f"\n{len(metinler)} adet veri başarıyla LanceDB veritabanına kaydedildi!")

# Test sorgusu
query = "artificial intelligence healthcare"
results = table.search(query).limit(3).to_pydantic(Document)

print("\n--- Arama Sonuçları ---")
for result in results:
    print(f"\nChunk ID: {result.chunk_id}")
    print(f"Kaynak: {result.kaynak}")
    print(f"Text: {result.text[:200]}...")