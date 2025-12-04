import chromadb
from chromadb.utils import embedding_functions
import os
import re

client = chromadb.PersistentClient(path="./yerel_veritabani")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="dokumanlarim",
    embedding_function=ef
)

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

metinler = metni_parcala(icerik, chunk_size=300, overlap=50)

print(f"Toplam {len(metinler)} adet metin parçası oluşturuldu.")

idler = [f"chunk_{i}" for i in range(len(metinler))]

metadatalar = [{"chunk_id": i, "kaynak": "metin_dosyasi.txt"} for i in range(len(metinler))]

batch_size = 1000

for i in range(0, len(metinler), batch_size):
    batch_metinler = metinler[i:i+batch_size]
    batch_idler = idler[i:i+batch_size]
    batch_metadatalar = metadatalar[i:i+batch_size]
    
    collection.add(
        documents=batch_metinler,
        metadatas=batch_metadatalar,
        ids=batch_idler
    )
    print(f"{i+len(batch_metinler)}/{len(metinler)} parça eklendi...")

print(f"\n{len(metinler)} adet veri başarıyla vektör veritabanına kaydedildi!")