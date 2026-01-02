import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# bu oluşturulacak mı ? 
DB_PATH = "C:/Users/hakan/Desktop/bitirme/Local_deep_research/DB/chorame/yerel_veritabani"

# ???????
COLLECTION_NAME = "dokumanlarim"

# Sabit
MODEL_NAME = "all-MiniLM-L6-v2"

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

def ollama(pdf_text: str) -> str:
    # System promptu template'in içine sabit olarak gömüyoruz.
    # Böylece her çağrıda tekrar tekrar tanımlamaya gerek kalmıyor.
    template = """
    System:
    You are an expert content summarizer specializing in multi-document analysis.
    
    Rules:
    1. Analyze the provided text strings representing the content of 5 English PDF documents.
    2. Identify the core thesis, key arguments, and critical data points in each document.
    3. Synthesize the information into a single coherent narrative that captures the essence of all 5 documents.
    4. Translate the final synthesized summary into fluent, professional Turkish.
    5. Return only the Turkish summary text. No explanations, no extra words, no introductory or concluding remarks.

    Content to analyze:
    {question}
    """

    # Template oluşturuluyor
    prompt = ChatPromptTemplate.from_template(template)

    # Model tanımlanıyor (Sıcaklık değeri 0 yapılırsa daha tutarlı sonuç verir)
    model = OllamaLLM(model="llama3.2:3b", temperature=0)

    # Zincir (Chain) kuruluyor
    chain = prompt | model

    # Fonksiyon çalıştırılıyor
    # Not: 'question' anahtarı template içindeki {question} yer tutucusuyla eşleşir.
    answer = chain.invoke({
        "question": pdf_text
    })
            
    return answer

def get_collection():
    # chromadb.PersistentClient yapısı "yoksa oluştur, varsa bağlan" olmadığı için yeni oluştur.
    client = chromadb.PersistentClient(path=DB_PATH)

    # Sabit işlemler devam
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

    # Sabit devam
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

def hnsw_vector_search(queries: List[str], top_k: int = 5):
    model = SentenceTransformer(MODEL_NAME)
    collection = get_collection()
    print(f"✓ Koleksiyon: {collection.name} | Kayıt sayısı: {collection.count()}", flush=True)
    
    # 1. Sonuçları toplayacağımız boş bir liste oluştur
    tum_sonuclar = [] 

    for q in queries:
        q_vec = model.encode(q).tolist()
        results = collection.query(query_embeddings=[q_vec], n_results=top_k, include=["documents", "metadatas", "distances"])
        print(f"\nSorgu: '{q}'", flush=True)
        
        if results['documents']: # Sonuç varsa
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ), 1):
                sim = 1 - dist
                print(f"  {i}. Benzerlik: {sim:.4f} | Dosya: {meta.get('filename')}", flush=True)
                
                # 2. Sonucu sözlük (dictionary) olarak listeye ekle
                tum_sonuclar.append({
                    "sirasi": i,
                    "dosya_adi": meta.get('filename'),
                    "benzerlik": float(f"{sim:.4f}"), # float'a çevir
                    "metin": doc
                })

    # 3. Listeyi geri döndür (Return)
    return tum_sonuclar

if __name__ == "__main__":

    # Buradaki sample_queries değişkeni girilen_metin mi ?
    sample_queries = [
        "artificial intelligence healthcare",
        "machine learning medical diagnosis",
        "deep learning neural networks",
    ]
    hnsw_vector_search(sample_queries, top_k=5)


class GirdiVerisi(BaseModel):
    girilen_metin: str

@app.post("/ask/question/ai")
def askQuestionAI(veri: GirdiVerisi) -> Dict[str, Any]: 
    # ChromaDB'de sorgu yap
    # Buraya hswvektorsearch gelecek ?
    sample_queries = [
        veri.girilen_metin
    ]
    aiResponse = hnsw_vector_search(sample_queries)
    return {"aiResponse": aiResponse}


@app.post("/summary")
async def summaryPdfs(veri: GirdiVerisi) -> Dict[str, Any]:
    try:
        # Gelen veriyi (veri.girilen_metin) işleme fonksiyonuna gönderiyoruz
        if not veri.girilen_metin or not veri.girilen_metin.strip():
            return {
                "status": "error",
                "message": "girilen_metin boş olamaz"
            }
        
        print(f"Summary request alındı: {veri.girilen_metin[:50]}...", flush=True)
        
        summary_result = ollama(veri.girilen_metin)
        
        # Başarılı dönüş formatı
        return {
            "status": "success",
            "summary": summary_result
        }
        
    except Exception as e:
        # Hata durumunda dönüş formatı
        import traceback
        print(f"Hata: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        return {
            "status": "error",
            "message": str(e)
        }

