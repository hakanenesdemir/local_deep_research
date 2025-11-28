import chromadb
from chromadb.utils import embedding_functions


def ollama(self, question: str) -> str:
    system = """
    You are a translator for manga dialogues.

    Rules:
    1. Translate from English to Turkish.
    2. Do not translate proper names (manga titles, character names, location names, etc.).
    3. Fix broken words or lines. Example: "THIS 15 17" → "This is it".
    4. Remove any "-" characters from the text unless they are part of a proper name.
    5. Return only the translated text. No explanations, no extra words.
    """
    template = """
    System: {system}
    Translate this: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model = "llama3.1:8b")

    chain = prompt | model
    

    answer = chain.invoke({
        "system": system,
        "question": question
    })
            
    return answer




client = chromadb.PersistentClient(path="./yerel_veritabani")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


collection = client.get_or_create_collection(
    name="dokumanlarim",
    embedding_function=ef
)

print(f"Toplam veri sayısı: {collection.count()}")

results = collection.get(
    limit=10,
    include=["documents", "metadatas"]
)

print("\n=== İLK 10 VERİ ===")
for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
    print(f"\n{i}. Veri:")
    print(f"   Metadata: {meta}")
    print(f"   İçerik: {doc[:100]}...")  

print("\n\n=== ARAMA SONUÇLARI ===")
sorgu = "artificial intelligence healthcare"
arama_sonuclari = collection.query(
    query_texts=[sorgu],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

print(f"'{sorgu}' için en yakın 5 sonuç:")
for i, (doc, meta, dist) in enumerate(zip(
    arama_sonuclari['documents'][0],
    arama_sonuclari['metadatas'][0],
    arama_sonuclari['distances'][0]
), 1):
    print(f"\n{i}. Sonuç (Benzerlik: {1-dist:.4f}):")
    print(f"   Metadata: {meta}")
    print(f"   İçerik: {doc[:150]}...")