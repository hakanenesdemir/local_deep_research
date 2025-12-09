import os
import json
import uuid
import re
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# --- AYARLAR ---
INPUT_FILE = "CUSTOM_DATASET/metin_dosyasi.json"
OUTPUT_FILE = "CUSTOM_DATASET/squad_dataset.json"
MODEL_NAME = "ministral-3:8b"
CONTEXT_PER_DOC = 5   # Her dökümandan kaç parça alınacak
Q_PER_CONTEXT = 5     # Her parçadan kaç soru üretilecek

class DatasetGenerator:
    def __init__(self):
        print(f"Model yükleniyor: {MODEL_NAME}")
        self.model = OllamaLLM(model=MODEL_NAME, temperature=0.1) 
        
    def extract_json_from_response(self, text: str):
        """
        LLM çıktısı içinden JSON bloğunu regex ile ayıklar.
        """
        try:
            # Markdown code block temizle
            text = text.replace("```json", "").replace("```", "").strip()
            
            # Eğer text bir liste [...] ile başlıyorsa direkt parse et
            if text.startswith("[") and text.endswith("]"):
                return json.loads(text)
            
            # Regex ile [...] bloğunu bulmaya çalış
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            
            return []
        except Exception as e:
            print(f"JSON Parse Hatası: {e}")
            return []

    def generate_qa_pairs(self, context: str) -> List[Dict]:
        system_prompt = """
        You are a strict data annotator for a SQuAD (Question Answering) dataset.
        
        TASK:
        1. Read the provided text snippet carefully.
        2. Extract exactly 5 question-answer pairs.
        3. CRITICAL RULE: The "answer" must be a SUBSTRING COPIED EXACTLY from the text. Do not rewrite or paraphrase.
        
        OUTPUT FORMAT (JSON Array only):
        [
            {"question": "What is X?", "answer": "exact words from text"},
            {"question": "Who did Y?", "answer": "exact words from text"}
        ]
        """
        
        template = """
        {system}
        
        TEXT SNIPPET:
        {context}
        
        JSON OUTPUT:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model
        
        try:
            response = chain.invoke({
                "system": system_prompt,
                "context": context
            })
            return self.extract_json_from_response(response)
        except Exception as e:
            print(f"LLM Hatası: {e}")
            return []

    def find_answer_start(self, context: str, answer: str) -> int:
        """
        Cevabın başlangıç indeksini bulur. Bulamazsa 2. şans olarak case-insensitive arar.
        """
        # 1. Tam eşleşme ara
        idx = context.find(answer)
        if idx != -1:
            return idx
            
        # 2. Bulamazsa, metni ve cevabı küçük harfe çevirip ara
        idx_lower = context.lower().find(answer.lower())
        
        # ndeksi döndürüyoruz.
        return idx_lower

    def process_file(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            print(f"HATA: {input_path} dosyası bulunamadı")
            return
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # JSON yapısına göre doğru listeyi bulma mantığı
        if isinstance(raw_data, dict) and "documents" in raw_data:
            print("Yapı tespit edildi: Veriler 'documents' listesinin içinde.")
            data_list = raw_data["documents"]
        elif isinstance(raw_data, list):
            data_list = raw_data
        else:
            # Tek bir obje ise listeye çevir
            data_list = [raw_data]

        squad_data = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100
        )

        print(f"Toplam {len(data_list)} döküman işlenecek.")

        for doc_idx, doc in enumerate(data_list):
            # Dosyadaki anahtar ismini kontrol et
            full_text = doc.get("full_text")
            title = doc.get("filename", f"doc_{doc_idx}")
            
            if not full_text:
                print(f"UYARI: {title} dökümanında 'full_text' boş. Mevcut anahtarlar: {list(doc.keys())}")
                continue

            print(f"İşleniyor: {title} ({len(full_text)} karakter)")
            
            # Metni parçala
            chunks = text_splitter.split_text(full_text)
            
            # Context seçimi
            if len(chunks) <= CONTEXT_PER_DOC:
                selected_chunks = chunks
            else:
                step = len(chunks) // CONTEXT_PER_DOC
                selected_chunks = [chunks[i] for i in range(0, len(chunks), step)][:CONTEXT_PER_DOC]

            paragraphs = []
            
            for i, chunk in enumerate(selected_chunks):
                print(f"generating Q&A for chunk {i+1}/{len(selected_chunks)}...", end="", flush=True)
                
                qa_pairs = self.generate_qa_pairs(chunk)
                
                valid_qas = 0
                qas_list = []
                
                for qa in qa_pairs:
                    q_text = qa.get("question")
                    a_text = qa.get("answer")
                    
                    if not q_text or not a_text:
                        continue
                        
                    start_index = self.find_answer_start(chunk, a_text)
                    
                    if start_index != -1:
                        # Eğer case-insensitive bulduysak, orijinal metindeki halini alalım ki SQuAD formatı bozulmasın
                        real_answer = chunk[start_index : start_index + len(a_text)]
                        
                        qas_list.append({
                            "question": q_text,
                            "id": str(uuid.uuid4()),
                            "answers": [{"text": real_answer, "answer_start": start_index}],
                            "is_impossible": False
                        })
                        valid_qas += 1
                
                print(f" {valid_qas} soru üretildi.")
                
                if qas_list:
                    paragraphs.append({
                        "context": chunk,
                        "qas": qas_list
                    })

            if paragraphs:
                squad_data.append({
                    "title": title,
                    "paragraphs": paragraphs
                })

        final_output = {"version": "v2.0", "data": squad_data}

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        
        print(f"\nİŞLEM TAMAMLANDI: {len(squad_data)} döküman için veri üretildi.")
        print(f"Kayıt yeri: {output_path}")

if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.process_file(INPUT_FILE, OUTPUT_FILE)