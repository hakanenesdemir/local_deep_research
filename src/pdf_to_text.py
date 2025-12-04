import os
import fitz
import json


pdf_folder_path = '/home/ugo/Documents/Python/git_clone/scrape-google-scholar-py/healthcare_ai_pdfs'


output_json_file = '/home/ugo/Documents/Python/bitirememe projesi/metin_dosyasi.json'


output_folder = os.path.dirname(output_json_file)
if not os.path.exists(output_folder):
    print(f"Uyarı: Çıktı klasörü '{output_folder}' mevcut değil. Oluşturulmaya çalışılacak.")
    os.makedirs(output_folder, exist_ok=True)

print(f"'{pdf_folder_path}' klasörü taranıyor...")

try:
    # Tüm PDF dosyalarını bul
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"'{pdf_folder_path}' klasöründe hiç PDF dosyası bulunamadı.")
    else:
        print(f"Toplam {len(pdf_files)} adet PDF dosyası bulundu.")

        documents = []

        for i, filename in enumerate(pdf_files, 1):
            filepath = os.path.join(pdf_folder_path, filename)

            try:
                print(f"[{i}/{len(pdf_files)}] İşleniyor: {filename}")
                doc = fitz.open(filepath)

                pages_content = []
                full_text = ""

                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    cleaned_text = " ".join(text.split())
                    
                    pages_content.append({
                        "page_number": page_num + 1,
                        "content": cleaned_text
                    })
                    full_text += cleaned_text + " "

                documents.append({
                    "id": i,
                    "filename": filename,
                    "filepath": filepath,
                    "page_count": len(doc),
                    "full_text": full_text.strip(),
                    "pages": pages_content
                })

                doc.close()

            except Exception as e:
                print(f"HATA: {filename} dosyası işlenirken bir sorun oluştu: {e}")
                documents.append({
                    "id": i,
                    "filename": filename,
                    "filepath": filepath,
                    "error": str(e)
                })

        # JSON dosyasına kaydet
        with open(output_json_file, 'w', encoding='utf-8') as outfile:
            json.dump({
                "total_documents": len(documents),
                "source_folder": pdf_folder_path,
                "documents": documents
            }, outfile, ensure_ascii=False, indent=2)

        print(f"\nİşlem tamamlandı!")
        print(f"Tüm metinler başarıyla şu dosyaya kaydedildi: '{output_json_file}'")

except Exception as e:
    print(f"Beklenmedik bir hata oluştu: {e}")