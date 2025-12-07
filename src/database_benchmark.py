import time
import psutil
import os
import json
from typing import Dict, List, Tuple
import lancedb
import chromadb
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class DatabaseBenchmark:
    def __init__(self):
        self.results = {
            "lancedb": {},
            "chromadb": {},
            "qdrant": {}
        }
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def measure_memory(self) -> float:
        """Bellek kullanımını MB cinsinden ölç"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def measure_disk_size(self, path: str) -> float:
        """Klasör boyutunu MB cinsinden ölç"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / 1024 / 1024
    
    # ==================== LanceDB Benchmark ====================
    def benchmark_lancedb(self):
        """LanceDB performansını ölç"""
        print("\n" + "="*50)
        print("LanceDB BENCHMARK")
        print("="*50)
        
        try:
            db_path = "/home/ugo/Documents/Python/bitirememe projesi/DB/lanceDatabase/db"
            
            # Bağlantı zamanı
            start = time.time()
            db = lancedb.connect(db_path)
            connection_time = time.time() - start
            self.results["lancedb"]["connection_time"] = connection_time
            print(f"✓ Bağlantı zamanı: {connection_time:.4f}s")
            
            # Tablo açma zamanı
            start = time.time()
            table = db.open_table("documents")
            open_time = time.time() - start
            self.results["lancedb"]["open_table_time"] = open_time
            print(f"✓ Tablo açma zamanı: {open_time:.4f}s")
            
            # Kayıt sayısı
            count = len(table.to_pandas())
            self.results["lancedb"]["record_count"] = count
            print(f"✓ Toplam kayıt: {count}")
            
            # Arama performansı
            queries = [
                "artificial intelligence healthcare",
                "machine learning medical",
                "deep learning diagnosis"
            ]
            
            search_times = []
            for query in queries:
                start = time.time()
                results = table.search(query).limit(5).to_pandas()
                search_time = time.time() - start
                search_times.append(search_time)
            
            avg_search_time = sum(search_times) / len(search_times)
            self.results["lancedb"]["avg_search_time"] = avg_search_time
            print(f"✓ Ortalama arama zamanı: {avg_search_time:.4f}s")
            
            # Disk boyutu
            disk_size = self.measure_disk_size(db_path)
            self.results["lancedb"]["disk_size_mb"] = disk_size
            print(f"✓ Disk boyutu: {disk_size:.2f} MB")
            
            # Bellek kullanımı
            start_mem = self.measure_memory()
            table.to_pandas()
            end_mem = self.measure_memory()
            memory_used = end_mem - start_mem
            self.results["lancedb"]["memory_used_mb"] = memory_used
            print(f"✓ Bellek kullanımı: {memory_used:.2f} MB")
            
        except Exception as e:
            print(f"✗ Hata: {e}")
            self.results["lancedb"]["error"] = str(e)
    
    # ==================== ChromaDB Benchmark ====================
    def benchmark_chromadb(self):
        """ChromaDB performansını ölç"""
        print("\n" + "="*50)
        print("ChromaDB BENCHMARK")
        print("="*50)
        
        try:
            db_path = "./yerel_veritabani"
            
            if not os.path.exists(db_path):
                print(f"✗ ChromaDB veritabanı bulunamadı: {db_path}")
                return
            
            # Bağlantı zamanı
            start = time.time()
            client = chromadb.PersistentClient(path=db_path)
            connection_time = time.time() - start
            self.results["chromadb"]["connection_time"] = connection_time
            print(f"✓ Bağlantı zamanı: {connection_time:.4f}s")
            
            # Collection açma
            start = time.time()
            collection = client.get_or_create_collection(name="dokumanlarim")
            collection_time = time.time() - start
            self.results["chromadb"]["collection_open_time"] = collection_time
            print(f"✓ Collection açma zamanı: {collection_time:.4f}s")
            
            # Kayıt sayısı
            count = collection.count()
            self.results["chromadb"]["record_count"] = count
            print(f"✓ Toplam kayıt: {count}")
            
            # Arama performansı
            queries = [
                "artificial intelligence healthcare",
                "machine learning medical",
                "deep learning diagnosis"
            ]
            
            search_times = []
            for query in queries:
                start = time.time()
                results = collection.query(query_texts=[query], n_results=5)
                search_time = time.time() - start
                search_times.append(search_time)
            
            avg_search_time = sum(search_times) / len(search_times)
            self.results["chromadb"]["avg_search_time"] = avg_search_time
            print(f"✓ Ortalama arama zamanı: {avg_search_time:.4f}s")
            
            # Disk boyutu
            disk_size = self.measure_disk_size(db_path)
            self.results["chromadb"]["disk_size_mb"] = disk_size
            print(f"✓ Disk boyutu: {disk_size:.2f} MB")
            
        except Exception as e:
            print(f"✗ Hata: {e}")
            self.results["chromadb"]["error"] = str(e)
    
    # ==================== Qdrant Benchmark ====================
    def benchmark_qdrant(self):
        """Qdrant performansını ölç"""
        print("\n" + "="*50)
        print("Qdrant BENCHMARK")
        print("="*50)
        
        try:
            # Bağlantı zamanı
            start = time.time()
            client = QdrantClient(host="localhost", port=6333)
            connection_time = time.time() - start
            self.results["qdrant"]["connection_time"] = connection_time
            print(f"✓ Bağlantı zamanı: {connection_time:.4f}s")
            
            # Collection bilgisi
            start = time.time()
            collection_info = client.get_collection("test_collection")
            info_time = time.time() - start
            self.results["qdrant"]["info_time"] = info_time
            print(f"✓ Collection info zamanı: {info_time:.4f}s")
            
            # Kayıt sayısı
            count = collection_info.points_count
            self.results["qdrant"]["record_count"] = count
            print(f"✓ Toplam kayıt: {count}")
            
            # Arama performansı
            queries = [
                "artificial intelligence healthcare",
                "machine learning medical",
                "deep learning diagnosis"
            ]
            
            search_times = []
            for query in queries:
                query_vector = self.model.encode(query).tolist()
                start = time.time()
                results = client.query_points(
                    collection_name="test_collection",
                    query=query_vector,
                    limit=5,
                    with_payload=True
                )
                search_time = time.time() - start
                search_times.append(search_time)
            
            avg_search_time = sum(search_times) / len(search_times)
            self.results["qdrant"]["avg_search_time"] = avg_search_time
            print(f"✓ Ortalama arama zamanı: {avg_search_time:.4f}s")
            
        except Exception as e:
            print(f"✗ Hata: {e}")
            self.results["qdrant"]["error"] = str(e)
    
    def run_all_benchmarks(self):
        """Tüm benchmark'leri çalıştır"""
        self.benchmark_lancedb()
        self.benchmark_chromadb()
        self.benchmark_qdrant()
        self.print_comparison()
        self.save_results()
    
    def print_comparison(self):
        """Karşılaştırmalı sonuçlar"""
        print("\n" + "="*50)
        print("KARŞILAŞTIRMA")
        print("="*50)
        
        print("\n📊 Arama Zamanı (Ortalama):")
        for db_name, metrics in self.results.items():
            if "avg_search_time" in metrics:
                print(f"  {db_name}: {metrics['avg_search_time']:.4f}s")
        
        print("\n💾 Disk Boyutu:")
        for db_name, metrics in self.results.items():
            if "disk_size_mb" in metrics:
                print(f"  {db_name}: {metrics['disk_size_mb']:.2f} MB")
        
        print("\n📈 Bağlantı Zamanı:")
        for db_name, metrics in self.results.items():
            if "connection_time" in metrics:
                print(f"  {db_name}: {metrics['connection_time']:.4f}s")
    
    def save_results(self):
        """Sonuçları JSON dosyasına kaydet"""
        output_file = "/home/ugo/Documents/Python/bitirememe projesi/DB/benchmark_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Sonuçlar kaydedildi: {output_file}")

if __name__ == "__main__":
    benchmark = DatabaseBenchmark()
    benchmark.run_all_benchmarks()