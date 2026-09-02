import random
import os
import time
from pathlib import Path

RAW_FILE = Path("d:/MAGIBU/turkish_datasets/tokenizer_project/output/turkish_corpus_raw.txt")
SHUF_FILE = Path("d:/MAGIBU/turkish_datasets/tokenizer_project/output/turkish_corpus_shuffled.txt")
TEMP_DIR = Path("d:/MAGIBU/turkish_datasets/tokenizer_project/output/temp_chunks")

def main():
    if not RAW_FILE.exists():
        print(f"HATA: {RAW_FILE} bulunamadi!")
        return

    TEMP_DIR.mkdir(exist_ok=True)
    random.seed(42)
    
    num_chunks = 20
    print(f"1. Asama: {RAW_FILE.name} RAM dostu sekilde {num_chunks} parcaya ayriliyor...")
    t0 = time.time()
    
    temp_files = [open(TEMP_DIR / f"chunk_{i}.txt", "w", encoding="utf-8") for i in range(num_chunks)]
    
    line_count = 0
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            temp_files[random.randint(0, num_chunks - 1)].write(line)
            line_count += 1
            if line_count % 2_000_000 == 0:
                print(f"  {line_count:,} satir paylastirildi...")
                
    for tf in temp_files:
        tf.close()
        
    print(f"Parcalama bitti. ({time.time() - t0:.1f} sn) Toplam {line_count:,} satir.")
    
    print("\n2. Asama: Her bir parca RAM'de karistirilip ana dosyaya yaziliyor...")
    t1 = time.time()
    with open(SHUF_FILE, "w", encoding="utf-8") as out_f:
        for i in range(num_chunks):
            chunk_path = TEMP_DIR / f"chunk_{i}.txt"
            with open(chunk_path, "r", encoding="utf-8") as tf:
                lines = tf.readlines()
            random.shuffle(lines)
            out_f.writelines(lines)
            chunk_path.unlink() 
            print(f"  chunk_{i}.txt karistirildi ve eklendi. ({len(lines):,} satir)")
            
    TEMP_DIR.rmdir()
    
    shuf_mb = SHUF_FILE.stat().st_size / 1024**2
    print(f"\nISLEM TAMAM! ({time.time() - t1:.1f} sn)")
    print(f"Karistirilmis dosya: {SHUF_FILE.name} ({shuf_mb:.1f} MB)")

if __name__ == "__main__":
    main()
