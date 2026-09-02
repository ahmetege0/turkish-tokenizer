import time
from pathlib import Path

def main():
    target_file = Path('output/turkish_corpus_shuffled.txt')
    if not target_file.exists():
        print(f"Dosya bulunamadi: {target_file}")
        return

    print("Kopya (duplicate) satirlar taranmaya basliyor...")
    print("RAM'i yormamak icin Hash (Parmak izi) yontemi kullaniliyor.\n")
    t0 = time.time()
    
    seen_hashes = set()
    total_lines = 0
    duplicate_lines = 0
    
    with open(target_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            total_lines += 1
            # Hafızada koca metinleri degil, sadece 8 byte'lik parmak izlerini tutuyoruz
            h = hash(line)
            
            if h in seen_hashes:
                duplicate_lines += 1
            else:
                seen_hashes.add(h)
                
            if total_lines % 2_000_000 == 0:
                print(f"  {total_lines:,} satir tarandi. Bulunan kopya: {duplicate_lines:,}")
                
    elapsed = time.time() - t0
    dup_ratio = (duplicate_lines / total_lines) * 100 if total_lines > 0 else 0
    
    print("\n--- KOPYA (DUPLICATE) ANALIZI SONUCU ---")
    print(f"Toplam Gecerli Satir : {total_lines:,}")
    print(f"Kopya (Tekrar) Satir : {duplicate_lines:,}")
    print(f"Essiz (Unique) Satir : {total_lines - duplicate_lines:,}")
    print(f"Cakisma (Kopya) Orani: %{dup_ratio:.2f}")
    print(f"Gecen Sure           : {elapsed:.1f} saniye")

if __name__ == "__main__":
    main()
