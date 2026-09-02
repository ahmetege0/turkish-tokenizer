import time
from pathlib import Path

def main():
    in_file = Path('output/turkish_corpus_shuffled.txt')
    out_file = Path('output/turkish_corpus_shuffled_unique.txt')
    
    print("Kopya satirlar temizleniyor ve yeni dosyaya yaziliyor...")
    seen_hashes = set()
    written = 0
    t0 = time.time()
    
    with open(in_file, "r", encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
        for line in fin:
            clean_line = line.strip()
            if not clean_line:
                continue
            h = hash(clean_line)
            if h not in seen_hashes:
                seen_hashes.add(h)
                fout.write(line)
                written += 1
                
    elapsed = time.time() - t0
    out_mb = out_file.stat().st_size / 1024**2
    print(f"\nIslem tamamlandi! ({elapsed:.1f} sn)")
    print(f"Yeni essiz dosya: {out_file.name}")
    print(f"Toplam Satir    : {written:,}")
    print(f"Yeni Boyut      : {out_mb:.1f} MB")

if __name__ == "__main__":
    main()
