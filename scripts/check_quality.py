import collections
import random
from pathlib import Path

def main():
    lines_to_read = 100_000
    char_counter = collections.Counter()
    word_counter = collections.Counter()
    total_len = 0
    total_words = 0
    weird_chars = collections.Counter()
    turkish_chars = set('çğıöşüÇĞIÖŞÜ')

    target_file = Path('output/turkish_corpus_shuffled.txt')
    if not target_file.exists():
        print(f"Dosya bulunamadi: {target_file}")
        return

    lines = []
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            for i in range(lines_to_read):
                line = f.readline()
                if not line: break
                
                line = line.strip()
                if not line: continue
                
                lines.append(line)
                total_len += len(line)
                char_counter.update(line)
                
                words = line.lower().split()
                total_words += len(words)
                word_counter.update(words)
                
                for c in line:
                    if not (c.isalnum() or c.isspace() or c in ".,!?;:'\"()-/"):
                        weird_chars[c] += 1

        print('--- 100,000 SATIRLIK ORNEKLEM ANALIZI ---')
        actual_lines = len(lines)
        print(f'Okunan Satir Sayisi     : {actual_lines:,}')
        print(f'Ortalama Satir Uzunlugu : {total_len/actual_lines:.1f} karakter')
        print(f'Ortalama Kelime Sayisi  : {total_words/actual_lines:.1f} kelime')
        
        top_words = word_counter.most_common(20)
        print(f'\nEn Cok Gecen 20 Kelime:')
        print(', '.join([f'{w}({c})' for w, c in top_words]))

        turk_ratio = sum(char_counter[c] for c in turkish_chars) / sum(char_counter.values()) * 100
        print(f'\nTurkce Karakter (ç,ğ,ı,ö,ş,ü) Yogunlugu: %{turk_ratio:.2f}')

        print(f'\nEn Sik Gecen 15 Ozel/Garip Karakter:')
        print(weird_chars.most_common(15))

        print('\n--- RASTGELE 5 CUMLE (Kalite Kontrol) ---')
        for _ in range(5):
            print('>', random.choice(lines))
            
    except Exception as e:
        print('Hata:', e)

if __name__ == "__main__":
    main()
