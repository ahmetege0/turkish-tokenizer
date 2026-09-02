import sys
import io
from tokenizers import Tokenizer

# PowerShell / Windows terminal encoding problemini cozmek icin (Ġ karakteri icin)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    try:
        tokenizer = Tokenizer.from_file("output/hf_turkish_base_50k.json")
    except Exception as e:
        print("Hata: Tokenizer yuklenemedi.", e)
        return
        
    test_sentences = [
        "lavanta bahçesindeki güzellik inanılmazdı",
        "araba kullanmayı öğrenmek zor değildi",
        "sosyal medyada çok şey paylaşıyoruz",
        "yok artık nasıl olur bu ya ??",
        "kardiyovasküler sistem hastalıkları",
        "müteselsilen sorumlu tutulmasına karar verilmiştir"
    ]
    
    print("--- 50K Temel Tokenizer Testi ---\n")
    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        print(f"Giriş: {sentence}")
        print(f"Tokenlar: {encoded.tokens}")
        print(f"Token Sayısı: {len(encoded.tokens)}\n")
        
    print("--- En Uzun 20 Token (Neler Öğrenmiş?) ---")
    vocab = tokenizer.get_vocab()
    # Kelimeleri uzunluklarina gore sirala
    long_tokens = sorted(vocab.keys(), key=len, reverse=True)
    
    # Ilk 20 tokeni ekrana bas
    count = 0
    for token in long_tokens:
        if token.startswith("Ġ"): # HF ByteLevel space isareti
            display_token = token[1:]
        else:
            display_token = token
            
        # Asiri uzun anlamsiz cizgileri/bosluklari atlayalim ki kelime gorelim
        if len(display_token) > 5 and display_token.isalpha():
            print(f"'{display_token}' ({len(display_token)} harf)")
            count += 1
            if count >= 20:
                break

if __name__ == "__main__":
    main()
