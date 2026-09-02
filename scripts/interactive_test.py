import sys
import io
import os
from tokenizers import Tokenizer
import glob

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("==================================================")
    print("Turkce Tokenizer Modelleri - Interaktif Test Araci")
    print("==================================================")
    
    # output klasorundeki tum json modellerini bul
    models = glob.glob("output/*.json")
    if not models:
        print("Hata: 'output' klasorunde hic .json uzantili tokenizer bulunamadi.")
        return
        
    print("Mevcut Tokenizer Modelleri:")
    for i, model_path in enumerate(models):
        print(f"[{i+1}] {os.path.basename(model_path)}")
        
    try:
        choice = int(input("\nHangi modeli test etmek istiyorsunuz? (Numara girin): ")) - 1
        if choice < 0 or choice >= len(models):
            print("Gecersiz secim.")
            return
        selected_model = models[choice]
    except Exception:
        print("Gecersiz numara girdiniz. Cikis yapiliyor.")
        return
        
    try:
        tokenizer = Tokenizer.from_file(selected_model)
        print(f"\n>> Basarili: {os.path.basename(selected_model)} yuklendi!\n")
    except Exception as e:
        print("Hata: Tokenizer yuklenemedi.", e)
        return
        
    print("Istediğiniz kelimeyi veya cumleyi yazip Enter'a basin.")
    print("Cikmak icin 'cikis' veya 'q' yazabilirsiniz.\n")
    
    while True:
        try:
            user_input = input("Metin girin: ")
            
            if user_input.strip().lower() in ['q', 'quit', 'cikis', 'çıkış']:
                print("Cikis yapiliyor...")
                break
                
            if not user_input.strip():
                continue
                
            encoded = tokenizer.encode(user_input)
            print(f"Token Sayisi : {len(encoded.tokens)}")
            print(f"Tokenlar     : {encoded.tokens}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Hata: {e}")

if __name__ == "__main__":
    main()
