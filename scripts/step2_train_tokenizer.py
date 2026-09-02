import sentencepiece as spm
from pathlib import Path
import time

def main():
    input_file = Path("output/turkish_corpus_shuffled_unique.txt")
    if not input_file.exists():
        print(f"HATA: {input_file} bulunamadi!")
        return

    print("--- Adim 2: 64K Turkce Tokenizer Egitimi Basliyor ---")
    print(f"Girdi dosyasi: {input_file}")
    print("Not: Bu islem islemcinize (CPU) bagli olarak 1 ila 2 saat surebilir. Lutfen pencereyi kapatmayin.\n")
    
    t0 = time.time()
    
    # Egitim Parametreleri
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix="output/turkish_64k",
        vocab_size=64000,
        model_type="bpe",
        character_coverage=0.9995,
        shuffle_input_sentence=True,
        max_sentence_length=4192,
        input_sentence_size=5_000_000, # 10 milyonun 5 milyonunu rastgele secer
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
        normalization_rule_name="nmt_nfkc_cf",
    )
    
    elapsed = time.time() - t0
    print(f"\nEgitim basariyla tamamlandi! ({elapsed / 60:.1f} dakika surdu)")
    print("Cikti dosyalari: output/turkish_64k.model, output/turkish_64k.vocab")

if __name__ == "__main__":
    main()
