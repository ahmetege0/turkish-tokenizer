from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC, Sequence
import time

def get_training_corpus():
    with open("output/turkish_corpus_shuffled_unique.txt", "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            clean_line = line.strip()
            if clean_line:
                batch.append(clean_line)
            if len(batch) >= 10000:
                yield batch
                batch = []
        if batch:
            yield batch

def main():
    print("--- Adim 2: Hugging Face BPE ile 100K Temel Turkce Tokenizer Egitimi ---")
    
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # Egitim ayarlari (Vocab Size: 100.000)
    trainer = BpeTrainer(
        vocab_size=100000,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        show_progress=True
    )
    
    print("Egitim basladi. 100K kelime bulacagi icin islem 15-20 dakika surebilir...")
    t0 = time.time()
    
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    output_file = "output/hf_turkish_base_100k.json"
    tokenizer.save(output_file)
    
    elapsed = time.time() - t0
    print(f"\nEgitim tamamlandi! ({elapsed / 60:.1f} dakika surdu)")
    print(f"Cikti dosyasi: {output_file}")

if __name__ == "__main__":
    main()
