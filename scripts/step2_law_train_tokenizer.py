"""
ADIM B: Hugging Face BPE ile 15K Hukuk Tokenizer Egitimi
=========================================================
100K'lik Genel Turkce Tokenizer (hf_turkish_base_100k.json) ile ayni
BPE parametreleriyle (Normalizer, PreTokenizer, Special Tokens) egitilir
ki iki modul birbiriyle %100 uyumlu olsun.

Girdi : output/turkish_law_corpus.txt  (step1_law_collect_corpus.py ciktisi)
Cikti : output/hf_turkish_law_15k.json

Kullanim: python step2_law_train_tokenizer.py
"""

import time
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC, Sequence

CORPUS_FILE = Path(__file__).parent.parent / "output" / "turkish_law_corpus.txt"
OUTPUT_FILE = Path(__file__).parent.parent / "output" / "hf_turkish_law_15k.json"

VOCAB_SIZE = 15_000
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]
BATCH_SIZE = 10_000  # RAM'i sismesin diye dosya parca parca (iterator ile) okunur


def get_training_corpus():
    """Korpus dosyasini tek seferde bellege almadan, BATCH_SIZE'lik
    satir gruplari halinde okuyup verir (train_from_iterator bunu bekler).
    """
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            clean_line = line.strip()
            if clean_line:
                batch.append(clean_line)
            if len(batch) >= BATCH_SIZE:
                yield batch
                batch = []
        if batch:
            yield batch


def main():
    print("--- Adim B: Hugging Face BPE ile 15K Hukuk Tokenizer Egitimi ---")

    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"{CORPUS_FILE} bulunamadi. Once 'python step1_law_collect_corpus.py' calistirilmali."
        )

    # BPE modeli: bilinmeyen token icin <unk> kullanilir (100K modelle ayni)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    print(f"Egitim basladi. Girdi: {CORPUS_FILE}")
    print(f"Hedef vocab: {VOCAB_SIZE:,} | Ozel tokenlar: {SPECIAL_TOKENS}")
    t0 = time.time()

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUTPUT_FILE))

    elapsed = time.time() - t0
    print(f"\nEgitim tamamlandi! ({elapsed / 60:.1f} dakika surdu)")
    print(f"Ogrenilen vocab boyutu: {tokenizer.get_vocab_size():,}")
    print(f"Cikti dosyasi: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
