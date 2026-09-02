# 64K Türkçe Tokenizer Oluşturma: Adım Adım Rehber

---

## Adım 1: Metin Dosyası Oluşturma (Streaming ve Kotalama)

Büyük veri setlerini (örn. 125 GB'lık FineWeb) diske indirmemek ve kotamızı aşmamak için **streaming** kullanılarak veriler harmanlanır.

Hedeflenen kota dağılımı ve 5GB'lık limit altında elde edilen nihai durum:

| Veri Seti | Ayrılan Kota | Sonuç |
|-----------|--------------|-------|
| **Cosmos Turkish** | 9,000,000 satır | Başarıyla çekildi (En yoğun oran) |
| **FineWeb 2 (TR)** | 3,000,000 satır | Başarıyla çekildi (Kaliteli Web) |
| **ForumSohbetleri** | 6,000,000 satır | 11 farklı alt-forumdan çekildi (Argo/Diyalog) |
| **MusteriYorumlari** | 2,000,000 satır | Başarıyla çekildi |
| **Havadis** | 1,500,000 satır | Başarıyla çekildi (Haber dili) |

> **Nihai Sonuç:** Gerçekleşen çekim sonucunda toplam **13,608,830 satırlık** ve **3.7 GB** boyutunda devasa bir `turkish_corpus_raw.txt` elde edilmiştir.

---

## Adım 1.5: RAM Dostu Parçalı Karıştırma (Chunked Shuffle)

3.7 GB'lık metni tek seferde Python hafızasına (RAM) alıp karıştırmak `MemoryError` verdiği için, veri seti önce rastgele 20 küçük parçaya ayrılıp, her parça hafızada kendi içinde karıştırıldıktan sonra tek bir dosyada birleştirilir.

```python
# step1_shuffle_only.py
# (Bu script raw dosyayı 20 parçaya böler, karıştırır ve output/turkish_corpus_shuffled.txt olarak kaydeder)
```

---

## Adım 2: SentencePiece ile 64K Tokenizer Eğitimi

Elde edilen 3.7 GB'lık, homojen şekilde karışmış `turkish_corpus_shuffled.txt` kullanılarak SentencePiece ile saf bir Türkçe BPE tokenizer (64,000 kelime dağarcığı) eğitilir.

```python
# step2_train_tokenizer.py
import sentencepiece as spm

print("Tokenizer egitimi basliyor...")
spm.SentencePieceTrainer.train(
    input="output/turkish_corpus_shuffled.txt",
    model_prefix="turkish_64k",
    vocab_size=64000,
    model_type="bpe",
    character_coverage=0.9995,
    shuffle_input_sentence=True,  # SentencePiece'in de internal karıştırma yapması için
    max_sentence_length=4192,
    input_sentence_size=5_000_000, # Bellek ve hiz dostu olması için 5 Milyon satır örneklem
    unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
    normalization_rule_name="nmt_nfkc_cf",
)
print("Egitim tamamlandi: turkish_64k.model, turkish_64k.vocab")
```
**Süre tahmini:** 1-2 saat (CPU, 5M satır örneklem için)

---

## Adım 3: Eğitilen Tokenizer'ı İncele

```python
# step3_inspect_tokenizer.py
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("turkish_64k.model")

# Test et
test_sentences = [
    "lavanta bahçesindeki güzellik inanılmazdı",
    "araba kullanmayı öğrenmek zor değildi",
    "sosyal medyada çok şey paylaşıyoruz",
    "yok artık nasıl olur bu ya ??",  # slang test
]

for sentence in test_sentences:
    tokens = sp.encode(sentence, out_type=str)
    print(f"Giriş: {sentence}")
    print(f"Tokenlar: {tokens}")
    print(f"Token sayısı: {len(tokens)}")
    print()

# En uzun tokenları gör
vocab = [(sp.id_to_piece(i), i) for i in range(sp.get_piece_size())]
long_tokens = sorted(vocab, key=lambda x: len(x[0]), reverse=True)[:50]
print("En uzun 50 token:")
for piece, idx in long_tokens:
    print(f"  {idx}: '{piece}' ({len(piece)} karakter)")
```

---

## Adım 4: Büyük Tokenizer'daki Frekans Analizi

Bu adım, orijinal büyük (multilingual) tokenizer'dan **hangi tokenlerin silineceğini** belirlemek için yapılır.
Silinecek adaylar genellikle çok nadir görülen, çok dilli uzun kelimelerdir.

```python
# step4_frequency_analysis.py
from datasets import load_dataset
from collections import Counter
from transformers import AutoTokenizer

BIG_TOKENIZER_NAME = "Cohere/Cohere-embed-multilingual-v3.0" # Örnek model
big_tokenizer = AutoTokenizer.from_pretrained(BIG_TOKENIZER_NAME)

wiki40 = load_dataset("alibayram/wikipedia-40-langs", split="train", streaming=True)
token_freq = Counter()
# Sadece ilk 1 milyon makale taranır (yaklaşık 1-2 saat sürer)
```

---

## Adım 5: BPE Tokenizer Editor ile Enjeksiyon

Bu adım **manuel/interaktif** olarak [BPE Tokenizer Editor](https://github.com/malibayram/bpe-tokenizer-editor/) aracı kullanılarak yapılır.
1. Büyük tokenizer editöre yüklenir.
2. Adım 4'te tespit edilen, frekansı 0 veya çok düşük olan nadir tokenlar silinir.
3. Adım 2'de elde ettiğimiz saf Türkçe 64K tokenlar (ve birleşme kuralları) eklenir.
4. Yeni 128K'lık "Hibrit Türk Tokenizer" kaydedilir.

---

## Adım 6: Kimlik Numarası (ID) Eşleştirme Sözlüğü

Eski tokenlerin yeni tokenizer içinde hangi ID'ye denk geldiğini bulmak için mapping dosyası oluşturulur. Bu dosya, model ağırlıklarını (embedding matrix) aktarırken hayati önem taşır (Yıkıcı unutmayı engellemek için).

```python
# step6_build_mapping_dict.py
# Bu script original_tokenizer ve new_hybrid_tokenizer'i kiyaslar.
# token_id_mapping.json uretir.
```

---

## Adım 7: Doğrulama (Validate)

Tüm süreç bittiğinde `test_en` ve `test_tr` metinleri üzerinde yeni tokenizer test edilir. İngilizce kelimelerin orijinal şekilde parçalandığından, Türkçe metinlerin ise artık kelime kelime/kökler halinde bütünsel kaldığından emin olunur.
