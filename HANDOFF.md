# Türkçe 128K Tokenizer — Devir Dokümanı

Son güncelleme: 2026-09-03. Bu doküman projeyi devralan kişinin/agent'ın bilmesi
gereken **her şeyi** içerir. `GUIDE.md` eskidir ve **yanlış bilgi içerir** (aşağıda açıklanıyor).

---

## 1. Hedef

Sıfırdan bir **genel amaçlı Türkçe 128.000 vocab'lık BPE tokenizer** eğitmek.

**Kapsam dışı:** Grafting / tokenizer cerrahisi. İlk görev tanımında "devasa çok dilli
embedding modeline Türkçe eklemek" vardı; kullanıcı bunu kapsam dışı bıraktı. Şu an
sadece bağımsız bir tokenizer üretiyoruz. Bu yüzden special token'lar bir modele
uymak zorunda **değil**.

**Neden bu tokenizer:** Resmi Türkçe kadar **gayriresmî** Türkçeyi de kapsaması
gerekiyor — argo, forum dili, diakritiksiz yazım (`dusunduk`, `her$eyin`), ürün
yorumları. Önceki denemede bu register tamamen eksikti (aşağıya bak).

---

## 2. Çalışma kuralları (kullanıcının koyduğu, uyulması zorunlu)

1. **Kod yazmadan önce sor.** Ne yapacağını, girdiyi, çıktıyı, kaç satır olacağını
   anlat, onay al. Onaysız script yazma, commit atma, push etme, HF'ye bir şey gönderme.
2. **Uzun işten önce kısa test.** 6 saatlik işe girmeden önce küçük ölçekte çalıştır,
   doğrula. Kullanıcı bu kuraldan saparsa **hatırlat**.
3. **Ara ara log al.** Uzun işlerde 5 dakikada bir ilerleme logu. Amaç: 6 saat sonra
   hata görmek yerine 10. dakikada görmek.
4. **Kod sade ve okunaklı olsun.** Over-engineering yok. Dosya adları İngilizce,
   yorumlar Türkçe. Kullanıcı kodu okuyup analiz ediyor.
5. **Tahmin etme, ölç.** Bu projede alınan her karar ölçüme dayandı.

---

## 3. Ortam

| | |
|---|---|
| GitHub | `https://github.com/ahmetege0/turkish-tokenizer` (public, `main`) |
| Lokal | `D:\MAGIBU\turkish_datasets\tokenizer_project` (Windows, Python 3.14) |
| Colab | Pro. **CPU runtime** — GPU **gereksiz** (HF `tokenizers` Rust/CPU-only, GPU kodu yok) |
| Veri | Google Drive: `/content/drive/MyDrive/tr-tokenizer-data` (kalıcı) |
| Drive alanı | 4.5 TB boş, sıkıntı yok |
| HF token | Colab Secrets'ta `HF_TOKEN` olarak tanımlı |

**Colab akışı:**
```bash
from google.colab import drive; drive.mount('/content/drive')
%cd /content/turkish-tokenizer && git pull
!python scripts/<script>.py --data /content/drive/MyDrive/tr-tokenizer-data
```

**Yüksek RAM sadece eğitimde gerekli.** Toplama ve doğrulama standart 12.7 GB'da geçti.

**`--shutdown` bayrağı `!python` alt sürecinden çalışmaz** (`runtime.unassign()` IPython
kernel istiyor). Kapatmak için hücrenin sonuna Python olarak:
```python
from google.colab import runtime; runtime.unassign()
```

---

## 4. Veri — TAMAMLANDI

`/content/drive/MyDrive/tr-tokenizer-data/`

```
shards/    21,827,681 satır  6,000,000,048 karakter  49 dosya   ← orijinal
train/     21,723,606 satır  5.97 GB                            ← eğitimde kullanılacak
holdout/      104,075 satır  30.8 MB                            ← değerlendirmede
state.json, collect.log, samples.txt, verify_report.md,
split_report.md, token_length_report.md
```

### Kaynak dağılımı (gerçekleşen)

| Kaynak | Repo | MB | Satır | Pay | Hedef | TR karakter |
|---|---|---:|---:|---:|---:|---:|
| forum | `turkish-nlp-suite/ForumSohbetleri` (11 config) | 1500.0 | 5,085,711 | %25.0 | %25 | %6.8 |
| fineweb2 | `HuggingFaceFW/fineweb-2` (`tur_Latn`) | 1321.6 | 5,784,779 | %22.0 | %20 | %8.9 |
| cosmos | `ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0` | 900.0 | 3,368,425 | %15.0 | %15 | %8.7 |
| havadis | `turkish-nlp-suite/Havadis` | 720.0 | 3,663,399 | %12.0 | %12 | %9.2 |
| ozenli | `turkish-nlp-suite/OzenliDerlem` (10 config) | 600.0 | 391,453 | %10.0 | %10 | %9.0 |
| wiki | `turkish-nlp-suite/temiz-Wiki` | 600.0 | 3,035,569 | %10.0 | %10 | %8.1 |
| akademik | `turkish-nlp-suite/AkademikDerlem` (5 config) | 300.0 | 163,176 | %5.0 | %5 | %9.4 |
| senti | `turkish-nlp-suite/SentiTurca` (3 config) | 44.7 | 172,535 | %0.7 | %1.2 | %8.3 |
| vitamin | `turkish-nlp-suite/vitamins-supplements-reviews` | 13.7 | 162,634 | %0.2 | %0.7 | %9.4 |

**Gayriresmî toplam %25.9** (forum + yorumlar). Önceki denemede **%0**'dı.
senti/vitamin hedefi tutturamadı çünkü havuzları küçük; açığı fineweb2 emdi (tasarım gereği).

forum'un TR karakter oranının düşük (%6.8) olması **hata değil** — forum kullanıcıları
diakritiksiz yazıyor. İstediğimiz register farkının kanıtı.

### Doğrulama sonuçları (`verify_corpus.py`)

- **Tekrar eden satır: 0** — bağımsız tarama, dedup doğrulandı
- Bozuk UTF-8 içeren satır: **2,374** (%0.011) — muhtemelen kaynak verinin kendisinde
  (kontrol edilmedi; `grep -h '�' shards/*.txt | head` ile bakılabilir)
- `state.json` ile fiili sayımlar birebir uyuştu

### Held-out ayrımı (`split_holdout.py`)

Hash tabanlı, **tekrar üretilebilir**: `blake2b(satır, person=b"holdout") % M == 0`.
Fiziksel ayrım — kopyalanmadı, eğitimden çıkarıldı. Kaynak etiketi iki tarafta da korunuyor.
Büyük 5 kaynak %0.46, küçük 4 kaynak 2.000 satırlık tabana takıldı.

---

## 5. Alınan tasarım kararları

| Karar | Değer | Gerekçe |
|---|---|---|
| Kütüphane | **HF `tokenizers`** (Rust) | `tokenizer.json` doğrudan `transformers`'a girer; SentencePiece dönüşümü ince hata kaynağı. **SentencePiece KULLANMIYORUZ.** |
| Model | **BPE** | Kullanıcı kararı (Unigram ölçülmedi) |
| Normalizer | **NFKC, lowercase YOK** | `.lower()` Türkçe'yi bozuyor: `İ`→`i`+ayrı nokta (1 karakter 2 oluyor), `I`→`i` (olması gereken `ı`). NFC/NFKC 12 Türkçe harfin hepsini koruyor. NFKC ayrıca `ﬁ→fi`, `Ａ→A`, `①→1` sadeleştirmesi yapıyor. |
| Pre-tokenizer | **`ByteLevel(add_prefix_space=False)`** | 256 sabit alfabe → hiçbir girdi `<unk>`'e düşmez. `byte_fallback` bayrağına gerek yok (o SentencePiece kavramı). |
| Decoder | **`ByteLevel`** | ⚠️ Eski tokenizer'da `null`'dı, `decode()` mojibake dönüyordu |
| Post-processor | **`ByteLevel`** | aynı sebep |
| `initial_alphabet` | **`ByteLevel.alphabet()`** | ⚠️ Verilmezse 256 byte'ın bir kısmı vocab'a girmez. Eski tokenizer'da **27'si eksikti**. |
| `vocab_size` | **128.000** | 5 special + 27 rezerv + 256 byte = 288; kalan **127.712** Türkçe alt-kelime |
| `min_frequency` | **2** | tek seferlik gürültüyü ele |
| `max_token_length` | **28** | ölçümle belirlendi, aşağıya bak |
| **Special token'lar** | **AÇIK** | kullanıcı araştırıyor, aşağıya bak |

### `max_token_length = 28` neden

**Kritik nokta: ByteLevel uzunluğu ≠ karakter uzunluğu.** Türkçe harfler UTF-8'de 2 byte,
ByteLevel her byte'ı bir karaktere eşliyor:

```
 ışıkları     9 karakter  →  13 ByteLevel
 açacağım     9 karakter  →  12 ByteLevel
```

`max_token_length` **ByteLevel uzayıyla** karşılaştırılır. Karakter sayısına bakıp eşik
koymak, en çok Türkçe karakter yoğun (yani en Türkçe) kelimeleri cezalandırır.

Ölçüm (138M parça, `measure_token_length.py`): p50=7, p90=12, p99=18, p99.9=23,
p99.99=28, max=725. En uzun 30 parçanın **tamamı çöp** (yapışmış menüler, `ohaaaaa...`,
`.........`, emoji serileri).

Ama sıradan uzun Türkçe kelimeler:
```
bilgisayarlaştırılmış    21 karakter → 27 ByteLevel
çalıştırabileceğimiz     20 karakter → 26 ByteLevel
sorumluluklarımızdandır  23 karakter → 27 ByteLevel
```
**25 eşiği bunların hepsini bölerdi.** 28'de hepsi tek token olabiliyor,
%99.990 kapsanıyor, 138M'de 13.219 parça kesiliyor.

### Special token'lar — AÇIK KARAR

Gerçek modellerden ölçülen veri (HF Hub config/vocab dosyalarından):

| Model | Model tipi | Pre-tokenizer | Special token'lar |
|---|---|---|---|
| RoBERTa | BPE | ByteLevel | `<s>=0 <pad>=1 </s>=2 <unk>=3 <mask>=50264` |
| GPT-2 | BPE | ByteLevel | — |
| XLM-RoBERTa | **Unigram** | Metaspace | `<s>=0 <pad>=1 </s>=2 <unk>=3 <mask>=250001` |
| multilingual-e5 | **Unigram** | Metaspace | aynı, vocab 250,002 |
| BGE-M3 | **Unigram** | Metaspace | aynı, `byte_fallback: False` |
| BERTurk | WordPiece | — | `[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4` |
| mBERT | WordPiece | — | `[PAD]=0` + **99 adet `[unused]`** rezerv |
| LaBSE | WordPiece | — | `[PAD]=0` + **99 adet `[unused]`** rezerv |

**Desen: BPE↔ByteLevel↔`<s>` tarzı, Unigram↔Metaspace, WordPiece↔`[CLS]` tarzı.**
Biz BPE+ByteLevel olduğumuz için `<s>` ailesindeyiz.

Üç seçenek sunuldu, karar verilmedi:
- **A)** `<s>=0 <pad>=1 </s>=2 <unk>=3`, `<mask>=127999` (son slot) — XLM-R düzeni
- **B)** `<s>=0 <pad>=1 </s>=2 <unk>=3 <mask>=4`, rezerv 5–31
- **C)** A + sona rezerv slotlar

**Önemli:** "Uyumluluk" argümanı pratikte geçersiz. Vocab birleştirme yöntemiyle
grafting zaten aile uyuşmazlığından (BPE vs Unigram) mümkün değil; tokenizer'ı komple
değiştirme yönteminde ise embedding matrisi sıfırdan kurulduğu için ID düzeni önemsiz.
Karar sadece **sadelik ve yaygınlık** üzerinden verilmeli.

---

## 6. Önceki denemede bulunan hatalar — TEKRARLAMA

Bu proje Ağustos 2026'da bir kez denenmişti. `GUIDE.md` o denemeyi anlatır ve
**yanıltıcıdır**. Bulunan hatalar:

1. **ForumSohbetleri hiç çekilememiş (0 satır).** `texts` alanı **liste** tipinde,
   eski kod `isinstance(v, str)` kontrolüyle her kaydı eledi. 2.78M forum kaydı
   sessizce kayboldu — projenin asıl amacı olan argo/gündelik register **%0** kaldı.
   GUIDE.md "11 alt-forumdan çekildi" diyor, log `[forum] TOPLAM: 0 satir` diyor.
2. **Eski tokenizer'ların `decode()`'u bozuk.** `decoder` ve `post_processor` `null`.
   Çıktı: `YarÄ±n ĠÄ°stanbul ' a...` — roundtrip başarısız.
3. **256 byte alfabesinin 27'si vocab'da yok.** `initial_alphabet` verilmemiş.
4. **Eski korpus kullanılamaz.** `step1.log`'a bakınca en az 3 process aynı anda aynı
   dosyaya `"w"` modunda yazmış. Satır sayısı log'la tutmuyor (13,654,065 vs 13,573,362),
   kaynak sınırları kaymış, kaynak etiketi geri kazanılamıyor.
5. **`hash()` kullanılmış** dedup'ta — Python'da string hash'i her process'te farklı,
   resume tamamen çöker. Biz `blake2b(digest_size=8)` kullanıyoruz.

Bu denemede ayrıca bulunanlar:

6. **`OzenliDerlem/Havadis` config'i `Havadis` dataset'iyle birebir aynı** → dışlandı.
7. **`SentiTurca` = MusteriYorumlari + BuyukSinema + hate** → o ikisi listeden çıkarıldı.
8. **OzenliDerlem ve AkademikDerlem metinlerinde hiç satır sonu yok.** Sadece `\n` ile
   bölüp uzun olanı atınca ozenli'nin %69'u, akademik'in %87'si kayboluyordu — üstelik
   **yanlı** şekilde (sadece kısa makaleler kalıyordu). `split_long()` ile cümle
   sınırlarından parçalanarak çözüldü.
9. **`forum/memurlar`'da ortalama 106.000 karakter/kayıt** → `MAX_CHARS_PER_RECORD=20_000`
   ile tek kaydın kotayı domine etmesi engellendi.

---

## 7. Mevcut scriptler

| Script | Ne yapar | Durum |
|---|---|---|
| `scripts/probe_sources.py` | Her repo'nun her config'ini yoklar, alan adlarını ve **tiplerini** (str/list) raporlar | ✅ çalıştı, 37 config, 0 hata |
| `scripts/collect_corpus.py` | 9 kaynaktan register-dengeli korpus toplar, çekme anında dedup | ✅ çalıştı, 55.7 dk |
| `scripts/verify_corpus.py` | Sayım, bağımsız tekrar taraması, UTF-8, kalite göstergeleri | ✅ çalıştı, 20.4 dk |
| `scripts/split_holdout.py` | Eğitim/held-out fiziksel ayrımı | ✅ çalıştı, 4.3 dk |
| `scripts/measure_token_length.py` | `max_token_length` için ölçüm | ✅ çalıştı, 9.8 dk |
| `scripts/step*.py`, `check_*.py`, `remove_duplicates.py` | **ESKİ deneme, kullanma** | ⛔ |

`collect_corpus.py` bayrakları: `--scale` (kotaları çarpar, test için 0.001),
`--fresh` (state'i yok say), `--out` (Drive yolu), `--shutdown` (Colab kapat).

**Dedup mekanizması (collect_corpus.py, satır ~136 ve ~267):**
Anahtar = `blake2b(normalize(satır), digest_size=8)`, normalize = küçük harf +
sadece harf/rakam + boşluk sıkıştırma. Dosyaya **orijinal** satır yazılır.
`seen` seti kaynaklar arasında sıfırlanmaz → dataset-içi ve dataset-arası dedup
tek mekanizma. O(1) arama. 21.8M anahtar ≈ 1.64 GB RAM (ölçüldü: 75 byte/anahtar).
Çekme sırası = öncelik sırası: nadir register önce, genel web sonra.

---

## 8. Sırada ne var

### Adım 1 — Special token kararı ⏳
Kullanıcı araştırıyor. A/B/C seçenekleri yukarıda. Karar verilmeden eğitim yapılamaz.

### Adım 2 — Smoke testi (~10 dk) 🔜
Küçük korpus (örn. 50K satır) + **~2000 vocab** ile tokenizer eğit, şunları **assert** et:
1. `decode(encode(x)) == x` — çeşitli Türkçe cümlelerde.
   ⚠️ NFKC uyguladığımız için karşılaştırma **NFKC-normalize edilmiş** metinle yapılmalı,
   ham metinle değil (`ﬁnal` → `final` dönecek).
2. **256 byte alfabesinin tamamı** vocab'da (eski tokenizer'da 27'si eksikti)
3. Special token'lar beklenen ID'lerde
4. `İSTANBUL` / `istanbul` / `İstanbul` doğru işleniyor, normalizer `İ`'yi bozmuyor
5. Emoji, bozuk byte, İngilizce karışık metin `<unk>` üretmiyor
6. `PreTrainedTokenizerFast` ile yüklenebiliyor
7. Hiçbir token 28 ByteLevel karakterden uzun değil

**Bu test geçmeden 128K eğitimine girilmez.**

### Adım 3 — RAM tavanı ölçümü (~30 dk) 🔜
8K → 16K → 32K vocab ile aynı korpus alt kümesinde tepe RAM ve süre ölç, 128K'ya
ekstrapole et. Amaç: yüksek RAM runtime'ında kaç GB gerekeceğini bilmek.
**Risk:** Türkçe eklemeli olduğu için benzersiz kelime tipi sayısı çok yüksek;
`tokenizers` trainer'ı bunların hepsini frekansıyla bellekte tutuyor. 12.7 GB standart
RAM'in yetmemesi kuvvetle muhtemel.
**Not:** Alt küme alırken tüm kaynaklardan orantılı örnekle (dosya adı kaynak etiketi taşıyor);
ilk N dosyayı almak temsili olmaz.

### Adım 4 — 128K eğitimi (3-5 saat) 🔜
Colab **yüksek RAM** CPU runtime. `train_from_iterator` ile `data/train/*.txt` okunur.
Ara log şart. `--shutdown` benzeri kapatma mekanizması kurulmalı (yukarıdaki kernel notu).

### Adım 5 — Değerlendirme 🔜
`data/holdout/` üzerinde, **kaynak bazında ayrı ayrı**:
- fertility (kelime başına token)
- compression ratio
- continued-word rate
- Baseline karşılaştırma: **BERTurk** (`dbmdz/bert-base-turkish-cased`),
  **Cosmos**, **XLM-R** (`xlm-roberta-base`)
- Morfolojik hizalanma (Zemberek/TrMor) — ilk görev tanımında vardı, **henüz planlanmadı**

### Adım 6 — Paketleme 🔜
`tokenizer.json` + `tokenizer_config.json`, `PreTrainedTokenizerFast` yüklenebilirlik
testi, rapor.

---

## 9. Bilinen açıklar / eksikler

- **Referans makaleye bakılmadı:** *"Optimal Turkish Subword Strategies at Scale"*
  (turkish-nlp-suite). Vocab boyutu ve morfoloji konusunda ampirik veri içerdiği
  söyleniyor. Kullanıcı "şimdilik kalsın" dedi. Kararlarımızla çelişebilir.
- **BPE vs Unigram ölçülmedi.** Kullanıcı doğrudan BPE dedi.
- **Havadis'e özel boilerplate filtresi yazılmadı.** Genel filtreler (uzunluk,
  kelime sayısı, harf oranı) + FineWeb2'nin `minhash_cluster_size`'ı var.
- **Dil filtresi sadece FineWeb2'de** (`language_score >= 0.90`). Diğer 8 kaynak
  küratörlü Türkçe dataset olduğu için filtresiz.
- **2.374 satırdaki `�` karakterinin kaynağı belirlenmedi** (%0.011, muhtemelen kaynak veri).
- **Near-duplicate (MinHash) taraması yapılmadı.** Bilinçli: tokenizer eğitimi sadece
  birleşme frekans istatistiği istiyor, az miktarda yakın-tekrar merge'leri kaydırmıyor.
  FineWeb2 için zaten `minhash_cluster_size` kullanıldı.
- **`--shutdown` alt süreçten çalışmıyor** (düzeltildi: artık çökmüyor, talimat basıyor).

---

## 10. Ölçülmüş sayılar (referans)

| | |
|---|---|
| Korpus | 21,827,681 satır / 6,000,000,048 karakter / 275 karakter per satır |
| Dedup seti | 21.8M anahtar ≈ 1.64 GB RAM (75 byte/anahtar, ölçüldü) |
| Toplama süresi | 55.7 dk (Colab CPU, standart RAM, Drive'a yazarak) |
| Drive yazma hızı | Colab yerel diskiyle **aynı** — darboğaz değil |
| Eski 100K eğitimi | 3 GB korpus, ~78 dk (lokal makine, 16 GB RAM) |
| ByteLevel şişmesi | Türkçe harf başına +1 karakter (UTF-8'de 2 byte) |
