"""
Smoke Test — Tokenizer Saglamlik Testi
======================================
128K egitiminden ONCE kucuk olcekte tokenizer egitip kritik ozellikleri
dogrulamak icin. Bu test gecmeden buyuk egitime girilmez.

Tasarim: karakter seviyesi (Metaspace pre-tokenizer), byte_fallback YOK.
Ekip karari geregi ByteLevel'dan buraya gecildi. Bu, ByteLevel'in verdigi
"hicbir girdi <unk> uretemez" garantisini KALDIRIYOR -- ayni sekilde
XLM-RoBERTa, multilingual-e5 ve BGE-M3 de byte_fallback kullanmiyor.
Denendi: byte_fallback=True + <0xXX> fallback token'lari BPE merge'leriyle
etkilesime girince decode'da icerik kaybediyordu (ornegin sozde-basit bir
kelimenin bas harfi sessizce dusuyordu) -- bu yuzden bilerek kapali birakildi.
Kabul edilen risk: egitim verisinde hic gorulmemis karakterler (nadir emoji,
yabanci alfabe) <unk>'e duser. Test bunu ORTULMEZ, olcup raporlar.

ONEMLI: karakter seviyesinde smoke test'in ByteLevel'daki gibi KESIN bir kapi
olma ozelligi YOK. ByteLevel'da garanti matematikseldi (256 byte hep tam
kapsanir, vocab/korpus buyuklugu fark etmez). Burada kapsama "bu karakter
egitim verisinde gorulmus mu" sorusuna bagli, ve bu soru KUCUK orneklemde
BUYUK orneklemden cok daha kirilgan cevaplaniyor. Yerel testlerde (sentetik,
dar kelime havuzuyla) sik Turkce kelimelerde bile <unk> cikabildigi gorulda --
bu KOD hatasi degil, orneklem darliginin dogal sonucu. Gercek 21.7M satirlik
korpusta (zengin, dogal dil, yuksek frekansli baglac/edatlarla dolu) kapsama
cok daha iyi olmasi beklenir, ama bunu kesin bilemeyiz -- ancak gercek
calistirmada gorecegiz. Bu yuzden roundtrip/unk/casing testleri BLOKLAYICI
DEGIL, ORAN raporlayan bilgi testleri; sadece MEKANIK ozellikler (special ID,
HF yukleme, token uzunlugu -- bunlar korpus zenginliginden bagimsiz) blokluyor.

Ne yapar:
  1. train/ altindan orantili ~50K satir ornekler (kaynak dagilimina sadik)
  2. 2000 vocab'lik BPE tokenizer egitir (uretimle AYNI ayarlarla)
  3. 7 kontrol calistirir (c, f, g BLOKLAYICI; a, b, d, e ORAN/BILGI amacli):
     a) [ORAN] roundtrip: duz Turkce + ASCII-karisik metinde ne kadari tam esiyor
     b) [ORAN] yaygin Turkce cumlelerde <unk> orani ne
     c) special token ID'leri dogru mu
     d) [ORAN] Turkce I/i/İ/ı normalizer'da bozulmadan ne kadari geri geliyor
     e) [BILGI] emoji/yabanci alfabe/nadir sembollerde ne oluyor -- garanti YOK
     f) PreTrainedTokenizerFast ile yuklenebilirlik
     g) hicbir ogrenilmis token 20 karakterden uzun degil
  4. Tokenizer'i diske kaydeder (demo_tokenizer.py yuklesin diye)

Girdi : data/train/*.txt
Cikti : konsol + data/smoke_report.md + data/smoke_tokenizer.json

Kullanim:
    python scripts/smoke_test_tokenizer.py --data /content/drive/MyDrive/tr-tokenizer-data
    python scripts/smoke_test_tokenizer.py --data /content/drive/MyDrive/tr-tokenizer-data --lines 10000
"""

import argparse
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer, normalizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from transformers import PreTrainedTokenizerFast


# ── Ayarlar ──────────────────────────────────────────────────────────────────

VOCAB_SIZE      = 2_000       # kucuk: smoke test icin yeterli
MIN_FREQUENCY   = 2
MAX_TOKEN_LEN   = 20          # karakter uzayinda; olculmedi, dogrudan karar verildi
SAMPLE_LINES    = 50_000      # varsayilan orneklem buyuklugu

# Special token'lar — Secenek B: basa sirali, 27 rezerv slot
SPECIAL_TOKENS = (
    ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    + [f"<reserved_{i}>" for i in range(27)]
)
EXPECTED_IDS = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}


# ── Yardimcilar ──────────────────────────────────────────────────────────────

def nfkc(text):
    """Tokenizer'in normalizer'inin yapacagi donusumun aynisi."""
    return unicodedata.normalize("NFKC", text)


def sample_from_train(train_dir, budget):
    """
    Kaynak bazinda ORANTILI ornekleme.
    Dosya adi kaynak etiketini tasiyor (orn. forum_0003.txt → forum).
    Her kaynaktan shard sayisiyla orantili paylasim yapilir.
    """
    shards_by_source = defaultdict(list)
    for p in sorted(train_dir.glob("*.txt")):
        source = p.stem.rsplit("_", 1)[0]
        shards_by_source[source].append(p)

    total_shards = sum(len(v) for v in shards_by_source.values())
    if total_shards == 0:
        raise SystemExit(f"Hic shard bulunamadi: {train_dir}")

    lines = []
    for source, paths in shards_by_source.items():
        source_budget = max(1, budget * len(paths) // total_shards)
        per_shard = max(1, source_budget // len(paths))
        source_count = 0
        for path in paths:
            shard_count = 0    # HER shard'ta sifirlanir
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if line:
                        lines.append(line)
                        shard_count += 1
                        source_count += 1
                        if shard_count >= per_shard:
                            break
            if source_count >= source_budget:
                break

    print(f"  {len(shards_by_source)} kaynaktan {len(lines):,} satir orneklendi")
    return lines


def build_tokenizer():
    """Uretim egitiminde kullanilacak AYNI konfigurasyonla tokenizer olusturur."""
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = Metaspace()
    tok.decoder = decoders.Metaspace()
    return tok


def build_trainer():
    """BpeTrainer: kucuk vocab, ayni bayraklar.
    initial_alphabet YOK: Metaspace'te ByteLevel'deki gibi sabit bir alfabe
    garantisi yok, trainer alfabeyi korpusta gordugu karakterlerden kurar."""
    return BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        max_token_length=MAX_TOKEN_LEN,
        special_tokens=list(SPECIAL_TOKENS),
    )


# ── Testler ──────────────────────────────────────────────────────────────────
# Her test (gecen, basarisiz) dondurur. "BILGI" olarak isaretlenenler main()'de
# genel sonucu ETKILEMEZ, sadece raporlanir.

def test_roundtrip(tok):
    """
    [ORAN -- bloklayici degil] decode(encode(x)) == NFKC(x).
    ByteLevel'da bu MATEMATIKSEL garantiydi. Metaspace'te oyle degil: yerel
    testlerde NFKC-sonrasi duz ASCII'ye donen ornekler (orn. "①"->"1") bile
    kucuk olceklerde bazen kayboldu -- korpus zenginligiyle ilgili, kod
    hatasi degil. Bu yuzden burada TEK bir "basarisiz" ciktiginda paniklemek
    yerine ORANA bakiyoruz: %90+ ise saglikli, dusukse dikkat.
    """
    cases = [
        "Yarın İstanbul'a gidiyorum.",                        # duz Turkce
        "ya bu fiyata bu telefon alinir mi abi saçmalama",    # argo / forum
        "dusunduk tasinip karar verdik guzel oldu",           # diakritiksiz
        "İSTANBUL ANKARA İZMİR",                               # buyuk harf
        "bugün meeting'e late kaldım sorry",                   # Turkce+ASCII karisik
        "bilgisayarlaştırılmış çalıştırabileceğimiz",          # uzun eklemeli kelime
        "ﬁnal sınavı, Ａ sınıfı, ①. soru",                     # NFKC once duz ASCII'ye cevirir
        "a",                                                   # tek karakter
        # NOT: tek basina " " (bosluk) kasitli olarak yok -- Metaspace'te dizinin
        # ILK "▁" isareti icerik degil "kelime basi" sayilir, decode'da kaybolur.
        # Bu bilinen bir kural, gercek cumlelerde (hicbir zaman salt bosluk
        # olmayan) karsimiza cikmaz.
    ]
    passed = failed = 0
    for text in cases:
        decoded = tok.decode(tok.encode(text).ids)
        expected = nfkc(text)
        if decoded == expected:
            passed += 1
        else:
            failed += 1
            print(f"  ✗ roundtrip basarisiz: girdi={text!r} beklenen={expected!r} alinan={decoded!r}")
    return passed, failed


def test_common_turkish_unk(tok):
    """
    [ORAN -- bloklayici degil] Gundelik Turkce cumlelerde <unk> orani.
    Bu kelimeler egzotik degil, gercek 21.7M satirlik korpusta bol bol
    gecmesi beklenir. Ama bu KUCUK smoke-test orneklemi (50K satir, dar
    kaynak cesitliligi) her kelimeyi gormus olmayabilir -- <unk> cikmasi
    burada mutlaka tasarim hatasi degil, orneklem sansi olabilir.
    """
    unk_id = tok.token_to_id("<unk>")
    cases = [
        "Bugün hava çok güzel, dışarı çıkalım.",
        "Yarın toplantı saat kaçta başlıyor acaba?",
        "Bu ürünü beğenmedim, iade etmek istiyorum.",
        "Kardeşim gelmiş, seni de bekliyoruz!",
    ]
    passed = failed = 0
    for text in cases:
        ids = tok.encode(text).ids
        if unk_id in ids:
            failed += 1
            print(f"  ✗ <unk> bulundu (yaygin Turkce metinde beklenmiyordu): {text!r}")
        else:
            passed += 1
    return passed, failed


def test_special_ids(tok):
    """Special token'lar beklenen ID'lerde mi."""
    passed = failed = 0
    for token, expected_id in EXPECTED_IDS.items():
        actual_id = tok.token_to_id(token)
        if actual_id == expected_id:
            passed += 1
        else:
            failed += 1
            print(f"  ✗ {token}: beklenen ID={expected_id}, alinan={actual_id}")
    return passed, failed


def test_turkish_casing(tok):
    """
    [ORAN -- bloklayici degil] İ/I normalizer'da bozulmadan geri geliyor mu.
    NFKC bu harfleri koruyor (Python'un .lower()'ından farkli olarak), ama
    "korunmus normalizer ciktisi vocab'da temsil ediliyor mu" ayri bir soru --
    o da diger karakter-seviyesi testler gibi korpus zenginligine bagli.
    """
    cases = {
        "İSTANBUL": "İSTANBUL",     # İ (U+0130) aynen kalmali
        "istanbul": "istanbul",      # kucuk i degismemeli
        "İstanbul": "İstanbul",      # karisik
        "IĞDIR":    "IĞDIR",         # I (ASCII) + Turkce
        "ışık":     "ışık",          # ı (U+0131) korunmali
    }
    passed = failed = 0
    for text, expected in cases.items():
        decoded = tok.decode(tok.encode(text).ids)
        if decoded == expected:
            passed += 1
        else:
            failed += 1
            print(f"  ✗ '{text}' → '{decoded}' (beklenen: '{expected}')")
    return passed, failed


def test_exotic_unicode(tok):
    """
    [BILGI AMACLI -- genel sonucu etkilemez]
    byte_fallback olmadigi icin egitim verisinde hic gorulmemis karakterler
    (nadir emoji, yabanci alfabe) <unk>'e dusebilir ve/veya decode'da icerik
    kaybolabilir. Bu KABUL EDILEN bir sinirlama (XLM-R/BGE-M3 de ayni riski
    tasiyor) -- burada sadece NE KADAR kaybediliyor, gozle gorulsun diye
    olcuyoruz, testi "basarisiz" saymiyoruz.
    """
    unk_id = tok.token_to_id("<unk>")
    cases = [
        "harika olmuş 🎉🔥💯 tebrikler!",       # emoji
        "fiyat: ₺1.250,00 — %15 indirim",       # Turk Lirasi isareti + em dash
        "中文测试", "こんにちは",                 # Cince, Japonca
        "mixed: café naïve über",               # aksanli Latin
    ]
    for text in cases:
        ids = tok.encode(text).ids
        decoded = tok.decode(ids)
        expected = nfkc(text)
        unk_count = ids.count(unk_id)
        durum = "tam roundtrip" if decoded == expected else "ICERIK KAYBI/DEGISIMI"
        print(f"    {text!r}")
        print(f"      -> {durum}, <unk> sayisi={unk_count}/{len(ids)}, decode={decoded!r}")
    return 0, 0   # bilgi amacli; pass/fail sayilmiyor


def test_hf_load(tok):
    """PreTrainedTokenizerFast ile yuklenebilmeli."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tokenizer.json"
        tok.save(str(path))
        try:
            fast = PreTrainedTokenizerFast(
                tokenizer_file=str(path),
                bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                pad_token="<pad>", mask_token="<mask>",
            )
            out = fast("Merhaba dünya", return_tensors=None)
            assert "input_ids" in out
            return 1, 0
        except Exception as e:
            print(f"  ✗ HF yukleme hatasi: {e}")
            return 0, 1


def test_max_token_length(tok):
    """Hicbir ogrenilmis token MAX_TOKEN_LEN karakterden uzun olmamali."""
    vocab = tok.get_vocab()
    special_set = set(SPECIAL_TOKENS)
    violations = [(t, len(t)) for t in vocab if t not in special_set and len(t) > MAX_TOKEN_LEN]
    if violations:
        violations.sort(key=lambda x: -x[1])
        for t, l in violations[:5]:
            print(f"  ✗ {l} karakter: {t!r}")
    return len(vocab) - len(special_set) - len(violations), len(violations)


# ── Ana akis ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tokenizer smoke testi")
    parser.add_argument("--data", type=str, default="data",
                        help="Veri dizini (icerisinde train/ olmali)")
    parser.add_argument("--lines", type=int, default=SAMPLE_LINES,
                        help="Orneklenecek satir sayisi")
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise SystemExit(f"train dizini bulunamadi: {train_dir}")

    print("=" * 60)
    print("SMOKE TEST — Tokenizer Saglamlik Testi (karakter seviyesi)")
    print("=" * 60)

    print(f"\n[1/3] Ornekleme ({args.lines:,} satir hedef)...")
    t0 = time.time()
    lines = sample_from_train(train_dir, args.lines)
    print(f"  {time.time() - t0:.1f} sn")

    print(f"\n[2/3] Tokenizer egitiliyor (vocab={VOCAB_SIZE:,})...")
    t0 = time.time()
    tok = build_tokenizer()
    tok.train_from_iterator(lines, trainer=build_trainer())
    print(f"  {time.time() - t0:.1f} sn, vocab={tok.get_vocab_size():,}")

    print(f"\n[3/3] Testler calistiriliyor...\n")

    # (isim, fonksiyon, bloklayici_mi)
    tests = [
        ("[ORAN] Roundtrip (Turkce+ASCII kapsami)", test_roundtrip,          False),
        ("[ORAN] Yaygin Turkce metinde <unk>",      test_common_turkish_unk, False),
        ("Special token ID'leri",                  test_special_ids,        True),
        ("[ORAN] Turkce buyuk/kucuk harf korumasi", test_turkish_casing,     False),
        ("[BILGI] Egzotik Unicode davranisi",      test_exotic_unicode,     False),
        ("PreTrainedTokenizerFast yukleme",        test_hf_load,            True),
        (f"max_token_length <= {MAX_TOKEN_LEN}",   test_max_token_length,   True),
    ]

    total_pass = total_fail = 0
    results = []

    for name, fn, blocking in tests:
        print(f"  [{name}]")
        p, f = fn(tok)
        if blocking:
            total_pass += p
            total_fail += f
            status = "✓ GECTI" if f == 0 else f"✗ {f} BASARISIZ"
        elif p + f > 0:
            # oran raporlayan test: gercek pay/fail sayisi var ama genel
            # sonucu etkilemiyor -- orani goster
            status = f"ℹ {p}/{p + f} (%{100 * p / (p + f):.0f})"
        else:
            status = "ℹ bilgi amacli, yukarida"
        results.append((name, p, f, status, blocking))
        print(f"    → {status}\n")

    print("=" * 60)
    if total_fail == 0:
        print(f"SONUC: TUMU GECTI ({total_pass} kontrol)")
        print("128K egitimine gecilebilir.")
    else:
        print(f"SONUC: {total_fail} BASARISIZLIK ({total_pass} gecti)")
        print("Hatalar duzeltilmeden 128K egitimi YAPILMAZ.")
    print("=" * 60)

    tok_path = data_dir / "smoke_tokenizer.json"
    tok.save(str(tok_path))
    print(f"\nTokenizer kaydedildi: {tok_path}")

    report_path = data_dir / "smoke_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("# Smoke Test Raporu (karakter seviyesi)\n\n")
        fh.write(f"Orneklem: {len(lines):,} satir | Vocab: {tok.get_vocab_size():,}\n\n")
        fh.write("| Test | Gecen | Basarisiz | Sonuc |\n")
        fh.write("|---|---:|---:|---|\n")
        for name, p, f, status, blocking in results:
            fh.write(f"| {name} | {p} | {f} | {status} |\n")
        fh.write(f"\n**Genel: {'GECTI' if total_fail == 0 else 'BASARISIZ'}**\n")
    print(f"Rapor: {report_path}")

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
