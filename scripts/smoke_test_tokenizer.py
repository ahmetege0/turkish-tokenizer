"""
Smoke Test — Tokenizer Saglamlik Testi
======================================
128K egitiminden ONCE kucuk olcekte tokenizer egitip kritik ozellikleri
dogrulamak icin. Bu test gecmeden buyuk egitime girilmez.

Ne yapar:
  1. train/ altindan orantili ~50K satir ornekler (kaynak dagilimine sadik)
  2. 2000 vocab'lik BPE tokenizer egitir (ayni ayarlarla: NFKC, ByteLevel, vb.)
  3. 7 assert calistirir:
     a) roundtrip:  decode(encode(x)) == NFKC(x)
     b) 256 byte alfabesi tamam mi
     c) special token ID'leri dogru mu
     d) Turkce I/i/İ/ı normalizer'da bozulmuyor mu
     e) emoji, bozuk byte, karisik metin → <unk> yok
     f) PreTrainedTokenizerFast ile yuklenebilirlik
     g) hicbir token 28 ByteLevel karakterden uzun degil
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

from tokenizers import Tokenizer, pre_tokenizers, normalizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast


# ── Ayarlar ──────────────────────────────────────────────────────────────────

VOCAB_SIZE      = 2_000       # kucuk: smoke test icin yeterli
MIN_FREQUENCY   = 2
MAX_TOKEN_LEN   = 28          # ByteLevel uzayinda, olcumle belirlendi
SAMPLE_LINES    = 50_000      # varsayilan orneklem buyuklugu

# Special token'lar — Secenek B: basa sirali, 27 rezerv slot
SPECIAL_TOKENS = (
    ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    + [f"<reserved_{i}>" for i in range(27)]
)
# Beklenen ID'ler
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
        # bu kaynaga dusen pay
        source_budget = max(1, budget * len(paths) // total_shards)
        per_shard = max(1, source_budget // len(paths))
        source_count = 0
        for path in paths:
            shard_count = 0    # HER shard'ta sifirlanir (eskiden sifirlanmiyordu:
                                # ilk shard'tan sonrakiler ~1 satirda kesiliyordu)
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
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


def build_trainer():
    """BpeTrainer: kucuk vocab, ayni bayraklar."""
    return BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        max_token_length=MAX_TOKEN_LEN,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=ByteLevel.alphabet(),
    )


# ── Testler ──────────────────────────────────────────────────────────────────

def test_roundtrip(tok):
    """decode(encode(x)) == NFKC(x) — cesitli Turkce cumlelerde."""
    cases = [
        # duez Turkce
        "Yarın İstanbul'a gidiyorum.",
        # argo / forum
        "ya bu fiyata bu telefon alinir mi abi saçmalama",
        # diakritiksiz
        "dusunduk tasinip karar verdik guzel oldu",
        # buyuk harf
        "İSTANBUL ANKARA İZMİR",
        # karisik dil
        "bugün meeting'e late kaldım sorry",
        # emoji
        "harika olmuş 🎉🔥💯 tebrikler!",
        # sayilar ve ozel karakterler
        "fiyat: ₺1.250,00 — %15 indirim",
        # NFKC donusumu olacak ornek
        "ﬁnal sınavı, Ａ sınıfı, ①. soru",
        # uzun agglutinatif kelime
        "bilgisayarlaştırılmış çalıştırabileceğimiz",
        # bos yakin ve tek karakter
        "a",
        " ",
    ]
    passed = 0
    failed = 0
    for text in cases:
        encoded = tok.encode(text)
        decoded = tok.decode(encoded.ids)
        expected = nfkc(text)
        if decoded == expected:
            passed += 1
        else:
            failed += 1
            print(f"  ✗ roundtrip basarisiz:")
            print(f"    girdi   : {text!r}")
            print(f"    beklenen: {expected!r}")
            print(f"    alinan  : {decoded!r}")
    return passed, failed


def test_byte_alphabet(tok):
    """256 byte'in tamami vocab'da olmali (eski tokenizer'da 27'si eksikti)."""
    vocab = tok.get_vocab()
    expected = set(ByteLevel.alphabet())
    missing = expected - set(vocab.keys())
    if missing:
        print(f"  ✗ {len(missing)} byte eksik: {sorted(missing)[:10]}...")
    return len(expected) - len(missing), len(missing)


def test_special_ids(tok):
    """Special token'lar beklenen ID'lerde mi."""
    passed = 0
    failed = 0
    for token, expected_id in EXPECTED_IDS.items():
        actual_id = tok.token_to_id(token)
        if actual_id == expected_id:
            passed += 1
        else:
            failed += 1
            print(f"  ✗ {token}: beklenen ID={expected_id}, alinan={actual_id}")
    return passed, failed


def test_turkish_casing(tok):
    """İ/I normalizer'da bozulmamali. NFKC bu harfleri koruyor."""
    cases = {
        "İSTANBUL": "İSTANBUL",     # İ (U+0130) aynen kalmali
        "istanbul": "istanbul",      # kucuk i degismemeli
        "İstanbul": "İstanbul",      # karisik
        "IĞDIR":    "IĞDIR",         # I (ASCII) + Turkce
        "ışık":     "ışık",          # ı (U+0131) korunmali
    }
    passed = 0
    failed = 0
    for text, expected in cases.items():
        decoded = tok.decode(tok.encode(text).ids)
        if decoded == expected:
            passed += 1
        else:
            failed += 1
            print(f"  ✗ '{text}' → '{decoded}' (beklenen: '{expected}')")
    return passed, failed


def test_no_unk(tok):
    """Hicbir girdi <unk> token'i uretmemeli (byte-level BPE garantisi)."""
    unk_id = tok.token_to_id("<unk>")
    cases = [
        "emoji: 🇹🇷🎶",
        "中文测试",                        # Cince
        "こんにちは",                       # Japonca
        "mixed: café naïve über",
        "\x00\x01\xff",                    # ham byte'lar
        "formül: E=mc² ∑∫∂",
    ]
    passed = 0
    failed = 0
    for text in cases:
        ids = tok.encode(text).ids
        if unk_id in ids:
            failed += 1
            print(f"  ✗ <unk> bulundu: {text!r}")
        else:
            passed += 1
    return passed, failed


def test_hf_load(tok):
    """PreTrainedTokenizerFast ile yuklenebilmeli."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tokenizer.json"
        tok.save(str(path))

        try:
            fast = PreTrainedTokenizerFast(
                tokenizer_file=str(path),
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
            )
            # basit bir encode/decode dene
            out = fast("Merhaba dünya", return_tensors=None)
            assert "input_ids" in out
            return 1, 0
        except Exception as e:
            print(f"  ✗ HF yukleme hatasi: {e}")
            return 0, 1


def test_max_token_length(tok):
    """Hicbir ogrenmis token 28 ByteLevel karakterden uzun olmamali."""
    vocab = tok.get_vocab()
    # special + byte token'lar haric, sadece merge'lerden ogrenilenler
    special_set = set(SPECIAL_TOKENS) | set(ByteLevel.alphabet())
    violations = []
    for token in vocab:
        if token in special_set:
            continue
        if len(token) > MAX_TOKEN_LEN:
            violations.append((token, len(token)))

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
    print("SMOKE TEST — Tokenizer Saglamlik Testi")
    print("=" * 60)

    # 1. Ornekleme
    print(f"\n[1/3] Ornekleme ({args.lines:,} satir hedef)...")
    t0 = time.time()
    lines = sample_from_train(train_dir, args.lines)
    print(f"  {time.time() - t0:.1f} sn")

    # 2. Egitim
    print(f"\n[2/3] Tokenizer egitiliyor (vocab={VOCAB_SIZE:,})...")
    t0 = time.time()
    tok = build_tokenizer()
    trainer = build_trainer()
    tok.train_from_iterator(lines, trainer=trainer)
    elapsed = time.time() - t0
    print(f"  {elapsed:.1f} sn, vocab={tok.get_vocab_size():,}")

    # 3. Testler
    print(f"\n[3/3] Testler calistiriliyor...\n")

    tests = [
        ("Roundtrip (decode∘encode == NFKC)",     test_roundtrip),
        ("256 byte alfabe tamligi",                test_byte_alphabet),
        ("Special token ID'leri",                  test_special_ids),
        ("Turkce buyuk/kucuk harf korumasi",       test_turkish_casing),
        ("<unk> uretilmeme garantisi",             test_no_unk),
        ("PreTrainedTokenizerFast yukleme",        test_hf_load),
        ("max_token_length <= 28",                 test_max_token_length),
    ]

    total_pass = 0
    total_fail = 0
    results = []

    for name, fn in tests:
        print(f"  [{name}]")
        p, f = fn(tok)
        total_pass += p
        total_fail += f
        status = "✓ GECTI" if f == 0 else f"✗ {f} BASARISIZ"
        results.append((name, p, f, status))
        print(f"    → {status}\n")

    # Ozet
    print("=" * 60)
    if total_fail == 0:
        print(f"SONUC: TUMU GECTI ({total_pass} kontrol)")
        print("128K egitimine gecilebilir.")
    else:
        print(f"SONUC: {total_fail} BASARISIZLIK ({total_pass} gecti)")
        print("Hatalar duzeltilmeden 128K egitimi YAPILMAZ.")
    print("=" * 60)

    # Tokenizer'i diske kaydet — demo_tokenizer.py buradan yukler
    tok_path = data_dir / "smoke_tokenizer.json"
    tok.save(str(tok_path))
    print(f"\nTokenizer kaydedildi: {tok_path}")

    # Rapor dosyasi
    report_path = data_dir / "smoke_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("# Smoke Test Raporu\n\n")
        fh.write(f"Orneklem: {len(lines):,} satir | Vocab: {tok.get_vocab_size():,}\n\n")
        fh.write("| Test | Gecen | Basarisiz | Sonuc |\n")
        fh.write("|---|---:|---:|---|\n")
        for name, p, f, status in results:
            fh.write(f"| {name} | {p} | {f} | {status} |\n")
        fh.write(f"\n**Genel: {'GECTI' if total_fail == 0 else 'BASARISIZ'}**\n")
    print(f"Rapor: {report_path}")

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
