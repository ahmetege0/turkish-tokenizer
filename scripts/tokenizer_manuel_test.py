"""
Tokenizer Manuel Test
=====================
Egitilen 128K tokenizer'i ELLE, gozle dogrulamak icin. Otomatik smoke test'ten
farki: burada amac "gecti/kaldi" degil, tokenizer'in gercekten Turkce ogrenip
ogrenmedigini INSAN GOZUYLE gormek.

6 test var, her biri farkli bir soruya cevap veriyor:
  1. Dosya gercek mi        -> boyut ve vocab sayisi tutuyor mu
  2. Encode/decode          -> bir cumle dogru parcalanip geri geliyor mu
  3. Morfoloji              -> Turkce ekleri (ler/imiz/den) taniyor mu
  4. Rakip karsilastirmasi  -> BERTurk ve XLM-R'den daha mi verimli  [EN ONEMLI]
  5. Vocab temizligi        -> icinde ne kadar URL/cop var
  6. Gerceklik kanitri      -> korpusa ozgu kelimeleri biliyor mu

Girdi : tokenizer.json (Drive'dan indirilmis olmali)
Cikti : sadece konsol

Kullanim:
    python scripts/tokenizer_manuel_test.py --tokenizer C:/yol/tokenizer.json
    python scripts/tokenizer_manuel_test.py --tokenizer ... --no-baseline
"""

import argparse
import sys
from pathlib import Path

from tokenizers import Tokenizer

# Windows konsolu varsayilan olarak cp1254 kullanir ve Turkce/▁ karakterlerinde
# patlar. Ciktiyi UTF-8'e sabitliyoruz ki script her yerde ayni calissin.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def baslik(no, ad, soru):
    print(f"\n{'=' * 70}")
    print(f"  {no}. {ad}")
    print(f"     Soru: {soru}")
    print("=" * 70)


# ── 1. Dosya gercek mi ───────────────────────────────────────────────────────

def test_dosya(tok, path):
    baslik(1, "DOSYA KONTROLU", "dosya gercek boyutta mi, vocab dolu mu?")
    mb = path.stat().st_size / 1e6
    vocab = tok.get_vocab_size()
    print(f"  dosya boyutu : {mb:.1f} MB")
    print(f"  vocab boyutu : {vocab:,}")
    print(f"  <s>   -> ID {tok.token_to_id('<s>')}")
    print(f"  <pad> -> ID {tok.token_to_id('<pad>')}")
    print(f"  </s>  -> ID {tok.token_to_id('</s>')}")
    print()
    print(f"  {'OK' if 3 < mb < 40 else 'SUPHELI'}: 128K vocab icin 5-15 MB bekleniyor")
    print(f"  {'OK' if vocab == 128_000 else 'SUPHELI'}: vocab tam 128.000 olmali")


# ── 2. Encode / decode ───────────────────────────────────────────────────────

def test_encode_decode(tok):
    baslik(2, "ENCODE / DECODE", "cumle dogru parcalanip aynen geri geliyor mu?")
    cumleler = [
        "Yarın İstanbul'a gidip arkadaşlarımızla buluşacağız.",
        "ya kanka bu fiyata alinir mi sence, bekleyelim mi?",
        "Çalışmanın sonuçları önemli bulgular içermektedir.",
    ]
    for s in cumleler:
        enc = tok.encode(s)
        geri = tok.decode(enc.ids)
        print(f"\n  metin  : {s}")
        print(f"  token  : {enc.tokens}")
        print(f"  sayi   : {len(enc.tokens)} token / {len(s.split())} kelime "
              f"= {len(enc.tokens) / len(s.split()):.2f} fertility")
        print(f"  geri   : {geri}")
        print(f"  {'OK' if geri == s else 'FARKLI'}: geri cevirme aynen esiyor mu")


# ── 3. Morfoloji ─────────────────────────────────────────────────────────────

def test_morfoloji(tok):
    baslik(3, "MORFOLOJI", "Turkce ekleri (ler/imiz/den) taniyor mu?")
    print("  Saglam bir tokenizer bu kelimeleri EK SINIRLARINDAN bolmeli,")
    print("  harf harf degil.\n")
    for w in ["evlerimizden", "kitaplarınızdan", "arkadaşlarımla",
              "gelemeyeceğimizi", "çalıştırabileceğimiz", "bilgisayarlaştırılmış"]:
        t = tok.encode(w).tokens
        print(f"  {w:24} → {len(t)} token: {t}")


# ── 4. Rakip karsilastirmasi (EN ONEMLI) ─────────────────────────────────────

def test_karsilastirma(tok):
    baslik(4, "RAKIP KARSILASTIRMASI", "BERTurk ve XLM-R'den daha verimli mi?")
    print("  Fertility = kelime basina token. DUSUK olan daha iyi:")
    print("  ayni metni daha az token'a sigdiriyor demek.\n")

    try:
        from transformers import AutoTokenizer
        print("  BERTurk ve XLM-R indiriliyor (ilk seferde birkac MB)...")
        berturk = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        xlmr = AutoTokenizer.from_pretrained("xlm-roberta-base")
    except Exception as e:
        print(f"  ATLANDI: rakip tokenizer'lar yuklenemedi ({type(e).__name__})")
        print("  Internet yoksa --no-baseline ile bu testi kapatabilirsin.")
        return

    metinler = [
        ("haber",      "Cumhurbaşkanı yarın Ankara'da önemli açıklamalarda bulunacak."),
        ("forum/argo", "ya kanka bu fiyata alinir mi sence bekleyelim mi bilmiyorum"),
        ("urun yorum", "Bu ürünü beğenmedim, kargo çok geç geldi ve iade edeceğim."),
        ("akademik",   "Çalışmanın sonuçları, değerlendirilmesi gereken bulgular içermektedir."),
        ("diakritiksiz", "dusunduk tasindik sonunda karar verdik guzel oldu herseyimiz"),
    ]

    print(f"\n  {'register':<14} {'BIZIM':>8} {'BERTurk':>9} {'XLM-R':>8}   kazanc")
    print("  " + "-" * 58)
    toplam = [0, 0, 0]
    for ad, m in metinler:
        kelime = len(m.split())
        a = len(tok.encode(m).ids)
        b = len(berturk.encode(m, add_special_tokens=False))
        c = len(xlmr.encode(m, add_special_tokens=False))
        toplam[0] += a; toplam[1] += b; toplam[2] += c
        # en iyi rakibe gore yuzde kazanc
        en_iyi_rakip = min(b, c)
        kazanc = 100 * (en_iyi_rakip - a) / en_iyi_rakip
        print(f"  {ad:<14} {a/kelime:>8.2f} {b/kelime:>9.2f} {c/kelime:>8.2f}   "
              f"{kazanc:+.0f}%")

    print("  " + "-" * 58)
    print(f"  {'TOPLAM token':<14} {toplam[0]:>8} {toplam[1]:>9} {toplam[2]:>8}")
    print(f"\n  Bizimki en az token uretiyorsa Turkce icin daha verimli demektir.")
    print(f"  XLM-R'nin 250K vocab'i 100 dile bolunuyor; bizim 128K'nin")
    print(f"  tamami Turkce'ye ayrildi -- kazanc oradan geliyor.")


# ── 5. Vocab temizligi ───────────────────────────────────────────────────────

def test_vocab_temizligi(tok):
    baslik(5, "VOCAB TEMIZLIGI", "icinde ne kadar URL/cop token var?")
    vocab = tok.get_vocab()
    url_ish = [t for t in vocab if any(x in t for x in ("http", ".com", ".aspx", "?", "="))]
    turkce = [t for t in vocab if any(c in t for c in "çğıöşüÇĞİÖŞÜ")]

    print(f"  toplam token       : {len(vocab):,}")
    print(f"  URL/sorgu iceren   : {len(url_ish):,}  (%{100*len(url_ish)/len(vocab):.2f})")
    print(f"  Turkce harf iceren : {len(turkce):,}  (%{100*len(turkce)/len(vocab):.1f})")

    print("\n  ID araliklarina gore ornekler (BPE ogrenme sirasi):")
    ters = {v: k for k, v in vocab.items()}
    for lo, ad in [(0, "special token"), (100, "taban karakterler"),
                   (5_000, "erken merge'ler"), (50_000, "orta merge'ler"),
                   (120_000, "gec merge'ler")]:
        ornek = [ters[i] for i in range(lo, min(lo + 8, len(vocab))) if i in ters]
        print(f"    ID {lo:>6}+ ({ad:<18}): {ornek}")


# ── 6. Gerceklik kaniti ──────────────────────────────────────────────────────

def test_gerceklik(tok):
    baslik(6, "GERCEKLIK KANITI", "korpustan gercekten ogrenmis mi?")
    print("  Bu kelimeler hicbir varsayilan/hazir listede yok -- sadece bizim")
    print("  Turkce korpusumuzdan ogrenilmis olabilir. Az token = ogrenmis.\n")
    for w in ["Cumhurbaşkanı", "belediyenin", "kargonun", "yorumlarınızı",
              "teşekkürler", "merhaba", "arkadaşım"]:
        t = tok.encode(w).tokens
        durum = "OGRENMIS" if len(t) <= 3 else "parcali"
        print(f"  {w:16} → {len(t)} token {str(t):<45} {durum}")


# ── Ana akis ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Tokenizer manuel test")
    ap.add_argument("--tokenizer", default="data/tokenizer.json",
                    help="tokenizer.json yolu (Drive'dan indirilmis)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="BERTurk/XLM-R karsilastirmasini atla (internet gerekmez)")
    args = ap.parse_args()

    path = Path(args.tokenizer)
    if not path.exists():
        raise SystemExit(
            f"Tokenizer bulunamadi: {path}\n"
            f"Once Drive'dan tokenizer.json'i indirip yolunu --tokenizer ile ver."
        )

    tok = Tokenizer.from_file(str(path))

    print("\n" + "#" * 70)
    print("#  TOKENIZER MANUEL TEST")
    print(f"#  {path}")
    print("#" * 70)

    test_dosya(tok, path)
    test_encode_decode(tok)
    test_morfoloji(tok)
    if not args.no_baseline:
        test_karsilastirma(tok)
    test_vocab_temizligi(tok)
    test_gerceklik(tok)

    print(f"\n{'=' * 70}")
    print("  Testler bitti. En onemlisi 4. test (rakip karsilastirmasi):")
    print("  bizimki daha az token uretiyorsa is gormus demektir.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
