"""
Tokenizer Demo — Encode / Decode Gorunumu
==========================================
Smoke test'te egitilen tokenizer'i yukler, dataset DISINDAKI metinlerle
encode/decode yapar. Tokenizer'in metni nasil parcaladigini GORSEL olarak gosterir.

Onkosul: once smoke_test_tokenizer.py calistirilmis ve
         data/smoke_tokenizer.json olusmus olmali.

Girdi : data/smoke_tokenizer.json (smoke test'in kaydettigi tokenizer)
Cikti : konsol (gorsel token dagilimi)

Kullanim:
    python scripts/demo_tokenizer.py --data /content/drive/MyDrive/tr-tokenizer-data
"""

import argparse
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer


# ── Dataset'te OLMAYAN test metinleri ────────────────────────────────────────
# Farkli register'lardan, tokenizer'in genis yelpazede nasil davrandigini gostermek icin.

TEST_TEXTS = [
    ("Guncel haber",
     "Cumhurbaşkanı Yardımcısı Cevdet Yılmaz, bugün KKTC'ye giderek kayıp "
     "vatandaşların aileleriyle görüşecek. Manisa'nın Yunusemre ilçesinde "
     "3.7 büyüklüğünde bir deprem meydana geldi."),

    ("Forum / argo",
     "ya bu fiyata bu telefon alinir mi abi sacmalama bence bekle black friday'e "
     "bi dusun hele bi kac kisi ayni seyi soyledi sana"),

    ("Teknik / akademik",
     "Transformer mimarisinde çok başlı dikkat mekanizması, giriş dizisindeki "
     "her token'ın diğer token'larla olan ilişkisini paralel olarak hesaplar. "
     "Bu yaklaşım, özyinelemeli sinir ağlarına kıyasla eğitim süresini "
     "önemli ölçüde kısaltmıştır."),

    ("Emoji / sosyal medya",
     "bugün hava müthiş güzel ☀️🌊 sahilde yürüyüş yaptık 🏖️ "
     "akşam da mangal var 🔥🥩 hayat güzel 💯"),

    ("Diakritiksiz yazim",
     "dusunsene adam geldi is yerinde kavga cikardi mudur bile sasirdi "
     "herseyin bir aciklamasi var ama bu kadari da fazla"),

    ("Ingilizce karisik",
     "meeting'den çıktım, deadline yarın ama feature request'ler bitmiyor. "
     "Backend'deki bug'ı fix'ledik ama deployment'ta sorun çıktı yine."),
]


# ── Gosterim ─────────────────────────────────────────────────────────────────

def show_tokens(tok, text, label):
    """Bir metni encode edip token'lari gorsel olarak gosterir."""
    encoded = tok.encode(text)
    decoded = tok.decode(encoded.ids)
    expected = unicodedata.normalize("NFKC", text)
    roundtrip_ok = decoded == expected

    word_count = len(text.split())
    token_count = len(encoded.ids)
    fertility = token_count / max(word_count, 1)

    print(f"{'─' * 60}")
    print(f"  [{label}]")
    print(f"  Metin: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"  Token sayisi: {token_count}  |  Kelime sayisi: {word_count}  |  "
          f"Fertility: {fertility:.2f}")
    print(f"  Roundtrip: {'✓' if roundtrip_ok else '✗ BASARISIZ'}")
    print()

    # Token'lari INSAN-OKUNABILIR goster.
    # Sorun: emoji gibi cok byte'li karakterler kucuk vocab'da ayri token'lara
    # bolunuyor. Tek byte'i decode edince gecerli UTF-8 olusmuyor → "�" cikiyor.
    # Cozum: ardisik "bozuk" token'lari gruplayip birlikte decode et.
    display = []
    ids = encoded.ids
    i = 0
    while i < len(ids):
        piece = tok.decode([ids[i]])
        if "�" not in piece:
            # Temiz token — dogrudan ekle
            if piece.startswith(" "):
                piece = "_" + piece[1:]
            display.append(piece)
            i += 1
        else:
            # Kismi byte — ardisik bozuklari grupla
            j = i + 1
            while j < len(ids):
                group_decoded = tok.decode(list(ids[i:j + 1]))
                if "�" not in group_decoded:
                    break
                j += 1
            group_decoded = tok.decode(list(ids[i:j + 1]))
            if "�" not in group_decoded:
                n_tokens = j + 1 - i
                if group_decoded.startswith(" "):
                    group_decoded = "_" + group_decoded[1:]
                display.append(f"{group_decoded}({n_tokens}B)")
                i = j + 1
            else:
                # Hala decode edilemiyor — hex olarak goster
                display.append(f"[0x{ids[i]:02X}]")
                i += 1

    # Satirda 80 karakter siniriyla token'lari goster
    line = "  "
    for d in display:
        candidate = line + "│" + d if len(line) > 2 else line + d
        if len(candidate) > 80:
            print(line)
            line = "  " + d
        else:
            line = candidate
    if line.strip():
        print(line)

    print()


# ── Ana akis ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tokenizer demo")
    parser.add_argument("--data", type=str, default="data",
                        help="Veri dizini (icerisinde smoke_tokenizer.json olmali)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    tok_path = data_dir / "smoke_tokenizer.json"

    if not tok_path.exists():
        raise SystemExit(
            f"Tokenizer bulunamadi: {tok_path}\n"
            f"Once smoke_test_tokenizer.py calistirilmali."
        )

    tok = Tokenizer.from_file(str(tok_path))
    print(f"Tokenizer yuklendi: {tok_path}  (vocab: {tok.get_vocab_size():,})\n")

    print("=" * 60)
    print("  TOKENIZER DEMO — Encode / Decode")
    print(f"  Vocab: {tok.get_vocab_size():,} (smoke test boyutu)")
    print("=" * 60)

    for label, text in TEST_TEXTS:
        show_tokens(tok, text, label)

    print("─" * 60)
    print("Not: Bu 2,000 vocab'lik bir tokenizer. 128K vocab'da fertility")
    print("cok daha iyi olacak (daha az token, daha uzun alt-kelimeler).")
    print("─" * 60)


if __name__ == "__main__":
    main()
