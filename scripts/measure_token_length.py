"""
Token Uzunlugu Olcumu
=====================
max_token_length esigini tahminle degil OLCUMLE belirlemek icin.

Neden "en uzun kelime" dogru olcut degil: 6 GB web metninde maksimum degeri
her zaman aykiri degerler belirler -- kesilmemis URL parcalari, birbirine
yapismis anahtar kelime yiginlari, uzatilmis "aaaaaa"lar. Aradigimiz sey
mesru Turkce kelimelerin dagilimi, yani yuksek bir YUZDELIK dilim.

Iki ayri uzunluk olculuyor, cunku bunlari karistirmak buyuk hata olur:

  karakter uzunlugu  -> insanin okudugu ("degerlendirmelerimizden" = 23)
  ByteLevel uzunlugu -> BPE'nin GORDUGU. ByteLevel her BYTE'i bir karaktere
                        esliyor; Turkce harfler UTF-8'de 2 byte oldugu icin
                        ayni kelime burada daha UZUN gorunur.

max_token_length ikinci uzayla karsilastirilir. Karakter uzunluguna bakip
esik koyarsak Turkce kelimeleri farkinda olmadan keseriz.

Olcum, BPE trainer'in gercekte gordugu parcalar uzerinde yapiliyor: metin
ayni ByteLevel pre-tokenizer'dan geciriliyor. Bir token asla parca sinirini
asamaz, dolayisiyla dogru olcut bu parcalarin uzunlugu.

Girdi : data/train/*.txt
Cikti : data/token_length_report.md  (+ konsol ozeti)

Kullanim:
    python scripts/measure_token_length.py --data /content/drive/MyDrive/tr-tokenizer-data
"""

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel

SAMPLE_PER_SOURCE = 200_000     # kaynak basina orneklenecek satir
LONG_THRESHOLD = 32             # bundan uzun parcalar ornek olarak saklanir
MAX_LONG_KEPT = 20_000          # bellek siniri
THRESHOLDS = [12, 16, 20, 24, 28, 32, 40]   # degerlendirilecek esikler
PERCENTILES = [50, 90, 99, 99.9, 99.99]

pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
decoder = ByteLevelDecoder()


def percentile(counter, p):
    """Tam sayi histogramindan yuzdelik. Siralamaya gerek yok, birikimli topla."""
    total = sum(counter.values())
    if not total:
        return 0
    hedef = total * p / 100
    birikim = 0
    for uzunluk in sorted(counter):
        birikim += counter[uzunluk]
        if birikim >= hedef:
            return uzunluk
    return max(counter)


def sample_lines(shards, budget):
    """Kaynagin shard'larina esit dagitarak satir orneginin bir kismini okur."""
    if not shards:
        return
    per_shard = max(1, budget // len(shards))
    for path in shards:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= per_shard:
                    break
                line = line.rstrip("\n")
                if line:
                    yield line


def measure(train_dir):
    by_source = defaultdict(lambda: {"bl": Counter(), "ch": Counter(), "lines": 0})
    long_pieces = Counter()

    shards_by_source = defaultdict(list)
    for path in sorted(train_dir.glob("*.txt")):
        shards_by_source[path.stem.rsplit("_", 1)[0]].append(path)

    t0 = time.time()
    for source, shards in sorted(shards_by_source.items()):
        st = by_source[source]
        for line in sample_lines(shards, SAMPLE_PER_SOURCE):
            st["lines"] += 1
            for piece, (start, end) in pre_tokenizer.pre_tokenize_str(line):
                st["bl"][len(piece)] += 1        # BPE'nin gordugu uzunluk
                st["ch"][end - start] += 1       # insanin gordugu uzunluk
                if len(piece) > LONG_THRESHOLD and len(long_pieces) < MAX_LONG_KEPT:
                    long_pieces[piece] += 1
        print(f"  {source:10} {st['lines']:>8,} satir, "
              f"{sum(st['bl'].values()):>10,} parca")

    print(f"\nOlcum bitti: {time.time() - t0:.0f} sn")
    return by_source, long_pieces


def build_report(by_source, long_pieces):
    bl_all, ch_all = Counter(), Counter()
    for st in by_source.values():
        bl_all.update(st["bl"])
        ch_all.update(st["ch"])
    toplam = sum(bl_all.values())

    out = ["# Token Uzunlugu Olcumu", "",
           f"Ornek: **{sum(st['lines'] for st in by_source.values()):,}** satir, "
           f"**{toplam:,}** parca.", "",
           "İki uzunluk ayri olculuyor. `max_token_length` **ByteLevel** uzunluguyla "
           "karsilastirilir; karakter uzunluguna bakip esik koymak hata olur.", ""]

    out.append("## Yuzdelikler\n")
    out.append("| kaynak | " + " | ".join(f"p{p} BL" for p in PERCENTILES)
               + " | max BL | " + " | ".join(f"p{p} kar" for p in PERCENTILES) + " | max kar |")
    out.append("|---" * (2 * len(PERCENTILES) + 3) + "|")
    for name, st in sorted(by_source.items()):
        bl = [str(percentile(st["bl"], p)) for p in PERCENTILES] + [str(max(st["bl"]))]
        ch = [str(percentile(st["ch"], p)) for p in PERCENTILES] + [str(max(st["ch"]))]
        out.append(f"| {name} | " + " | ".join(bl + ch) + " |")
    bl = [str(percentile(bl_all, p)) for p in PERCENTILES] + [str(max(bl_all))]
    ch = [str(percentile(ch_all, p)) for p in PERCENTILES] + [str(max(ch_all))]
    out.append(f"| **TUMU** | " + " | ".join(bl + ch) + " |")

    out.append("\n## Esik secenekleri\n")
    out.append("| esik (ByteLevel) | kapsanan parca | kesilen parca | kesilen sayi |")
    out.append("|---:|---:|---:|---:|")
    for t in THRESHOLDS:
        kapsanan = sum(c for u, c in bl_all.items() if u <= t)
        out.append(f"| {t} | %{100 * kapsanan / toplam:.3f} | "
                   f"%{100 * (toplam - kapsanan) / toplam:.3f} | {toplam - kapsanan:,} |")
    out.append("")
    out.append("> \"Kesilen\" demek o parcanin atilmasi DEGIL: BPE o uzunlukta tek token "
               "uretemez, kelime birden fazla token'a bolunur. Bedeli fertility artisi.")

    out.append("\n## En uzun parcalar (cop mu, mesru kelime mi?)\n")
    out.append("| ByteLevel uz. | karakter | parca |")
    out.append("|---:|---:|---|")
    for piece, _ in sorted(long_pieces.items(), key=lambda kv: -len(kv[0]))[:30]:
        okunabilir = decoder.decode([piece]).strip()
        out.append(f"| {len(piece)} | {len(okunabilir)} | `{okunabilir[:90]}` |")

    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="korpus klasoru")
    args = ap.parse_args()

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise SystemExit(f"{train_dir} yok. Once split_holdout.py calistirilmali.")

    print(f"Ornekleniyor (kaynak basina {SAMPLE_PER_SOURCE:,} satir)...\n")
    by_source, long_pieces = measure(train_dir)

    report_file = data_dir / "token_length_report.md"
    report_file.write_text(build_report(by_source, long_pieces), encoding="utf-8")

    bl_all = Counter()
    for st in by_source.values():
        bl_all.update(st["bl"])
    toplam = sum(bl_all.values())

    print("\n" + "=" * 66)
    print("ByteLevel uzunlugu (BPE'nin gordugu) — TUM KAYNAKLAR")
    for p in PERCENTILES:
        print(f"  p{p:<6} {percentile(bl_all, p):>4}")
    print(f"  max     {max(bl_all):>4}")
    print("=" * 66)
    print(f"{'esik':>6} {'kapsanan':>12} {'kesilen parca':>16}")
    for t in THRESHOLDS:
        kapsanan = sum(c for u, c in bl_all.items() if u <= t)
        print(f"{t:>6} {'%' + f'{100 * kapsanan / toplam:.3f}':>12} "
              f"{toplam - kapsanan:>16,}")
    print("=" * 66)
    print(f"\nAyrintili rapor: {report_file}")


if __name__ == "__main__":
    main()
