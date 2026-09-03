"""
Korpus Dogrulama
================
Toplanan korpusu okur, hicbir seyi degistirmez. Amaci "veri gercekten
istedigimiz gibi mi" sorusunu olcumle cevaplamak.

Bazi kontroller HER satirda, bazilari ORNEKLEM uzerinde yapiliyor:
21.8M satirda satir basina 275 karakter var; her karakteri Python
dongusunde gezmek saatler surer. Ucuz olanlar (uzunluk, hash, bozuk byte)
tam taranir; pahali olanlar (Turkce karakter orani, supheli kalip) her
SAMPLE_EVERY satirda bir orneklenir -- 400 binden fazla ornek, istatistik
icin fazlasiyla yeterli.

Girdi : data/shards/*.txt + data/state.json
Cikti : data/verify_report.md  (+ konsol ozeti)

Kullanim:
    python scripts/verify_corpus.py
    python scripts/verify_corpus.py --data /content/drive/MyDrive/tr-tokenizer-data
"""

import argparse
import json
import random
import time
from collections import defaultdict
from hashlib import blake2b
from pathlib import Path

SAMPLE_EVERY = 50        # pahali kontroller icin: her 50. satir
N_SAMPLES = 5            # kaynak basina gosterilecek ornek satir
SEED = 42

TR_CHARS = "çğıöşüÇĞİÖŞÜ"
LENGTH_BUCKETS = [(0, 19), (20, 49), (50, 99), (100, 199), (200, 499),
                  (500, 999), (1000, 1999), (2000, 10 ** 9)]


def pct(part, whole, digits=1):
    """Orneklem bossa yaniltici 0.0% yerine '-' dondurur."""
    return "-" if not whole else f"{100 * part / whole:.{digits}f}%"


def bucket_label(lo, hi):
    return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"


def dedup_key(s):
    """collect_corpus.py'deki ile AYNI anahtar -- tekrar taramasi ayni olcute gore."""
    norm = " ".join("".join(c if c.isalnum() else " " for c in s.lower()).split())
    return int.from_bytes(blake2b(norm.encode("utf-8"), digest_size=8).digest(), "big")


class Stats:
    """Tek bir kaynagin sayaclari."""

    def __init__(self):
        self.lines = 0
        self.chars = 0
        self.bad_bytes = 0          # U+FFFD: bozuk UTF-8 isareti
        self.buckets = defaultdict(int)
        self.shards = 0
        # orneklem uzerinden
        self.sampled = 0
        self.tr_chars = 0
        self.sampled_chars = 0
        self.allcaps = 0
        self.has_url = 0
        self.low_variety = 0        # ayni kelimenin tekrarlandigi satir
        self.samples = []           # reservoir sampling ile N_SAMPLES satir


def add_sample(st, line, rng):
    """Reservoir sampling: butun satirlari bellekte tutmadan rastgele N tane."""
    if len(st.samples) < N_SAMPLES:
        st.samples.append(line)
    else:
        j = rng.randrange(st.sampled)
        if j < N_SAMPLES:
            st.samples[j] = line


def scan_line_sampled(st, line, rng):
    """Pahali kontroller. Sadece her SAMPLE_EVERY satirda bir cagriliyor."""
    st.sampled += 1
    st.sampled_chars += len(line)
    # str.count C seviyesinde tarar; karakter karakter Python dongusunden ~100x hizli
    st.tr_chars += sum(line.count(c) for c in TR_CHARS)

    if len(line) > 40 and line.isupper():
        st.allcaps += 1
    if "http" in line:
        st.has_url += 1
    words = line.split()
    if len(words) >= 8 and len(set(words)) < len(words) * 0.3:
        st.low_variety += 1

    add_sample(st, line, rng)


def scan(shard_dir, check_duplicates=True):
    """Butun shard'lari okur. Kaynak adi dosya adinin ilk parcasindan gelir."""
    stats = defaultdict(Stats)
    seen = set() if check_duplicates else None
    duplicates = 0
    rng = random.Random(SEED)

    shards = sorted(shard_dir.glob("*.txt"))
    if not shards:
        raise SystemExit(f"Shard bulunamadi: {shard_dir}")

    total_lines = 0
    t0 = t_log = time.time()
    print(f"{len(shards)} shard taraniyor...\n")

    for path in shards:
        source = path.stem.rsplit("_", 1)[0]      # forum_0003.txt -> forum
        st = stats[source]
        st.shards += 1

        # errors="replace": bozuk byte varsa okuma patlamasin, U+FFFD olarak sayilsin
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                n = len(line)
                st.lines += 1
                st.chars += n
                total_lines += 1

                if "�" in line:
                    st.bad_bytes += 1

                for lo, hi in LENGTH_BUCKETS:
                    if lo <= n <= hi:
                        st.buckets[bucket_label(lo, hi)] += 1
                        break

                if seen is not None:
                    k = dedup_key(line)
                    if k in seen:
                        duplicates += 1
                    else:
                        seen.add(k)

                if total_lines % SAMPLE_EVERY == 0:
                    scan_line_sampled(st, line, rng)

                if time.time() - t_log >= 30:
                    t_log = time.time()
                    rate = total_lines / (time.time() - t0)
                    print(f"  {total_lines:>12,} satir  ({rate:,.0f}/sn)  "
                          f"son dosya: {path.name}")

    print(f"\nTarama bitti: {total_lines:,} satir, {(time.time() - t0) / 60:.1f} dk")
    return stats, duplicates, len(seen) if seen is not None else 0


def compare_state(stats, state_file, out):
    """state.json ile fiili sayimlari karsilastirir: eksik yazilmis shard var mi."""
    if not state_file.exists():
        out.append("\n> `state.json` bulunamadi, karsilastirma atlandi.\n")
        return

    state = json.loads(state_file.read_text(encoding="utf-8"))
    out.append("\n## state.json karsilastirmasi\n")
    out.append("| kaynak | state satir | fiili satir | state karakter | fiili karakter | durum |")
    out.append("|---|---:|---:|---:|---:|:--:|")

    for name in sorted(set(state) | set(stats)):
        s = state.get(name, {})
        f = stats.get(name)
        s_lines, s_chars = s.get("lines", 0), s.get("chars", 0)
        f_lines = f.lines if f else 0
        f_chars = f.chars if f else 0
        ok = "OK" if (s_lines == f_lines and s_chars == f_chars) else "FARK"
        out.append(f"| {name} | {s_lines:,} | {f_lines:,} | "
                   f"{s_chars:,} | {f_chars:,} | {ok} |")


def build_report(stats, duplicates, unique_keys, state_file):
    out = ["# Korpus Dogrulama Raporu", ""]

    total_lines = sum(s.lines for s in stats.values())
    total_chars = sum(s.chars for s in stats.values())
    total_bad = sum(s.bad_bytes for s in stats.values())

    out.append("## Genel")
    out.append("")
    out.append(f"- Toplam satir: **{total_lines:,}**")
    out.append(f"- Toplam karakter: **{total_chars:,}** ({total_chars / 1e9:.2f} GB)")
    out.append(f"- Ortalama satir uzunlugu: **{total_chars / max(total_lines, 1):.0f}** karakter")
    out.append(f"- Bozuk UTF-8 iceren satir: **{total_bad:,}**"
               + ("  ← TEMIZ" if total_bad == 0 else "  ← INCELENMELI"))
    out.append(f"- Tekrar eden satir: **{duplicates:,}**"
               + ("  ← dedup dogrulandi" if duplicates == 0 else "  ← DEDUP KACIRMIS"))
    out.append(f"- Essiz anahtar: **{unique_keys:,}**")

    out.append("\n## Kaynak dagilimi\n")
    out.append("| kaynak | shard | satir | karakter | pay % | ort. uzunluk | TR karakter % |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1].chars):
        share = 100 * s.chars / total_chars if total_chars else 0
        avg = s.chars / max(s.lines, 1)
        out.append(f"| {name} | {s.shards} | {s.lines:,} | {s.chars:,} | "
                   f"{share:.1f} | {avg:.0f} | {pct(s.tr_chars, s.sampled_chars)} |")

    out.append("\n## Satir uzunlugu dagilimi\n")
    labels = [bucket_label(lo, hi) for lo, hi in LENGTH_BUCKETS]
    out.append("| kaynak | " + " | ".join(labels) + " |")
    out.append("|---" * (len(labels) + 1) + "|")
    for name, s in sorted(stats.items()):
        row = [f"{100 * s.buckets.get(l, 0) / max(s.lines, 1):.1f}%" for l in labels]
        out.append(f"| {name} | " + " | ".join(row) + " |")
    out.append("")
    out.append("> `0-19` sifir olmali (MIN_LINE_CHARS=20). `2000+` sutunu "
               "`split_long`'un tavana dayadigi parcalari gosterir.")

    out.append("\n## Supheli kalip orani (orneklem)\n")
    out.append("| kaynak | ornek | tamami buyuk harf | URL iceren | kelime tekrari |")
    out.append("|---|---:|---:|---:|---:|")
    for name, s in sorted(stats.items()):
        out.append(f"| {name} | {s.sampled:,} | {pct(s.allcaps, s.sampled, 2)} | "
                   f"{pct(s.has_url, s.sampled, 2)} | {pct(s.low_variety, s.sampled, 2)} |")

    compare_state(stats, state_file, out)

    out.append("\n## Rastgele ornekler\n")
    for name, s in sorted(stats.items()):
        out.append(f"\n### {name}\n")
        for line in s.samples:
            out.append(f"> {line[:400]}\n")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="korpus klasoru")
    ap.add_argument("--no-dup-check", action="store_true",
                    help="tekrar taramasini atla (~1.6 GB RAM tasarrufu)")
    args = ap.parse_args()

    data_dir = Path(args.data)
    stats, duplicates, unique_keys = scan(data_dir / "shards",
                                          check_duplicates=not args.no_dup_check)

    report = build_report(stats, duplicates, unique_keys, data_dir / "state.json")
    report_file = data_dir / "verify_report.md"
    report_file.write_text(report, encoding="utf-8")

    # Konsola kisa ozet; ayrintili hali raporda
    total_lines = sum(s.lines for s in stats.values())
    total_chars = sum(s.chars for s in stats.values())
    total_bad = sum(s.bad_bytes for s in stats.values())
    print("\n" + "=" * 70)
    print(f"  satir            : {total_lines:,}")
    print(f"  karakter         : {total_chars:,}  ({total_chars / 1e9:.2f} GB)")
    print(f"  bozuk UTF-8      : {total_bad:,}")
    print(f"  tekrar eden satir: {duplicates:,}")
    print(f"  kaynak sayisi    : {len(stats)}")
    print("=" * 70)
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1].chars):
        share = 100 * s.chars / total_chars if total_chars else 0
        print(f"  {name:9} {s.chars / 1e6:9,.1f} MB  %{share:5.1f}  "
              f"{s.lines:>11,} satir  TR {pct(s.tr_chars, s.sampled_chars)}")
    print("=" * 70)
    print(f"\nAyrintili rapor: {report_file}")


if __name__ == "__main__":
    main()
