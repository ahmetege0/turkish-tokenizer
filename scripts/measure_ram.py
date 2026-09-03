"""
RAM Tavani Olcumu
=================
128K egitiminin ne kadar RAM gerektirdigini TAHMIN ETMEDEN OLCMEK icin.

Ayni train verisinin orantili bir alt kumesi uzerinde artan vocab boyutlariyla
(8K, 16K, 32K) tokenizer egitir, her birinde tepe RAM ve sureyi olcer.
3 noktadan 128K'ya ekstrapole eder.

Turkce eklemeli dil: benzersiz kelime tipi sayisi cok yuksek, BPE trainer
bunlarin hepsini frekansla bellekte tutuyor. 12.7 GB standart Colab RAM'i
yetmeyebilir.

Girdi : data/train/*.txt (orantili alt kume)
Cikti : konsol + data/ram_report.md

Kullanim:
    python scripts/measure_ram.py --data /content/drive/MyDrive/tr-tokenizer-data
"""

import argparse
import gc
import os
import time
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer, normalizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# ── Ayarlar ──────────────────────────────────────────────────────────────────

VOCAB_STEPS    = [8_000, 16_000, 32_000]
TARGET_VOCAB   = 128_000
MIN_FREQUENCY  = 2
MAX_TOKEN_LEN  = 28
SAMPLE_LINES   = 500_000       # olcum icin yeterli, tam korpus degil

SPECIAL_TOKENS = (
    ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    + [f"<reserved_{i}>" for i in range(27)]
)


# ── RAM olcumu ───────────────────────────────────────────────────────────────

def get_peak_ram_mb():
    """Linux'ta (Colab) surecin tepe RAM kullanimini MB olarak dondurur."""
    # resource.getrusage Linux'ta KB, macOS'ta byte dondurur
    import resource
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux: ru_maxrss KB cinsindendir
    return rusage.ru_maxrss / 1024


def get_current_ram_mb():
    """Anlik RAM kullanimini MB olarak dondurur."""
    # /proc/self/status'tan VmRSS okumak en guvenilir yol (Colab = Linux)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024   # kB → MB
    except FileNotFoundError:
        pass
    # Fallback: resource
    return get_peak_ram_mb()


# ── Ornekleme ────────────────────────────────────────────────────────────────

def sample_lines(train_dir, budget):
    """Kaynak bazinda orantili ornekleme (smoke test ile ayni mantik)."""
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
        count = 0
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if line:
                        lines.append(line)
                        count += 1
                        if count >= per_shard:
                            break
            if count >= source_budget:
                break

    print(f"  {len(shards_by_source)} kaynaktan {len(lines):,} satir orneklendi")
    return lines


# ── Egitim ───────────────────────────────────────────────────────────────────

def train_and_measure(lines, vocab_size):
    """Verilen vocab boyutuyla egitir, tepe RAM ve sure dondurur."""
    # Egitim oncesi RAM — baseline
    gc.collect()
    ram_before = get_current_ram_mb()

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        max_token_length=MAX_TOKEN_LEN,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=ByteLevel.alphabet(),
    )

    t0 = time.time()
    tok.train_from_iterator(lines, trainer=trainer)
    elapsed = time.time() - t0

    peak_ram = get_peak_ram_mb()
    ram_after = get_current_ram_mb()

    # Tokenizer nesnesini sil, bellek geri al
    del tok, trainer
    gc.collect()

    return {
        "vocab_size": vocab_size,
        "elapsed_sec": elapsed,
        "ram_before_mb": ram_before,
        "ram_after_mb": ram_after,
        "peak_ram_mb": peak_ram,
    }


# ── Ekstrapole ───────────────────────────────────────────────────────────────

def extrapolate(results, target_vocab):
    """
    Olcum noktalarindan 128K'ya basit log-lineer ekstrapole.
    BPE'de merge tablosu vocab ile dogru orantili buyur; kelime frekans
    tablosu sabit kalir. Bu yuzden RAM ~ a + b * vocab seklinde.
    """
    # En kucuk ve en buyuk olcumden lineer ekstrapole
    r1 = results[0]
    r2 = results[-1]
    v1, m1 = r1["vocab_size"], r1["peak_ram_mb"]
    v2, m2 = r2["vocab_size"], r2["peak_ram_mb"]

    if v2 != v1:
        slope = (m2 - m1) / (v2 - v1)
        predicted = m1 + slope * (target_vocab - v1)
    else:
        predicted = m2

    # Sure ekstrapolasyonu
    t1, t2 = r1["elapsed_sec"], r2["elapsed_sec"]
    if v2 != v1:
        time_slope = (t2 - t1) / (v2 - v1)
        predicted_time = t1 + time_slope * (target_vocab - v1)
    else:
        predicted_time = t2

    return predicted, predicted_time


# ── Ana akis ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAM tavani olcumu")
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
    print("RAM TAVANI OLCUMU")
    print("=" * 60)

    # Ornekleme (bir kez, hepsi ayni veri uzerinde)
    print(f"\nOrnekleme ({args.lines:,} satir hedef)...")
    lines = sample_lines(train_dir, args.lines)

    # Olcumler
    results = []
    for vocab in VOCAB_STEPS:
        print(f"\n{'─' * 60}")
        print(f"  Vocab: {vocab:,}")
        print(f"{'─' * 60}")
        r = train_and_measure(lines, vocab)
        results.append(r)
        print(f"  Sure     : {r['elapsed_sec']:.1f} sn")
        print(f"  RAM once : {r['ram_before_mb']:.0f} MB")
        print(f"  RAM sonra: {r['ram_after_mb']:.0f} MB")
        print(f"  Tepe RAM : {r['peak_ram_mb']:.0f} MB")

    # Ekstrapole
    predicted_ram, predicted_time = extrapolate(results, TARGET_VOCAB)

    print(f"\n{'=' * 60}")
    print(f"EKSTRAPOLE → {TARGET_VOCAB:,} vocab")
    print(f"{'=' * 60}")
    print(f"  Tahmini tepe RAM : {predicted_ram:,.0f} MB ({predicted_ram/1024:.1f} GB)")
    print(f"  Tahmini sure     : {predicted_time:,.0f} sn ({predicted_time/3600:.1f} saat)")
    print()

    # Karar
    STANDARD_RAM = 12_700    # MB, Colab standart
    HIGH_RAM = 51_000        # MB, Colab yuksek RAM
    safety = 1.3             # %30 guvenlik payi

    needed = predicted_ram * safety
    if needed < STANDARD_RAM:
        verdict = "STANDART runtime (12.7 GB) YETERLI"
    elif needed < HIGH_RAM:
        verdict = "YUKSEK RAM runtime (51 GB) GEREKLI"
    else:
        verdict = "⚠️ 51 GB bile yetmeyebilir — veriyi kucultmeyi dusun"

    print(f"  Guvenlik payiyla (%30): {needed:,.0f} MB ({needed/1024:.1f} GB)")
    print(f"  → {verdict}")

    # Rapor
    report_path = data_dir / "ram_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("# RAM Tavani Olcum Raporu\n\n")
        fh.write(f"Orneklem: {len(lines):,} satir\n\n")
        fh.write("| Vocab | Sure (sn) | Tepe RAM (MB) | Tepe RAM (GB) |\n")
        fh.write("|---:|---:|---:|---:|\n")
        for r in results:
            fh.write(f"| {r['vocab_size']:,} | {r['elapsed_sec']:.1f} "
                     f"| {r['peak_ram_mb']:,.0f} | {r['peak_ram_mb']/1024:.1f} |\n")
        fh.write(f"\n## Ekstrapole → {TARGET_VOCAB:,}\n\n")
        fh.write(f"- Tahmini tepe RAM: **{predicted_ram:,.0f} MB ({predicted_ram/1024:.1f} GB)**\n")
        fh.write(f"- Tahmini sure: **{predicted_time:,.0f} sn ({predicted_time/3600:.1f} saat)**\n")
        fh.write(f"- Guvenlik payiyla: {needed:,.0f} MB ({needed/1024:.1f} GB)\n")
        fh.write(f"- **Karar: {verdict}**\n")
    print(f"\nRapor: {report_path}")


if __name__ == "__main__":
    main()
