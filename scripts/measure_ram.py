"""
RAM Tavani Olcumu
=================
128K egitiminin ne kadar RAM gerektirdigini TAHMIN ETMEDEN OLCMEK icin.

Ayni train verisinin daha temsili bir alt kumesi uzerinde artan vocab
boyutlariyla (8K, 16K, 32K) tokenizer egitir, her birinde tepe RAM ve
sureyi olcer. 3 noktadan 128K'ya ekstrapole eder.

Turkce eklemeli dil: benzersiz kelime tipi sayisi cok yuksek, BPE trainer
bunlarin hepsini frekansla bellekte tutuyor. 12.7 GB standart Colab RAM'i
yetmeyebilir.

Iki metodolojik duzeltme (ilk versiyonda vardi, giderildi):

  1) resource.ru_maxrss SURECIN TUM OMRU boyunca hic azalmayan bir tavan
     degeridir. Ayni process icinde 8K/16K/32K'yi sirayla olcmek, her
     olcumun KENDI tepe RAM'ini degil, O ANA KADARKI KUMULATIF tavani
     rapor eder. Duzeltme: her vocab boyutu AYRI bir alt-surecte
     (--single) olculuyor; ru_maxrss o zaman gercekten o surece ozel olur.

  2) Ilk versiyon sadece 500K satirlik (tam korpusun %2.3'u) sabit bir
     ornekte vocab boyutunu degistiriyordu -- korpus buyuklugunun RAM'e
     etkisi hic olculmuyordu. SAMPLE_LINES artik 3M (korpusun ~%14'u):
     hala tam veri degil ama Heaps yasasi geregi essiz kelime tipi
     cesitliliginin buyuk kismini yakalar.

Girdi : data/train/*.txt (orantili alt kume)
Cikti : konsol + data/ram_report.md

Kullanim:
    python scripts/measure_ram.py --data /content/drive/MyDrive/tr-tokenizer-data
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── Ayarlar ──────────────────────────────────────────────────────────────────

VOCAB_STEPS    = [8_000, 16_000, 32_000]
TARGET_VOCAB   = 128_000
MIN_FREQUENCY  = 2
MAX_TOKEN_LEN  = 28
SAMPLE_LINES   = 3_000_000     # ~%14 of full 21.7M-line corpus

SPECIAL_TOKENS = (
    ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    + [f"<reserved_{i}>" for i in range(27)]
)

RESULT_MARKER = "RESULT_JSON:"   # alt-surecin sonucu bu prefiksle basar


# ── RAM olcumu (sadece --single modunda, tek vocab, TEK surec) ──────────────

def get_peak_ram_mb():
    """Linux'ta (Colab) surecin tepe RAM kullanimini MB olarak dondurur.
    ru_maxrss surecin TUM omru boyunca hic azalmayan bir tavandir --
    bu yuzden dogru olmasi icin process baslangicindan itibaren
    SADECE BIR vocab boyutu olculmelidir (--single)."""
    import resource
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return rusage.ru_maxrss / 1024


def get_current_ram_mb():
    """Anlik RAM kullanimini MB olarak dondurur (baseline icin)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
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

    print(f"  {len(shards_by_source)} kaynaktan {len(lines):,} satir orneklendi",
          file=sys.stderr)
    return lines


# ── Egitim ───────────────────────────────────────────────────────────────────

def train_and_measure(lines, vocab_size):
    """Verilen vocab boyutuyla egitir, tepe RAM ve sure dondurur.
    Bu fonksiyon --single modunda cagrildiginda surecin TEK egitimi olur,
    bu yuzden get_peak_ram_mb() dogru (o surece ozel) deger dondurur."""
    from tokenizers import Tokenizer, normalizers, decoders, processors
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

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
    Olcum noktalarindan 128K'ya basit lineer ekstrapole (ilk ve son nokta).
    BPE'de merge tablosu vocab ile dogru orantili buyur; kelime frekans
    tablosu (artik daha buyuk ve temsili bir orneklemde olculuyor) sabit
    kalir. Bu yuzden RAM ~ a + b * vocab seklinde beklenir.
    """
    r1, r2 = results[0], results[-1]
    v1, m1 = r1["vocab_size"], r1["peak_ram_mb"]
    v2, m2 = r2["vocab_size"], r2["peak_ram_mb"]

    slope = (m2 - m1) / (v2 - v1) if v2 != v1 else 0
    predicted_ram = m1 + slope * (target_vocab - v1)

    t1, t2 = r1["elapsed_sec"], r2["elapsed_sec"]
    time_slope = (t2 - t1) / (v2 - v1) if v2 != v1 else 0
    predicted_time = t1 + time_slope * (target_vocab - v1)

    return predicted_ram, predicted_time


def check_linearity(results):
    """
    Ortadaki (16K) noktayi ekstrapolasyonda KULLANMIYORUZ (sadece ilk/son
    nokta kullaniliyor) ama onu dogrusallik SINAMASI icin kullanabiliriz:
    gercek 16K degeri, 8K-32K dogrusunun ustunde beklenenden cok sapiyorsa,
    RAM-vocab iliskisi dogrusal olmayabilir ve 128K'ya (32K'nin 4 kati
    otesine) ekstrapolasyon guvenilir olmayabilir.
    """
    if len(results) < 3:
        return None
    v1, v2, v3 = (r["vocab_size"] for r in results)
    m1, m2, m3 = (r["peak_ram_mb"] for r in results)
    expected_mid = m1 + (m3 - m1) * (v2 - v1) / (v3 - v1)
    deviation_pct = abs(m2 - expected_mid) / max(expected_mid, 1) * 100
    return expected_mid, deviation_pct


# ── Alt-surec modu: TEK vocab boyutu, TEK surec ──────────────────────────────

def run_single(args):
    """Bu surecte SADECE bir vocab boyutu egitilir. ru_maxrss bu yuzden
    dogrudur. Sonuc, ebeveyn surecin ayristirmasi icin ozel bir prefiksle
    stdout'a JSON olarak basilir."""
    train_dir = Path(args.data) / "train"
    lines = sample_lines(train_dir, args.lines)
    result = train_and_measure(lines, args.vocab)
    print(RESULT_MARKER + json.dumps(result))


# ── Ana surec: her vocab boyutu icin ayri alt-surec baslatir ────────────────

def run_orchestrator(args):
    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise SystemExit(f"train dizini bulunamadi: {train_dir}")

    print("=" * 60)
    print("RAM TAVANI OLCUMU")
    print(f"Ornek boyutu: {args.lines:,} satir (~%{100*args.lines/21_723_606:.0f} "
          f"tam korpusun)")
    print("Her vocab boyutu, dogru ru_maxrss icin AYRI bir alt-surecte olculuyor.")
    print("=" * 60)

    results = []
    for vocab in VOCAB_STEPS:
        print(f"\n{'─' * 60}")
        print(f"  Vocab: {vocab:,}  (izole alt-surec baslatiliyor...)")
        print(f"{'─' * 60}")
        t0 = time.time()

        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--single",
             "--vocab", str(vocab), "--data", str(data_dir), "--lines", str(args.lines)],
            capture_output=True, text=True,
        )
        wall = time.time() - t0

        if proc.stderr:
            print(proc.stderr.rstrip())

        if proc.returncode != 0:
            print(proc.stdout[-3000:])
            raise SystemExit(f"Alt-surec basarisiz oldu (vocab={vocab}, "
                             f"exit={proc.returncode})")

        line = next((ln for ln in proc.stdout.splitlines()
                    if ln.startswith(RESULT_MARKER)), None)
        if not line:
            print(proc.stdout)
            raise SystemExit(f"Alt-surecten sonuc satiri alinamadi (vocab={vocab})")

        r = json.loads(line[len(RESULT_MARKER):])
        results.append(r)
        print(f"  Sure (surec ici)  : {r['elapsed_sec']:.1f} sn")
        print(f"  Sure (toplam)     : {wall:.1f} sn")
        print(f"  RAM once          : {r['ram_before_mb']:.0f} MB")
        print(f"  RAM sonra         : {r['ram_after_mb']:.0f} MB")
        print(f"  Tepe RAM (dogru)  : {r['peak_ram_mb']:.0f} MB")

    # Ekstrapole
    predicted_ram, predicted_time = extrapolate(results, TARGET_VOCAB)
    linearity = check_linearity(results)

    print(f"\n{'=' * 60}")
    print(f"EKSTRAPOLE → {TARGET_VOCAB:,} vocab")
    print(f"{'=' * 60}")
    if linearity:
        expected_mid, deviation_pct = linearity
        print(f"  Dogrusallik kontrolu: 16K icin beklenen {expected_mid:.0f} MB, "
              f"olculen {results[1]['peak_ram_mb']:.0f} MB (sapma %{deviation_pct:.1f})")
        if deviation_pct > 15:
            print("  ⚠️  UYARI: sapma yuksek. RAM-vocab iliskisi dogrusal "
                  "olmayabilir; asagidaki tahmine ihtiyatla yaklas.")
    print(f"  Tahmini tepe RAM : {predicted_ram:,.0f} MB ({predicted_ram/1024:.1f} GB)")
    print(f"  Tahmini sure     : {predicted_time:,.0f} sn ({predicted_time/3600:.1f} saat)")
    print()

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

    report_path = data_dir / "ram_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("# RAM Tavani Olcum Raporu\n\n")
        fh.write(f"Ornek: {args.lines:,} satir (~%{100*args.lines/21_723_606:.0f} "
                 f"tam korpusun). Her vocab AYRI alt-surecte olculdu.\n\n")
        fh.write("| Vocab | Sure (sn) | Tepe RAM (MB) | Tepe RAM (GB) |\n")
        fh.write("|---:|---:|---:|---:|\n")
        for r in results:
            fh.write(f"| {r['vocab_size']:,} | {r['elapsed_sec']:.1f} "
                     f"| {r['peak_ram_mb']:,.0f} | {r['peak_ram_mb']/1024:.1f} |\n")
        if linearity:
            fh.write(f"\nDogrusallik sapmasi (16K): %{linearity[1]:.1f}\n")
        fh.write(f"\n## Ekstrapole → {TARGET_VOCAB:,}\n\n")
        fh.write(f"- Tahmini tepe RAM: **{predicted_ram:,.0f} MB "
                 f"({predicted_ram/1024:.1f} GB)**\n")
        fh.write(f"- Tahmini sure: **{predicted_time:,.0f} sn "
                 f"({predicted_time/3600:.1f} saat)**\n")
        fh.write(f"- Guvenlik payiyla: {needed:,.0f} MB ({needed/1024:.1f} GB)\n")
        fh.write(f"- **Karar: {verdict}**\n")
    print(f"\nRapor: {report_path}")


# ── Ana akis ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAM tavani olcumu")
    parser.add_argument("--data", type=str, default="data",
                        help="Veri dizini (icerisinde train/ olmali)")
    parser.add_argument("--lines", type=int, default=SAMPLE_LINES,
                        help="Orneklenecek satir sayisi")
    parser.add_argument("--single", action="store_true",
                        help=argparse.SUPPRESS)   # sadece alt-surec kendi kendini bu bayrakla cagirir
    parser.add_argument("--vocab", type=int, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single:
        if args.vocab is None:
            raise SystemExit("--single icin --vocab zorunlu")
        run_single(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
