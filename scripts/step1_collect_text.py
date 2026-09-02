"""
Adim 1: Tokenizer Egitimi icin Turkce Metin Derlemesi
=====================================================
MAX_FILE_SIZE_GB = 5.0 ile sinirli.
Her datasetten oraniyla pay alacak sekilde yazilir.

Kullanim: python step1_collect_text.py
Cikti:
  output/turkish_corpus_raw.txt
  output/turkish_corpus_shuffled.txt
"""

import sys, time, random, logging, os
from pathlib import Path
from datasets import load_dataset

# ── Ayarlar ──────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path(__file__).parent.parent / "output"
RAW_FILE     = OUTPUT_DIR / "turkish_corpus_raw.txt"
SHUF_FILE    = OUTPUT_DIR / "turkish_corpus_shuffled.txt"
LOG_FILE     = OUTPUT_DIR / "step1.log"

MAX_FILE_SIZE_BYTES = int(5.0 * 1024**3)   # 5 GB hard limit

# Her kaynaktan istenen satir sayisi (buyutuldu)
# 5 GB ~ 22M satir @ ort 234 byte/satir
# Oran korunarak dagitiliyor:
#   cosmos   %43 | fineweb2 %14 | forum %29 | musteri %9 | havadis %5
QUOTAS = {
    "cosmos":    9_000_000,
    "fineweb2":  3_000_000,
    "forum":     6_000_000,   # 11 alt-dataset paylasir
    "musteri":   2_000_000,
    "havadis":   1_500_000,
}

FORUM_CONFIGS = [
    "donanimarsivi", "donanimhaber", "forumum", "iyinet",
    "kadinlarklubu", "memurlar", "tahribat", "technopatsosyal",
    "turkiyeforum", "wardom", "wmaraci",
]

MIN_LEN  = 10
MAX_LEN  = 2000
SEED     = 42
random.seed(SEED)

# ── Logging ──────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Yardimci ─────────────────────────────────────────────────────────────────
def get_text(sample, fields):
    for f in fields:
        v = sample.get(f)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def clean(text):
    text = " ".join(text.split())
    return text

def file_size(path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0

def write_lines(ds, fields, quota, label, out_f, size_tracker):
    """
    ds          : streaming dataset
    fields      : metin alanlari
    quota       : bu kaynaktan max satir
    label       : log etiketi
    out_f       : acik dosya
    size_tracker: [toplam_byte] listesi (mutables)
    Dondurur: yazilan satir sayisi, boyut limitine ulasildi mi
    """
    written = skipped = 0
    t0 = time.time()
    size_hit = False

    for sample in ds:
        if written >= quota:
            break
        if size_tracker[0] >= MAX_FILE_SIZE_BYTES:
            size_hit = True
            break

        raw = get_text(sample, fields)
        if not raw:
            skipped += 1
            continue

        paragraphs = raw.split("\n") if "\n" in raw else [raw]
        for para in paragraphs:
            if written >= quota or size_tracker[0] >= MAX_FILE_SIZE_BYTES:
                break
            line = clean(para.strip())
            if len(line) < MIN_LEN:
                skipped += 1
                continue
            encoded = (line + "\n").encode("utf-8")
            out_f.write(line + "\n")
            size_tracker[0] += len(encoded)
            written += 1

        if written % 100_000 == 0 and written > 0:
            t = time.time() - t0
            spd = written / t if t > 0 else 1
            cur_gb = size_tracker[0] / 1024**3
            log.info(
                f"  [{label}] {written:>8,}/{quota:,}"
                f"  {spd:,.0f} sat/sn"
                f"  kalan ~{(quota-written)/spd/60:.1f} dk"
                f"  | dosya {cur_gb:.2f} GB"
            )

    t = time.time() - t0
    if size_hit:
        log.info(f"  [{label}] DURDURULDU (5 GB limiti): {written:,} satir | {t:.0f} sn")
    else:
        log.info(f"  [{label}] TAMAM: {written:,} satir | {t:.0f} sn")
    return written, size_hit


def main():
    total_quota = sum(QUOTAS.values())
    log.info("=" * 60)
    log.info("ADIM 1: Turkce metin derleme")
    log.info(f"Hedef    : {total_quota:,} satir")
    log.info(f"Boyut lim: {MAX_FILE_SIZE_BYTES/1024**3:.1f} GB")
    log.info("=" * 60)

    # Paylaşılan boyut takipçisi (pass-by-reference için liste)
    size_tracker = [0]
    total_written = 0
    t_all = time.time()
    stopped = False

    with open(RAW_FILE, "w", encoding="utf-8") as out_f:

        # ── COSMOS ──────────────────────────────────────────────
        if not stopped:
            log.info("\n[cosmos] yukleniyor...")
            ds = load_dataset(
                "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0",
                split="train", streaming=True,
            )
            n, stopped = write_lines(ds, ["text"], QUOTAS["cosmos"],
                                     "cosmos", out_f, size_tracker)
            total_written += n

        # ── FINEWEB2 ─────────────────────────────────────────────
        if not stopped:
            log.info("\n[fineweb2] yukleniyor (tur_Latn)...")
            ds = load_dataset(
                "HuggingFaceFW/fineweb-2",
                name="tur_Latn", split="train", streaming=True,
            )
            n, stopped = write_lines(ds, ["text"], QUOTAS["fineweb2"],
                                     "fineweb2", out_f, size_tracker)
            total_written += n

        # ── FORUM (11 alt-config) ─────────────────────────────────
        if not stopped:
            log.info(f"\n[forum] {len(FORUM_CONFIGS)} alt-config yukleniyor...")
            per_cfg = QUOTAS["forum"] // len(FORUM_CONFIGS)
            forum_written = 0
            for cfg in FORUM_CONFIGS:
                if stopped:
                    break
                log.info(f"  -> {cfg} ({per_cfg:,} satir)")
                try:
                    ds = load_dataset(
                        "turkish-nlp-suite/ForumSohbetleri",
                        name=cfg, split="train", streaming=True,
                    )
                    n, stopped = write_lines(
                        ds, ["texts", "text", "content", "body", "post"],
                        per_cfg, f"forum/{cfg}", out_f, size_tracker,
                    )
                    forum_written += n
                except Exception as e:
                    log.warning(f"  [UYARI] {cfg} atildi: {e}")
            log.info(f"[forum] TOPLAM: {forum_written:,} satir")
            total_written += forum_written

        # ── MUSTERI ───────────────────────────────────────────────
        if not stopped:
            log.info("\n[musteri] yukleniyor...")
            ds = load_dataset(
                "turkish-nlp-suite/MusteriYorumlari",
                split="train", streaming=True,
            )
            n, stopped = write_lines(
                ds, ["text", "review", "content"], QUOTAS["musteri"],
                "musteri", out_f, size_tracker,
            )
            total_written += n

        # ── HAVADIS ───────────────────────────────────────────────
        if not stopped:
            log.info("\n[havadis] yukleniyor...")
            ds = load_dataset(
                "turkish-nlp-suite/Havadis",
                split="train", streaming=True,
            )
            n, stopped = write_lines(
                ds, ["text", "content", "article"], QUOTAS["havadis"],
                "havadis", out_f, size_tracker,
            )
            total_written += n

    raw_mb = RAW_FILE.stat().st_size / 1024**2
    log.info(f"\nHam dosya: {raw_mb:.1f} MB | {total_written:,} satir")

    # ── SHUFFLE ───────────────────────────────────────────────────
    log.info("\nKaristirma (shuffle) basladi...")
    with open(RAW_FILE, encoding="utf-8") as f:
        lines = f.readlines()
    log.info(f"  {len(lines):,} satir bellegede, karistiriliyor...")
    random.shuffle(lines)
    with open(SHUF_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)
    shuf_mb = SHUF_FILE.stat().st_size / 1024**2
    log.info(f"  Karistirilmis: {shuf_mb:.1f} MB -> {SHUF_FILE.name}")

    elapsed = (time.time() - t_all) / 60
    log.info("\n" + "=" * 60)
    log.info("ADIM 1 TAMAMLANDI")
    log.info(f"  Toplam satir     : {total_written:,}")
    log.info(f"  Ham dosya boyutu : {raw_mb/1024:.2f} GB")
    log.info(f"  Limit asimi      : {'EVET - limit ile durduruldu' if stopped else 'HAYIR - kota bitti'}")
    log.info(f"  Gecen sure       : {elapsed:.1f} dakika")
    log.info("=" * 60)
    log.info("Siradaki: python step2_train_tokenizer.py")

if __name__ == "__main__":
    main()

