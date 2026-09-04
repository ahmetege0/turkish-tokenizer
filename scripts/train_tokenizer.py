"""
128K Turkce Tokenizer Egitimi
=============================
Projenin ana cikti adimi: train/ altindaki 21.7M satirlik (5.97 GB) korpusun
TAMAMI uzerinde 128.000 kelimelik BPE tokenizer egitir.

Konfigurasyon smoke_test_tokenizer.py'den IMPORT ediliyor (kopyalanmiyor).
Sebep: egitilen tokenizer ile dogrulanan tokenizer'in ayarlari asla ayrisamaz.
Tek fark vocab_size -- smoke test 2.000 ile calisir, burada 128.000.

Egitim bitince smoke test'in AYNI kontrolleri 128K tokenizer'a uygulanir,
boylece "3-5 saat surdu ama saglam mi" sorusu beklemeden cevaplanir.

BILINEN SINIRLAMA -- checkpoint/resume YOK:
train_from_iterator tek, bloklayan bir Rust cagrisi. Colab baglantisi
kopunca o ana kadarki ilerleme tamamen gider, bastan baslanir. Kutuphanenin
yapisal sinirlamasi, bizim kodumuzla cozulebilecek bir sey degil.

Girdi : data/train/*.txt
Cikti : data/tokenizer.json      (--lines verilirse tokenizer_trial_*.json)
        data/train_report.md
        data/train.log

Kullanim:
    # tam egitim (3-5 saat, YUKSEK RAM runtime onerilir)
    python scripts/train_tokenizer.py --data /content/drive/MyDrive/tr-tokenizer-data

    # once kucuk bir deneme (zaman/RAM hissi icin, ayri dosyaya yazar)
    python scripts/train_tokenizer.py --data ... --lines 2000000

Colab'de is bitince runtime'i kapatmak icin (--shutdown alt surecte
CALISMAZ, hucrenin kendisi yapmali):
    !python scripts/train_tokenizer.py --data ...
    from google.colab import runtime
    runtime.unassign()
"""

import argparse
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from tokenizers.trainers import BpeTrainer

# Tarif TEK KAYNAKTAN gelsin diye smoke test'ten aliniyor.
# build_tokenizer  : normalizer + pre-tokenizer + decoder (NFKC, Metaspace)
# SPECIAL_TOKENS   : Secenek B, 5 ozel + 27 rezerv
# MAX_TOKEN_LEN    : 20 (karakter uzayinda)
# test_*           : egitim sonrasi calistirilacak dogrulamalar
from smoke_test_tokenizer import (
    MAX_TOKEN_LEN,
    MIN_FREQUENCY,
    SPECIAL_TOKENS,
    build_tokenizer,
    test_common_turkish_unk,
    test_exotic_unicode,
    test_hf_load,
    test_max_token_length,
    test_roundtrip,
    test_special_ids,
    test_turkish_casing,
)

# ── Ayarlar ──────────────────────────────────────────────────────────────────

TARGET_VOCAB   = 128_000
HEARTBEAT_SEC  = 300      # 5 dakikada bir ilerleme logu (proje kurali)

log = logging.getLogger("train")


# ── RAM okuma (tasinabilir) ──────────────────────────────────────────────────

def read_ram_mb():
    """
    ANLIK RAM (MB). Egitim ilerledikce RAM'in nasil tirmandigini canli
    gormek icin -- kumulatif tepe degil, su andaki kullanim.
    Linux'ta /proc/self/status, degilse psutil, ikisi de yoksa None.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024      # kB -> MB
    except OSError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def peak_ram_mb():
    """
    Surecin TEPE RAM'i (MB). ru_maxrss surecin tum omru boyunca hic azalmayan
    bir tavan; bu surec tek bir egitim yaptigi icin dogrudan o egitimin tepesi.
    Linux disinda (orn. yerelde Windows) None doner.
    """
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        return None


# ── Veri akisi ───────────────────────────────────────────────────────────────

def iter_train_lines(train_dir, progress, max_lines=None):
    """
    train/ altindaki butun shard'lari satir satir verir (generator).

    Tum korpusu bellege ALMAZ: 5.97 GB akis halinde gecer. Egitimde RAM'i
    belirleyen sey ham veri degil, trainer'in kurdugu kelime-frekans ve
    merge tablolaridir.

    progress: kalp atisi thread'inin okudugu paylasilan sayac sozlugu.
              Boylece "su an kacinci shard, kacinci satir" canli gorunur.
    max_lines: None ise tamami; sayi verilirse o kadar satirda durur (deneme).
    """
    shards = sorted(train_dir.glob("*.txt"))
    if not shards:
        raise SystemExit(f"Shard bulunamadi: {train_dir}")

    progress["shards_total"] = len(shards)
    for index, path in enumerate(shards, start=1):
        progress["shard"] = index
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                progress["lines"] += 1
                yield line
                if max_lines and progress["lines"] >= max_lines:
                    return


# ── Kalp atisi ───────────────────────────────────────────────────────────────

def start_heartbeat(progress, started_at):
    """
    train_from_iterator TEK, bloklayan bir Rust cagrisi -- Python tarafindan
    icine log satiri sokamayiz. Ama tokenizers Rust hesaplama sirasinda GIL'i
    biraktigi icin arka planda bir Python thread'i calisabiliyor.

    3-5 saatlik gozetimsiz calismada tek canli sinyalimiz bu: gecen sure,
    korpusun ne kadarinin okundugu ve ANLIK RAM.

    (stop_event, thread) dondurur; cagiran taraf 'finally' icinde kapatir.
    """
    stop = threading.Event()

    def beat():
        while not stop.wait(HEARTBEAT_SEC):
            ram = read_ram_mb()
            ram_text = f"{ram:,.0f} MB" if ram else "?"
            log.info(
                f"  ... {(time.time() - started_at) / 60:.0f} dk gecti | "
                f"shard {progress['shard']}/{progress['shards_total']} | "
                f"{progress['lines']:,} satir okundu | RAM {ram_text}"
            )

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    return stop, thread


# ── Egitim sonrasi dogrulama ─────────────────────────────────────────────────

def run_validation(tok):
    """
    Smoke test'in AYNI kontrollerini 128K tokenizer'a uygular.
    Fonksiyonlar smoke_test_tokenizer'dan import edildigi icin testler
    birebir ayni -- iki dosyanin zamanla ayrisma riski yok.

    Bloklayici olanlar MEKANIK ozellikleri sinar (korpus zenginliginden
    bagimsiz). [ORAN] olanlar karakter-seviyesi kapsamayi olcer; kucuk
    olcekte kirilgan olabildikleri icin genel sonucu belirlemezler,
    ama 128K/21.7M olceginde yuksek cikmalari beklenir.
    """
    tests = [
        ("[ORAN] Roundtrip",                     test_roundtrip,          False),
        ("[ORAN] Yaygin Turkce <unk>",           test_common_turkish_unk, False),
        ("Special token ID'leri",                test_special_ids,        True),
        ("[ORAN] Turkce buyuk/kucuk harf",       test_turkish_casing,     False),
        ("[BILGI] Egzotik Unicode",              test_exotic_unicode,     False),
        ("PreTrainedTokenizerFast yukleme",      test_hf_load,            True),
        (f"max_token_length <= {MAX_TOKEN_LEN}", test_max_token_length,   True),
    ]

    results = []
    blocking_failures = 0

    for name, fn, blocking in tests:
        log.info(f"  [{name}]")
        passed, failed = fn(tok)
        if blocking:
            blocking_failures += failed
            status = "✓ GECTI" if failed == 0 else f"✗ {failed} BASARISIZ"
        elif passed + failed > 0:
            status = f"ℹ {passed}/{passed + failed} (%{100 * passed / (passed + failed):.0f})"
        else:
            status = "ℹ bilgi amacli, yukarida"
        results.append((name, passed, failed, status))
        log.info(f"    → {status}")

    return results, blocking_failures


# ── Rapor ────────────────────────────────────────────────────────────────────

def write_report(path, tok, args, stats, results):
    """Egitim ozetini insan-okunabilir markdown olarak yazar."""
    vocab = tok.get_vocab()
    longest = sorted(vocab, key=len, reverse=True)[:15]

    lines = [
        "# 128K Tokenizer Egitim Raporu", "",
        f"Tarih: {datetime.now():%Y-%m-%d %H:%M}", "",
        "## Konfigurasyon", "",
        "| Ayar | Deger |", "|---|---|",
        f"| Model | BPE (byte_fallback yok) |",
        f"| Normalizer | NFKC (lowercase yok) |",
        f"| Pre-tokenizer / Decoder | Metaspace (karakter seviyesi) |",
        f"| vocab_size | {args.vocab:,} |",
        f"| min_frequency | {MIN_FREQUENCY} |",
        f"| max_token_length | {MAX_TOKEN_LEN} |",
        f"| Special token | {len(SPECIAL_TOKENS)} (5 ozel + 27 rezerv) |",
        "",
        "## Sonuc", "",
        f"- Okunan satir: **{stats['lines']:,}**",
        f"- Sure: **{stats['minutes']:.1f} dakika**",
        f"- Tepe RAM: **{stats['peak_ram']:,.0f} MB "
        f"({stats['peak_ram'] / 1024:.1f} GB)**" if stats["peak_ram"] else "- Tepe RAM: olculemedi",
        f"- Ulasilan vocab: **{tok.get_vocab_size():,}**",
        "",
        "## Dogrulama", "",
        "| Kontrol | Gecen | Basarisiz | Sonuc |", "|---|---:|---:|---|",
    ]
    for name, passed, failed, status in results:
        lines.append(f"| {name} | {passed} | {failed} | {status} |")

    lines += ["", "## En uzun 15 token", "",
              "`max_token_length` calisiyorsa hicbiri "
              f"{MAX_TOKEN_LEN} karakteri asmamali.", "",
              "| uzunluk | token |", "|---:|---|"]
    for token in longest:
        lines.append(f"| {len(token)} | `{token}` |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Colab kapatma ────────────────────────────────────────────────────────────

def shutdown_colab():
    """
    Is bitince runtime'i kapatir ki bosuna islem birimi yanmasin.
    runtime.unassign() IPython kernel'i uzerinden calistigi icin
    "!python ..." ALT SURECINDEN cagrilinca patlar -- o durumda cokmek
    yerine kullaniciya hucreye eklenecek iki satiri gosteriyoruz.
    """
    try:
        from google.colab import drive, runtime
    except ImportError:
        print("Colab disinda calisiliyor; runtime kapatma atlandi.")
        return

    logging.shutdown()
    try:
        drive.flush_and_unmount()
        print("Drive bosaltildi.")
    except Exception as e:
        print(f"Drive flush atlandi ({type(e).__name__}).")

    try:
        runtime.unassign()
    except Exception:
        print("\n" + "=" * 66)
        print("RUNTIME OTOMATIK KAPANAMADI: alt surecte IPython kernel yok.")
        print("Egitim BITTI, cikti dosyalari yazildi. Kapatmak icin hucreye:")
        print("    from google.colab import runtime")
        print("    runtime.unassign()")
        print("=" * 66)


# ── Ana akis ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="128K Turkce tokenizer egitimi")
    parser.add_argument("--data", default="data",
                        help="Veri dizini (icerisinde train/ olmali)")
    parser.add_argument("--vocab", type=int, default=TARGET_VOCAB,
                        help=f"Hedef vocab boyutu (varsayilan {TARGET_VOCAB:,})")
    parser.add_argument("--lines", type=int, default=None,
                        help="Sadece ilk N satir (deneme icin). Verilirse cikti "
                             "AYRI dosyaya yazilir, asil tokenizer.json ezilmez.")
    parser.add_argument("--shutdown", action="store_true",
                        help="Is bitince (veya hata alinca) Colab runtime'ini kapat")
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise SystemExit(f"train dizini bulunamadi: {train_dir}")

    # Deneme calistirmalari asil ciktiyi EZMESIN
    out_name = "tokenizer.json" if args.lines is None else f"tokenizer_trial_{args.lines}.json"
    out_path = data_dir / out_name

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(data_dir / "train.log", encoding="utf-8")],
    )

    log.info("=" * 66)
    log.info(f"128K TOKENIZER EGITIMI | vocab={args.vocab:,}")
    log.info(f"Girdi : {train_dir}")
    log.info(f"Cikti : {out_path}")
    if args.lines:
        log.info(f"DENEME modu: sadece ilk {args.lines:,} satir")
    log.info(f"Kalp atisi: {HEARTBEAT_SEC // 60} dakikada bir")
    log.info("=" * 66)

    tok = build_tokenizer()          # smoke test ile AYNI: NFKC + Metaspace
    trainer = BpeTrainer(
        vocab_size=args.vocab,
        min_frequency=MIN_FREQUENCY,
        max_token_length=MAX_TOKEN_LEN,
        special_tokens=list(SPECIAL_TOKENS),
        # initial_alphabet YOK: Metaspace'te sabit alfabe garantisi yok,
        # trainer alfabeyi korpusta gordugu karakterlerden kurar.
    )

    progress = {"shard": 0, "shards_total": 0, "lines": 0}
    started_at = time.time()
    stop, thread = start_heartbeat(progress, started_at)

    try:
        tok.train_from_iterator(
            iter_train_lines(train_dir, progress, args.lines), trainer=trainer
        )
    except BaseException as e:
        log.exception(f"EGITIM YARIDA KESILDI: {type(e).__name__}: {e}")
        log.info("Checkpoint yok -- bastan baslanmasi gerekiyor.")
        raise
    finally:
        stop.set()
        thread.join(timeout=1)

    minutes = (time.time() - started_at) / 60
    peak = peak_ram_mb()
    log.info(f"\nEgitim bitti: {minutes:.1f} dk | vocab={tok.get_vocab_size():,} "
             f"| tepe RAM {f'{peak:,.0f} MB' if peak else '?'}")

    # ONCE KAYDET, sonra dogrula -- saatlerce suren is bir test hatasi
    # yuzunden kaybolmasin.
    tok.save(str(out_path))
    log.info(f"Tokenizer kaydedildi: {out_path}")

    log.info("\nDogrulama calistiriliyor...\n")
    results, blocking_failures = run_validation(tok)

    stats = {"lines": progress["lines"], "minutes": minutes, "peak_ram": peak}
    report_path = data_dir / "train_report.md"
    write_report(report_path, tok, args, stats, results)

    log.info("\n" + "=" * 66)
    if blocking_failures == 0:
        log.info("SONUC: Bloklayici kontrollerin TAMAMI gecti.")
    else:
        log.info(f"SONUC: {blocking_failures} BLOKLAYICI KONTROL BASARISIZ.")
        log.info("Tokenizer kaydedildi ama kullanilmadan once incelenmeli.")
    log.info(f"Rapor: {report_path}")
    log.info("=" * 66)

    if args.shutdown:
        shutdown_colab()

    sys.exit(1 if blocking_failures else 0)


if __name__ == "__main__":
    main()
