"""
ADIM A: Hukuk Tokenizer'i icin Turkce Hukuk Metni Derlemesi
=============================================================
8 farkli Hugging Face veri setinden (Ictihat / Mevzuat / QA) KARAKTER
sayisina gore oranli olarak metin cekilir, temizlenir, TUM kaynaklar
arasinda ORTAK tek bir hash havuzuyla kopyalari elenir ve
output/turkish_law_corpus.txt dosyasina satir satir yazilir.

Kategori paylari (toplam hedef: 120 MB):
  Ictihat (%60) : hamzabagirsakci(40) + emsal_kararlar(8) + erdem-erdem(8)
                  + legal_nli(2) + mteb_aym(2)                 [agirlik=60]
  Mevzuat (%30) : hasankursun(20) + muhamparlak_mevzuat(10)    [agirlik=30]
  QA      (%10) : orioncaf_qa(10)                              [agirlik=10]

Onemli notlar:
  - hamzabagirsakci ve erdem-erdem AYNI mahkemelerden (Yargitay/Danistay)
    BAGIMSIZ IKI FARKLI scrape oldugu icin ictihat metinleri buyuk
    ihtimalle ortusuyor. Bu yuzden TUM kaynaklar (sadece ictihat degil)
    tek bir global hash havuzunda dedup edilir -- daha guvenli ve basit.
  - Mevzuat (~32 MB) ve QA (~4.2 MB) kaynaklarinin benzersiz metin hacmi
    kendi kategori hedeflerinin (36 MB / 12 MB) altinda kaliyor. Bu
    kategoriler icin, toplanan essiz satirlar karistirilip HEDEFE ULASANA
    KADAR TEKRAR TEKRAR eklenir (oversampling). Amac bir dil modeli
    egitmek degil, BPE birlesme istatistiklerinde bu kategorinin agirligini
    spec'teki orana tasimak oldugu icin bu bilincli bir tekrar stratejisidir.
  - hamzabagirsakci'nin 5 alt-config'i (aym_bb/aym_norm/danistay/emsal/
    yargitay) ayri ayri esit paylarla okunur; tek "all" akisindan okumak
    parquet shard sirasina bagli olarak tek bir mahkeme turune
    (orn. sadece aym_bb) asiri agirlik verebilirdi.

Kullanim: python step1_law_collect_corpus.py
Cikti   : output/turkish_law_corpus.txt
"""

import sys
import time
import random
import logging
from pathlib import Path
from datasets import load_dataset

# ── Ayarlar ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUT_FILE = OUTPUT_DIR / "turkish_law_corpus.txt"
LOG_FILE = OUTPUT_DIR / "step1_law.log"

TOTAL_TARGET_CHARS = 120_000_000  # ~120 MB hedef derlem (15K vocab icin fazlasiyla yeterli)
CATEGORY_SHARE = {"ictihat": 0.60, "mevzuat": 0.30, "qa": 0.10}

MIN_LEN = 10  # bu uzunluktan kisa satirlar (bos basliklar vb.) atilir
SEED = 42
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


# ── Yardimci fonksiyonlar ────────────────────────────────────────────────────
def clean(text):
    """Bosluklari sadelestirir ve encode edilemeyen karakterleri sessizce atar.

    hamzabagirsakci ve hasankursun kaynaklarinda bazi 'S' (Ş) ve 'G' (Ğ)
    harfleri, kaynak taraftaki bir scraping hatasi (UTF-8 cok-baytli
    karakterin ilk baytinin duşmesi) yuzunden bozuk (lone surrogate)
    geliyor. Bu karakterler dosyaya yazilirken Python'da UnicodeEncodeError
    firlatir; bu yuzden once utf-8 encode/decode ile sessizce temizlenir.
    """
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return " ".join(text.split())


def collect_source(ds, field_names, target_chars, label, buffer, global_seen):
    """Bir kaynaktan (streaming dataset) metin toplar.

    field_names birden fazla alan icerirse (orn. QA icin ["question","answer"])
    HER biri ayri satir(lar) olarak eklenir -- ilk-dolu-alani-al mantigi degil,
    hepsi kullanilir. Her alanin icerigi '\\n' iceriyorsa paragraflara bolunur.

    global_seen: TUM kaynaklar arasinda paylasilan normalize-metin hash seti.
    Ayni karar/madde birden fazla datasette gecebildigi icin (orn.
    hamzabagirsakci ile erdem-erdem ayni mahkeme kararlarini bagimsiz
    scrape etmis olabilir) dedup TEK bir ortak havuzda yapilir.

    Dondurur: bu kaynaktan gercekten toplanan (essiz) karakter sayisi.
    """
    written_chars = 0
    written_lines = 0
    t0 = time.time()

    for sample in ds:
        if written_chars >= target_chars:
            break
        for field in field_names:
            if written_chars >= target_chars:
                break
            raw = sample.get(field)
            if not raw or not isinstance(raw, str):
                continue
            for para in raw.split("\n"):
                if written_chars >= target_chars:
                    break
                line = clean(para)
                if len(line) < MIN_LEN:
                    continue
                h = hash(line)
                if h in global_seen:
                    continue
                global_seen.add(h)
                buffer.append(line)
                written_chars += len(line)
                written_lines += 1

        if written_lines and written_lines % 50_000 == 0:
            t = time.time() - t0
            log.info(
                f"  [{label}] {written_chars/1024**2:6.1f}/{target_chars/1024**2:.1f} MB"
                f"  ({written_lines:,} satir, {t:.0f} sn)"
            )

    t = time.time() - t0
    log.info(f"  [{label}] TAMAM: {written_chars/1024**2:.2f} MB ({written_lines:,} satir) | {t:.0f} sn")
    return written_chars


def fill_by_oversampling(buffer, category_target_chars, current_chars, label):
    """Kategori hedefine essiz veriyle ulasilamadiysa (kaynak kucukse),
    toplanan essiz satirlari karistirip tekrar tekrar ekleyerek hedefe
    tamamlar (oversampling). Zaten hedefi gecmis/esit kategoriler icin
    hicbir sey yapmaz.
    """
    if current_chars >= category_target_chars or not buffer:
        return buffer, current_chars

    needed = category_target_chars - current_chars
    log.info(
        f"  [{label}] essiz veri hedefin altinda kaldi "
        f"({current_chars/1024**2:.1f}/{category_target_chars/1024**2:.1f} MB) -> "
        f"{needed/1024**2:.1f} MB, essiz satirlar tekrarlanarak (oversampling) tamamlaniyor"
    )

    pool = list(buffer)  # sabit kaynak havuz (extra'ya eklerken buyumesin diye kopya)
    extra = []
    extra_chars = 0
    while extra_chars < needed:
        random.shuffle(pool)
        for line in pool:
            if extra_chars >= needed:
                break
            extra.append(line)
            extra_chars += len(line)

    buffer.extend(extra)
    return buffer, current_chars + extra_chars


def main():
    log.info("=" * 60)
    log.info("ADIM A: Hukuk Korpusu Derlemesi")
    log.info(f"Hedef toplam: {TOTAL_TARGET_CHARS/1024**2:.0f} MB")
    log.info("=" * 60)

    global_seen = set()
    category_buffers = {"ictihat": [], "mevzuat": [], "qa": []}
    category_chars = {"ictihat": 0, "mevzuat": 0, "qa": 0}
    category_targets = {k: int(TOTAL_TARGET_CHARS * v) for k, v in CATEGORY_SHARE.items()}

    t_all = time.time()

    # ============================================================
    # ICTIHAT (%60) -- agirlik toplami = 60
    # ============================================================
    ic_target = category_targets["ictihat"]
    ic_weight_sum = 60

    # -- hamzabagirsakci/turkish-court-decisions (agirlik 40) --------------
    hb_target = int(ic_target * 40 / ic_weight_sum)
    hb_configs = ["aym_bb", "aym_norm", "danistay", "emsal", "yargitay"]
    per_cfg_target = hb_target // len(hb_configs)
    log.info(f"\n[hamzabagirsakci] hedef {hb_target/1024**2:.1f} MB, {len(hb_configs)} alt-config'e esit bolunuyor")
    for cfg in hb_configs:
        try:
            ds = load_dataset(
                "hamzabagirsakci/turkish-court-decisions",
                name=cfg, split="train", streaming=True,
            )
            n = collect_source(
                ds, ["text"], per_cfg_target, f"hamzabagirsakci/{cfg}",
                category_buffers["ictihat"], global_seen,
            )
            category_chars["ictihat"] += n
        except Exception as e:
            log.warning(f"  [UYARI] hamzabagirsakci/{cfg} atlandi: {e}")

    # -- muhamparlak/turkish-law-bge-m3-embeddings [emsal_kararlar] (agirlik 8) --
    ek_target = int(ic_target * 8 / ic_weight_sum)
    log.info(f"\n[emsal_kararlar] hedef {ek_target/1024**2:.1f} MB")
    ds = load_dataset("muhamparlak/turkish-law-bge-m3-embeddings", name="emsal_kararlar", split="train", streaming=True)
    ds = ds.remove_columns(["vector"])  # 1024-boyutlu embedding gereksiz, indirmeyi/RAM'i sisirir
    category_chars["ictihat"] += collect_source(
        ds, ["text"], ek_target, "emsal_kararlar", category_buffers["ictihat"], global_seen,
    )

    # -- erdem-erdem/Turkish-Law-Documents-700k-clustered (agirlik 8) ------
    ee_target = int(ic_target * 8 / ic_weight_sum)
    log.info(f"\n[erdem-erdem] hedef {ee_target/1024**2:.1f} MB")
    ds = load_dataset("erdem-erdem/Turkish-Law-Documents-700k-clustered", split="train", streaming=True)
    category_chars["ictihat"] += collect_source(
        ds, ["text"], ee_target, "erdem-erdem", category_buffers["ictihat"], global_seen,
    )

    # -- Turkish-NLI/legal_nli_TR_V1 (agirlik 2) -- premise + hypothesis ----
    nli_target = int(ic_target * 2 / ic_weight_sum)
    log.info(f"\n[legal_nli] hedef {nli_target/1024**2:.1f} MB")
    ds = load_dataset("Turkish-NLI/legal_nli_TR_V1", split="train", streaming=True)
    category_chars["ictihat"] += collect_source(
        ds, ["premise", "hypothesis"], nli_target, "legal_nli", category_buffers["ictihat"], global_seen,
    )

    # -- mteb/turkish-constitutional-court-violation-clean (agirlik 2) -----
    mteb_target = int(ic_target * 2 / ic_weight_sum)
    log.info(f"\n[mteb_aym] hedef {mteb_target/1024**2:.1f} MB")
    ds = load_dataset("mteb/turkish-constitutional-court-violation-clean", split="train", streaming=True)
    category_chars["ictihat"] += collect_source(
        ds, ["text"], mteb_target, "mteb_aym", category_buffers["ictihat"], global_seen,
    )

    category_buffers["ictihat"], category_chars["ictihat"] = fill_by_oversampling(
        category_buffers["ictihat"], ic_target, category_chars["ictihat"], "ictihat",
    )

    # ============================================================
    # MEVZUAT (%30) -- agirlik toplami = 30
    # ============================================================
    mv_target = category_targets["mevzuat"]
    mv_weight_sum = 30

    # -- hasankursun/turkish-legislation-corpus (agirlik 20) ----------------
    hk_target = int(mv_target * 20 / mv_weight_sum)
    log.info(f"\n[hasankursun] hedef {hk_target/1024**2:.1f} MB")
    ds = load_dataset("hasankursun/turkish-legislation-corpus", split="train", streaming=True)
    category_chars["mevzuat"] += collect_source(
        ds, ["text"], hk_target, "hasankursun", category_buffers["mevzuat"], global_seen,
    )

    # -- muhamparlak/turkish-law-bge-m3-embeddings [mevzuat] (agirlik 10) ---
    mm_target = int(mv_target * 10 / mv_weight_sum)
    log.info(f"\n[muhamparlak_mevzuat] hedef {mm_target/1024**2:.1f} MB")
    ds = load_dataset("muhamparlak/turkish-law-bge-m3-embeddings", name="mevzuat", split="train", streaming=True)
    ds = ds.remove_columns(["vector"])
    category_chars["mevzuat"] += collect_source(
        ds, ["text"], mm_target, "muhamparlak_mevzuat", category_buffers["mevzuat"], global_seen,
    )

    category_buffers["mevzuat"], category_chars["mevzuat"] = fill_by_oversampling(
        category_buffers["mevzuat"], mv_target, category_chars["mevzuat"], "mevzuat",
    )

    # ============================================================
    # QA (%10) -- OrionCAF/turkish_law_qa_dataset -- question + answer -----
    # ============================================================
    qa_target = category_targets["qa"]
    log.info(f"\n[orioncaf_qa] hedef {qa_target/1024**2:.1f} MB")
    ds = load_dataset("OrionCAF/turkish_law_qa_dataset", split="train", streaming=True)
    category_chars["qa"] += collect_source(
        ds, ["question", "answer"], qa_target, "orioncaf_qa", category_buffers["qa"], global_seen,
    )

    category_buffers["qa"], category_chars["qa"] = fill_by_oversampling(
        category_buffers["qa"], qa_target, category_chars["qa"], "qa",
    )

    # ============================================================
    # BIRLESTIR + KARISTIR + YAZ
    # ============================================================
    log.info("\nTum kategoriler birlestiriliyor ve karistiriliyor (shuffle)...")
    all_lines = category_buffers["ictihat"] + category_buffers["mevzuat"] + category_buffers["qa"]
    random.shuffle(all_lines)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    total_chars = sum(category_chars.values())
    out_mb = OUT_FILE.stat().st_size / 1024**2
    elapsed = (time.time() - t_all) / 60

    log.info("\n" + "=" * 60)
    log.info("ADIM A TAMAMLANDI")
    log.info(f"  Toplam satir : {len(all_lines):,}")
    log.info(f"  Ictihat      : {category_chars['ictihat']/1024**2:7.1f} MB  (%{100*category_chars['ictihat']/total_chars:.1f})")
    log.info(f"  Mevzuat      : {category_chars['mevzuat']/1024**2:7.1f} MB  (%{100*category_chars['mevzuat']/total_chars:.1f})")
    log.info(f"  QA           : {category_chars['qa']/1024**2:7.1f} MB  (%{100*category_chars['qa']/total_chars:.1f})")
    log.info(f"  Dosya boyutu : {out_mb:.1f} MB -> {OUT_FILE}")
    log.info(f"  Gecen sure   : {elapsed:.1f} dakika")
    log.info("=" * 60)
    log.info("Siradaki: python step2_law_train_tokenizer.py")


if __name__ == "__main__":
    main()
