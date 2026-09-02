"""
Korpus Toplama
==============
Turkce tokenizer egitimi icin, 9 kaynaktan register-dengeli korpus toplar.

Cekme sirasi = oncelik sirasi. Nadir register'lar (yorum, forum) once gelir;
genel web (cosmos, fineweb2) en sonda. Bir satir daha once yazildiysa tekrar
yazilmaz -- yani cakisan icerik hep NADIR olan kaynaga yazilmis olur.

Dedup cekme aninda yapilir (ayri bir temizlik gecisi yok). Kota sayaci sadece
KABUL EDILEN karakterde artar; boylece dedup sonrasi oranlar hedefte kalir.

Girdi : yok (HF Hub'dan streaming)
Cikti : data/shards/<kaynak>_NNNN.txt   korpus (duz metin, satir basina bir kayit)
        data/state.json                 resume durumu
        data/collect.log                log
        data/samples.txt                periyodik ornekler (gozle kalite kontrolu)

Kullanim:
    python scripts/collect_corpus.py --scale 0.001   # mini test (~6 MB)
    python scripts/collect_corpus.py                 # tam calisma (6 GB)
    python scripts/collect_corpus.py --fresh         # state'i yok say, bastan basla

Colab'de gozetimsiz calisma. runtime.unassign() IPython kernel'i gerektirdigi
icin "!python ..." alt surecinden calismaz; kapatmayi hucrenin kendisi yapar:

    !python scripts/collect_corpus.py --out /content/drive/MyDrive/tr-tokenizer-data --shutdown
    from google.colab import runtime
    runtime.unassign()
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from hashlib import blake2b
from itertools import islice
from pathlib import Path

from datasets import load_dataset

# ── Ayarlar ──────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")          # --out ile degistirilebilir (orn. Google Drive)
SHARD_DIR   = DATA_DIR / "shards"
STATE_FILE  = DATA_DIR / "state.json"
LOG_FILE    = DATA_DIR / "collect.log"
SAMPLE_FILE = DATA_DIR / "samples.txt"

WRITE_BUFFER = 1 << 20              # 1 MB. Google Drive'a yazarken cok sayida kucuk
                                    # yazma cok yavas; buyuk tampon bunu ortadan kaldirir.

TOTAL_GB = 6.0                  # hedef: kabul edilen karakter toplami

MIN_LINE_CHARS = 20             # bunun altindakiler boilerplate ("Devamini oku", "Paylas")
MAX_LINE_CHARS = 2000
MIN_WORDS = 3
MIN_LETTER_RATIO = 0.5          # harf orani dusukse: menu, sayi yigini, tablo artigi

MAX_CHARS_PER_RECORD = 20_000   # tek dev kayit kotayi domine etmesin
                                # (forum/memurlar'da ort. 106.000 karakter/kayit olctuk)

SHARD_LINES = 500_000           # bu kadar satirda yeni shard dosyasi
LOG_EVERY_SEC = 300             # ilerleme logu: 5 dakikada bir
SAVE_EVERY_SEC = 60             # state kaydi daha sik: kopmada en fazla 1 dk pozisyon kaybi
SAMPLE_EVERY = 200_000          # bu kadar kabul satirda bir, 1 ornek samples.txt'ye

# fineweb2 kayit filtresi (bu iki alan sadece fineweb2'de var)
MIN_LANG_SCORE = 0.90           # Turkce olma skoru
MAX_DUP_CLUSTER = 1_000         # internetteki kopya sayisi; probe'da 266.154 kopyali spam gorduk

# ── Kaynaklar: sira = oncelik. quota = TOTAL_GB'nin orani. ───────────────────
# Son kaynak (fineweb2) oncekilerin dolduramadigi acigi kapatir.
FORUM_CONFIGS = ["donanimarsivi", "donanimhaber", "forumum", "iyinet", "kadinlarklubu",
                 "memurlar", "tahribat", "technopatsosyal", "turkiyeforum", "wardom", "wmaraci"]

# "Havadis" config'i BILEREK yok: probe'da havadis dataset'iyle birebir ayni cikti.
OZENLI_CONFIGS = ["GeziNotlari", "KulturHaritasi", "MasalMasal", "Perdearkasi-Yorumlar",
                  "PopulerBilim", "Serzenisler", "SusluTrendler", "TeknoYazilar",
                  "ViralMedya", "YazarinKaleminden"]

SOURCES = [
    {"name": "senti",    "repo": "turkish-nlp-suite/SentiTurca",
     "field": "text",  "quota": 0.012, "configs": ["movies", "e-commerce", "hate"]},

    {"name": "vitamin",  "repo": "turkish-nlp-suite/vitamins-supplements-reviews",
     "field": "text",  "quota": 0.007, "configs": ["default"]},

    {"name": "forum",    "repo": "turkish-nlp-suite/ForumSohbetleri",
     "field": "texts", "quota": 0.250, "configs": FORUM_CONFIGS},   # 'texts' LISTE tipinde

    {"name": "wiki",     "repo": "turkish-nlp-suite/temiz-Wiki",
     "field": "text",  "quota": 0.100, "configs": ["default"]},

    {"name": "havadis",  "repo": "turkish-nlp-suite/Havadis",
     "field": "text",  "quota": 0.120, "configs": ["default"]},

    {"name": "ozenli",   "repo": "turkish-nlp-suite/OzenliDerlem",
     "field": "text",  "quota": 0.100, "configs": OZENLI_CONFIGS},

    {"name": "akademik", "repo": "turkish-nlp-suite/AkademikDerlem",
     "field": "text",  "quota": 0.050, "configs": ["makaleler", "akademik-ozetler",
                                                   "medikal-makaleler", "medikal-ozetler",
                                                   "bilkent-writings"]},

    {"name": "cosmos",   "repo": "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0",
     "field": "text",  "quota": 0.150, "configs": ["default"]},

    {"name": "fineweb2", "repo": "HuggingFaceFW/fineweb-2",
     "field": "text",  "quota": 0.200, "configs": ["tur_Latn"]},
]

log = logging.getLogger("collect")


# ── Satir isleme ─────────────────────────────────────────────────────────────
def clean_line(s):
    """Butun bosluk turlerini tek boslukla degistirir; satir sonu birakmaz."""
    return " ".join(s.split())


def reject_reason(s):
    """Satir elenmeliyse sebebini, gecerliyse None dondurur (sayaclar icin)."""
    if len(s) < MIN_LINE_CHARS or len(s) > MAX_LINE_CHARS:
        return "length"
    if s.count(" ") < MIN_WORDS - 1:
        return "few_words"
    if sum(c.isalpha() for c in s) < len(s) * MIN_LETTER_RATIO:
        return "low_letters"
    return None


def dedup_key(s):
    """
    Dedup anahtari. ORIJINAL satir degil, normalize edilmis hali hash'lenir:
    kucuk harf + sadece harf/rakam + bosluk sikistirma. Boylece sadece noktalama
    veya buyuk/kucuk harf farki olan tekrarlar da yakalanir.
    Dosyaya yazilan satir her zaman ORIJINAL kalir.

    blake2b kullaniliyor, Python'un hash()'i DEGIL: hash() string'ler icin her
    process'te farkli sonuc verir; resume sonrasi butun anahtarlar degisir,
    dedup coker.
    """
    norm = " ".join("".join(c if c.isalnum() else " " for c in s.lower()).split())
    return int.from_bytes(blake2b(norm.encode("utf-8"), digest_size=8).digest(), "big")


def split_long(text, limit):
    """
    Limitten uzun metni CUMLE SINIRLARINDAN ~limit'lik parcalara boler.

    Neden gerekli: OzenliDerlem ve AkademikDerlem'de metin tek parca geliyor,
    icinde hic satir sonu yok. Sadece "\\n" ile bolup uzun olani atinca
    ozenli'nin %69'unu, akademik'in %87'sini kaybediyorduk -- ustelik yanli
    sekilde, cunku sadece kisa makaleler hayatta kaliyordu.

    Noktalamasiz dev bir blok gelirse kelime sinirindan bolunur; hicbir sey
    sessizce atilmaz. Gercek cop zaten low_letters/few_words filtresine takilir.
    """
    if len(text) <= limit:
        yield text
        return

    buf = ""
    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        while len(sentence) > limit:          # noktalamasiz dev blok: kelimeden bol
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if buf:
                yield buf
                buf = ""
            yield sentence[:cut]
            sentence = sentence[cut:].lstrip()
        if len(buf) + len(sentence) + 1 > limit:
            if buf:
                yield buf
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        yield buf


def record_lines(record, field):
    """
    Bir kayittan ham satirlari uretir.
    Alan liste ise (forum'un 'texts' alani) her eleman ayri ayri islenir.
    Satir sonlarindan bolunur, cok uzun parcalar ayrica cumlelere ayrilir.
    """
    value = record.get(field)
    for part in (value if isinstance(value, list) else [value]):
        if isinstance(part, str):
            for raw in part.split("\n"):
                yield from split_long(raw, MAX_LINE_CHARS)


def record_ok(record, source_name):
    """Kayit seviyesi filtre. Su an sadece fineweb2'de bu meta alanlar var."""
    if source_name != "fineweb2":
        return True
    return (record.get("language_score", 1.0) >= MIN_LANG_SCORE
            and record.get("minhash_cluster_size", 1) <= MAX_DUP_CLUSTER)


# ── Shard yazici ─────────────────────────────────────────────────────────────
class ShardWriter:
    """SHARD_LINES satirda bir yeni dosyaya gecer. Dosya adi: <name>_0001.txt"""

    def __init__(self, name, start_index):
        self.name = name
        self.index = start_index
        self.count = 0
        self.fh = None

    def write(self, line):
        if self.fh is None or self.count >= SHARD_LINES:
            self.close()
            self.index += 1
            path = SHARD_DIR / f"{self.name}_{self.index:04d}.txt"
            self.fh = open(path, "w", encoding="utf-8", buffering=WRITE_BUFFER)
            self.count = 0
            log.info(f"  yeni shard: {path.name}")
        self.fh.write(line + "\n")
        self.count += 1

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


# ── Cikti yeri ve Colab kapatma ──────────────────────────────────────────────
def set_data_dir(path):
    """
    Cikti klasorunu belirler. Google Drive yolu verilirse (orn.
    /content/drive/MyDrive/tr-tokenizer-data) veriler runtime olse bile kalir
    ve sonraki oturumda resume calisir.
    """
    global DATA_DIR, SHARD_DIR, STATE_FILE, LOG_FILE, SAMPLE_FILE
    DATA_DIR = Path(path)
    SHARD_DIR = DATA_DIR / "shards"
    STATE_FILE = DATA_DIR / "state.json"
    LOG_FILE = DATA_DIR / "collect.log"
    SAMPLE_FILE = DATA_DIR / "samples.txt"


def shutdown_colab():
    """
    Colab runtime'ini kapatir; is bitince bosuna islem birimi yakilmasin.
    Sira onemli: once log dosyasi kapanir, sonra Drive'daki bekleyen yazmalar
    diske iner, en son runtime sonlandirilir.
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
        print(f"Drive flush atlandi ({type(e).__name__}); Drive bagli olmayabilir.")

    try:
        print("Colab runtime kapatiliyor.")
        runtime.unassign()
    except Exception:
        # unassign() runtime'i IPython kernel'i uzerinden (JavaScript ile) kapatiyor.
        # Script "!python ..." ile ALT SUREC olarak kosarsa o surecte kernel yoktur
        # ve cagri patlar. Isin kendisi bitmis durumda; kapatmayi notebook hucresi
        # yapmali. Cokmek yerine net talimat basiyoruz.
        print("\n" + "=" * 66)
        print("RUNTIME OTOMATIK KAPANAMADI: alt surecte IPython kernel yok.")
        print("Veri Drive'a yazildi, kayip YOK. Runtime'i kapatmak icin ayni")
        print("hucrenin sonuna su iki satiri ekle (! olmadan, Python olarak):")
        print("    from google.colab import runtime")
        print("    runtime.unassign()")
        print("=" * 66)


# ── Durum (resume) ───────────────────────────────────────────────────────────
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def rebuild_seen():
    """
    Resume'da: yazilmis shard'lari okuyup dedup set'ini yeniden kurar.
    Set'i diske yazmiyoruz -- shard'lar zaten durumun kendisi; bozulabilecek
    ikinci bir dosya tutmanin faydasi yok.
    """
    seen = set()
    shards = sorted(SHARD_DIR.glob("*.txt"))
    if not shards:
        return seen
    log.info(f"Resume: {len(shards)} shard okunup dedup seti yeniden kuruluyor...")
    t0 = time.time()
    for path in shards:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                seen.add(dedup_key(line.rstrip("\n")))
    log.info(f"  {len(seen):,} anahtar geri yuklendi ({time.time() - t0:.0f} sn)")
    return seen


# ── Tek kaynagi cek ──────────────────────────────────────────────────────────
def collect_source(source, quota_chars, seen, state, samples_fh):
    """Bir kaynagin butun config'lerini sirayla gezer, kota dolunca durur."""
    name = source["name"]
    st = state.setdefault(name, {"chars": 0, "lines": 0, "shard": 0, "configs": {}})
    writer = ShardWriter(name, st["shard"])

    counters = {"read": 0, "accepted": 0, "dup": 0,
                "length": 0, "few_words": 0, "low_letters": 0, "record_filtered": 0}
    t0 = t_log = t_save = time.time()

    log.info(f"[{name}] {source['repo']} | hedef {quota_chars / 1e6:,.0f} MB "
             f"| {len(source['configs'])} config")

    for config in source["configs"]:
        if st["chars"] >= quota_chars:
            break

        already = st["configs"].get(config, 0)   # resume: bu config'te kacinci kayittayiz
        try:
            stream = load_dataset(source["repo"], name=config, split="train", streaming=True)
        except Exception as e:
            # Sessizce gecmiyoruz: onceki denemede tam bu noktada 2.78M kayit kaybolmustu
            log.error(f"  [{name}/{config}] ACILAMADI: {type(e).__name__}: {e}")
            continue

        if already:
            log.info(f"  [{name}/{config}] resume: ilk {already:,} kayit atlaniyor")
            stream = islice(stream, already, None)

        n_records = already
        for record in stream:
            if st["chars"] >= quota_chars:
                break
            n_records += 1
            counters["read"] += 1

            if not record_ok(record, name):
                counters["record_filtered"] += 1
                continue

            chars_from_record = 0
            for raw in record_lines(record, source["field"]):
                if st["chars"] >= quota_chars or chars_from_record >= MAX_CHARS_PER_RECORD:
                    break

                line = clean_line(raw)
                reason = reject_reason(line)
                if reason:
                    counters[reason] += 1
                    continue

                key = dedup_key(line)
                if key in seen:            # kaynak-ici VE kaynaklar-arasi cakisma, tek mekanizma
                    counters["dup"] += 1
                    continue

                seen.add(key)
                writer.write(line)
                st["chars"] += len(line)
                st["lines"] += 1
                chars_from_record += len(line)
                counters["accepted"] += 1

                if st["lines"] % SAMPLE_EVERY == 0:
                    samples_fh.write(f"\n--- {name}/{config} @ {st['lines']:,} satir ---\n{line}\n")
                    samples_fh.flush()

            now = time.time()
            if now - t_log >= LOG_EVERY_SEC:
                t_log = now
                pct = 100 * st["chars"] / quota_chars if quota_chars else 100
                speed = counters["accepted"] / max(now - t0, 1)
                log.info(
                    f"  [{name}/{config}] {st['chars'] / 1e6:7.1f}/{quota_chars / 1e6:,.0f} MB "
                    f"({pct:5.1f}%) | kabul {counters['accepted']:>9,} ({speed:,.0f}/sn) "
                    f"| tekrar {counters['dup']:>9,} | eleme: uzunluk {counters['length']:,} "
                    f"kelime {counters['few_words']:,} harf {counters['low_letters']:,} "
                    f"kayit {counters['record_filtered']:,} "
                    # 75 byte/anahtar: olculdu (set tablosu ~27 B + int nesnesi ~48 B)
                    f"| set {len(seen) / 1e6:.1f}M (~{len(seen) * 75 / 1e9:.2f} GB)"
                )
            if now - t_save >= SAVE_EVERY_SEC:
                t_save = now
                st["configs"][config] = n_records
                st["shard"] = writer.index
                save_state(state)

        st["configs"][config] = n_records

    writer.close()
    st["shard"] = writer.index
    save_state(state)

    log.info(f"[{name}] BITTI: {st['chars'] / 1e6:,.1f} MB / {st['lines']:,} satir "
             f"| {counters['read']:,} kayit okundu | {(time.time() - t0) / 60:.1f} dk")
    log.info(f"  eleme dagilimi: {counters}")


# ── Ana akis ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="kotalari carpar; test icin 0.001 (~6 MB)")
    ap.add_argument("--fresh", action="store_true",
                    help="state.json'i yok say, bastan basla")
    ap.add_argument("--out", default="data",
                    help="cikti klasoru; Drive yolu verilirse runtime olse de kalir")
    ap.add_argument("--shutdown", action="store_true",
                    help="is bitince (veya hata alinca) Colab runtime'ini kapat")
    args = ap.parse_args()

    set_data_dir(args.out)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(LOG_FILE, encoding="utf-8")],
    )

    total_chars = int(TOTAL_GB * 1e9 * args.scale)
    state = {} if args.fresh else load_state()
    seen = set() if args.fresh else rebuild_seen()

    log.info("=" * 78)
    log.info(f"KORPUS TOPLAMA | hedef {total_chars / 1e6:,.0f} MB | scale={args.scale}")
    log.info(f"{len(SOURCES)} kaynak | dedup: blake2b-64 | log her {LOG_EVERY_SEC} sn")
    log.info(f"cikti: {DATA_DIR.resolve()}")
    log.info("=" * 78)

    t0 = time.time()
    try:
        with open(SAMPLE_FILE, "a", encoding="utf-8") as samples_fh:
            for i, source in enumerate(SOURCES):
                if i == len(SOURCES) - 1:
                    # Son kaynak oncekilerin dolduramadigi acigi kapatir
                    done = sum(state.get(s["name"], {}).get("chars", 0) for s in SOURCES[:-1])
                    quota = max(total_chars - done, 0)
                else:
                    quota = int(total_chars * source["quota"])
                collect_source(source, quota, seen, state, samples_fh)

        # ── Ozet ──
        grand = sum(v["chars"] for v in state.values())
        log.info("=" * 78)
        log.info("TOPLAMA TAMAMLANDI")
        for source in SOURCES:
            st = state.get(source["name"], {"chars": 0, "lines": 0})
            share = 100 * st["chars"] / grand if grand else 0
            log.info(f"  {source['name']:9} {st['chars'] / 1e6:9,.1f} MB  {st['lines']:>10,} satir  "
                     f"%{share:5.1f}  (hedef %{source['quota'] * 100:.1f})")
        log.info(f"  {'TOPLAM':9} {grand / 1e6:9,.1f} MB  "
                 f"{sum(v['lines'] for v in state.values()):>10,} satir")
        log.info(f"  essiz satir (dedup seti): {len(seen):,}")
        log.info(f"  sure: {(time.time() - t0) / 60:.1f} dk")
        log.info("=" * 78)

    except BaseException as e:
        # Hatayi da log dosyasina yaz: kullanici donduğunde ne oldugunu gorsun.
        log.exception(f"CALISMA YARIDA KESILDI: {type(e).__name__}: {e}")
        log.info("Shard'lar ve state.json diskte; ayni komutla --fresh OLMADAN "
                 "calistirirsan kaldigi yerden devam eder.")
        raise

    finally:
        # Hata da olsa basari da olsa: runtime bosuna islem birimi yakmasin.
        if args.shutdown:
            shutdown_colab()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    # datasets kutuphanesi arka planda indirme thread'leri birakiyor ve Python
    # cikista onlari bekliyor. Dosyalar kapandi, state kaydedildi, log bosaltildi;
    # asili kalmamak icin sureci burada bitiriyoruz.
    logging.shutdown()
    os._exit(exit_code)
