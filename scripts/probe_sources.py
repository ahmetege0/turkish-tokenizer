"""
Kaynak Dogrulama (probe)
========================
Her veri kaynaginin HER config'inden sadece birkac kayit okur ve raporlar:
  - Repo/config erisilebiliyor mu
  - Alan adlari ve TIPLERI (str / list / int ...)
  - Metin alani hangisi (liste tipindekiler dahil)
  - Ortalama karakter/kayit  -> kota planlamasi icin

Korpus YAZMAZ, veri INDIRMEZ. Tek cikti: output/sources_probe.json

Neden her config ayri ayri: bir config'in alan semasi digerlerinden farkli
olabilir. Onceki denemede ForumSohbetleri'nin 'texts' alani LISTE oldugu icin
str kontrolunden gecemedi ve 2.78M kayit sessizce kayboldu.

Kullanim: python scripts/probe_sources.py
"""

import json
import traceback
from pathlib import Path

from datasets import load_dataset, get_dataset_config_names

OUT_DIR = Path("output")
OUT_FILE = OUT_DIR / "sources_probe.json"

N_PREVIEW = 2       # icerigi kaydedilecek kayit sayisi
N_MEASURE = 50      # ortalama uzunluk icin okunacak kayit sayisi
PREVIEW_CHARS = 600   # cumle sonunda kesilir, ortasinda degil

# (etiket, repo, configs)
#   configs = "ALL"  -> repo'daki tum config'ler yoklanir
#   configs = [...]  -> sadece bu config'ler (fineweb-2'de 1870 config var!)
SOURCES = [
    ("musteri",   "turkish-nlp-suite/MusteriYorumlari",             "ALL"),
    ("senti",     "turkish-nlp-suite/SentiTurca",                   "ALL"),
    ("sinema",    "turkish-nlp-suite/BuyukSinema",                  "ALL"),
    ("vitamin",   "turkish-nlp-suite/vitamins-supplements-reviews", "ALL"),
    ("forum",     "turkish-nlp-suite/ForumSohbetleri",              "ALL"),
    ("wiki",      "turkish-nlp-suite/temiz-Wiki",                   "ALL"),
    ("havadis",   "turkish-nlp-suite/Havadis",                      "ALL"),
    ("ozenli",    "turkish-nlp-suite/OzenliDerlem",                 "ALL"),
    ("akademik",  "turkish-nlp-suite/AkademikDerlem",               "ALL"),
    ("cosmos",    "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0",       "ALL"),
    ("fineweb2",  "HuggingFaceFW/fineweb-2",                        ["tur_Latn"]),
]


def text_len(v):
    """Bir alanin tasidigi toplam metin uzunlugu. Liste ise elemanlarin toplami."""
    if isinstance(v, str):
        return len(v)
    if isinstance(v, list):
        return sum(len(x) for x in v if isinstance(x, str))
    return -1


def clip(s):
    """Metni PREVIEW_CHARS civarinda, ama CUMLE SONUNDA keser."""
    s = " ".join(s.split())
    if len(s) <= PREVIEW_CHARS:
        return s
    tail = s[PREVIEW_CHARS:PREVIEW_CHARS + 200]
    cut = min((tail.find(c) for c in ".!?" if c in tail), default=-1)  # EN YAKIN cumle sonu
    if cut != -1:
        return s[:PREVIEW_CHARS + cut + 1]
    sp = s.rfind(" ", 0, PREVIEW_CHARS)
    return s[:sp if sp > 0 else PREVIEW_CHARS] + " ..."


def describe(v):
    """Bir alani {tip, uzunluk, onizleme} olarak ozetler. Listeleri de acar."""
    if isinstance(v, str):
        return {"type": "str", "len": len(v),
                "preview": clip(v)}
    if isinstance(v, list):
        items = [x for x in v if isinstance(x, str)]
        return {"type": "list", "n_items": len(v), "total_chars": sum(map(len, items)),
                "avg_item_chars": round(sum(map(len, items)) / len(items), 1) if items else 0,
                "preview": clip(items[0]) if items else ""}
    return {"type": type(v).__name__, "value": str(v)[:80]}


def probe_config(repo, config):
    """Tek bir repo+config'i yoklar. Hata olursa exception firlatir."""
    kwargs = {"split": "train", "streaming": True}
    if config:
        kwargs["name"] = config

    it = iter(load_dataset(repo, **kwargs))
    previews, records = [], []

    for i in range(N_MEASURE):
        try:
            rec = next(it)
        except StopIteration:
            break
        records.append(rec)
        if i < N_PREVIEW:
            previews.append({k: describe(v) for k, v in rec.items()})

    if not previews:
        raise RuntimeError("stream acildi ama hic kayit gelmedi")

    # Metin alani: TEK kayda degil, olculen TUM kayitlarin toplamina bakarak sec.
    # (Tek kayda bakmak yaniltir: bir yorumun urun adi, yorum metninden uzun olabilir.)
    field_totals = {}
    for rec in records:
        for k, v in rec.items():
            n = text_len(v)
            if n >= 0:
                field_totals[k] = field_totals.get(k, 0) + n
    text_field = max(field_totals, key=field_totals.get) if field_totals else None

    return {
        "fields": {k: previews[0][k]["type"] for k in previews[0]},
        "text_field": text_field,
        "field_avg_chars": {k: round(v / len(records), 1) for k, v in sorted(
            field_totals.items(), key=lambda kv: -kv[1])},
        "records_measured": len(records),
        "avg_chars_per_record": round(field_totals.get(text_field, 0) / len(records), 1),
        "previews": previews,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, failures = {}, {}

    for label, repo, want in SOURCES:
        print(f"\n{'='*72}\n[{label}]  {repo}")

        try:
            configs = get_dataset_config_names(repo)
        except Exception as e:
            configs = []
            print(f"  config listesi alinamadi: {type(e).__name__}: {e}")

        todo = configs if want == "ALL" else want
        if not todo:
            todo = [None]
        print(f"  yoklanacak config ({len(todo)}): {todo if len(todo) <= 12 else todo[:12]}")

        results[label] = {"repo": repo, "configs": {}}

        for cfg in todo:
            name = f"{label}/{cfg}" if cfg else label
            try:
                r = probe_config(repo, cfg)
                results[label]["configs"][str(cfg)] = r
                types = ", ".join(f"{k}:{v}" for k, v in r["fields"].items())
                print(f"    OK  {str(cfg):<24} alan[{types}]")
                print(f"        metin='{r['text_field']}'  ort={r['avg_chars_per_record']:,.0f} kar/kayit")
            except Exception:
                failures[name] = traceback.format_exc(limit=2)
                print(f"    >>> BASARISIZ: {name}")

    OUT_FILE.write_text(
        json.dumps({"ok": results, "failed": sorted(failures)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    n_ok = sum(len(v["configs"]) for v in results.values())
    print(f"\n{'='*72}")
    print(f"BASARILI config sayisi : {n_ok}")
    print(f"BASARISIZ config sayisi: {len(failures)}")
    for name, tb in failures.items():
        print(f"\n--- {name} ---\n{tb}")
    print(f"Rapor: {OUT_FILE}")
    if failures:
        print("\nDIKKAT: Basarisiz config'ler var. Cekmeye baslamadan once cozulmeli.")


if __name__ == "__main__":
    main()
