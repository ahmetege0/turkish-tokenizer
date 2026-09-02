"""
Kaynak Dogrulama (probe)
========================
Her veri kaynagindan SADECE ilk birkac kaydi okur ve sunlari raporlar:
  - Repo erisilebiliyor mu, hangi config'ler var
  - Kayitlarin alan adlari (key'leri) ve tipleri
  - String alanlarin ornek icerigi
  - Ortalama satir uzunlugu (kota hesabi icin)

Korpus YAZMAZ, veri INDIRMEZ. Tek cikti: output/sources_probe.json

Onemli: Bir kaynak patlarsa sessizce gecmez, sonda "BASARISIZ" olarak listelenir.
(Onceki denemede ForumSohbetleri sessizce 0 satir dondurup fark edilmemisti.)

Kullanim: python scripts/probe_sources.py
"""

import json
import traceback
from pathlib import Path

from datasets import load_dataset, get_dataset_config_names

OUT_DIR = Path("output")
OUT_FILE = OUT_DIR / "sources_probe.json"

N_PREVIEW = 3      # icerigi basilacak kayit sayisi
N_MEASURE = 200    # ortalama uzunluk icin okunacak kayit sayisi
PREVIEW_CHARS = 200

# (etiket, [aday repo id'leri], istenen config veya None)
# Aday listesi: ilki calismazsa sonraki denenir (OpenSubtitles icin gerekli).
SOURCES = [
    ("musteri",   ["turkish-nlp-suite/MusteriYorumlari"],                      None),
    ("forum",     ["turkish-nlp-suite/ForumSohbetleri"],                       None),
    ("altyazi",   ["Helsinki-NLP/open_subtitles", "open_subtitles"],           "en-tr"),
    ("wiki",      ["turkish-nlp-suite/temiz-Wiki"],                            None),
    ("havadis",   ["turkish-nlp-suite/Havadis"],                               None),
    ("ozenli",    ["turkish-nlp-suite/OzenliDerlem"],                          None),
    ("akademik",  ["turkish-nlp-suite/AkademikDerlem"],                        None),
    ("cosmos",    ["ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0"],                None),
    ("fineweb2",  ["HuggingFaceFW/fineweb-2"],                                 "tur_Latn"),
]


def describe_record(rec):
    """Bir kaydin alanlarini {ad: (tip, onizleme)} olarak ozetler."""
    out = {}
    for k, v in rec.items():
        if isinstance(v, str):
            preview = v[:PREVIEW_CHARS].replace("\n", "\n")
            out[k] = {"type": "str", "len": len(v), "preview": preview}
        else:
            out[k] = {"type": type(v).__name__, "value": str(v)[:80]}
    return out


def pick_text_field(rec):
    """En uzun string alani metin alani olarak tahmin eder."""
    best, best_len = None, -1
    for k, v in rec.items():
        if isinstance(v, str) and len(v) > best_len:
            best, best_len = k, len(v)
    return best


def probe_one(label, repo_ids, config):
    """Tek bir kaynagi dener. Basarili olursa dict, olmazsa exception firlatir."""
    last_error = None

    for repo in repo_ids:
        try:
            print(f"  repo deneniyor: {repo}")

            try:
                configs = get_dataset_config_names(repo)
            except Exception as e:
                configs = []
                print(f"    (config listesi alinamadi: {type(e).__name__})")

            if configs:
                print(f"    mevcut config'ler ({len(configs)}): {configs[:15]}")

            # Istenen config yoksa ilk config'e dus
            use_config = config
            if use_config and configs and use_config not in configs:
                print(f"    UYARI: '{use_config}' config listesinde YOK -> '{configs[0]}' denenecek")
                use_config = configs[0]
            elif not use_config and configs:
                use_config = configs[0]

            kwargs = {"split": "train", "streaming": True}
            if use_config:
                kwargs["name"] = use_config

            ds = load_dataset(repo, **kwargs)
            it = iter(ds)

            previews, lengths, text_field = [], [], None
            for i in range(N_MEASURE):
                try:
                    rec = next(it)
                except StopIteration:
                    break
                if i == 0:
                    text_field = pick_text_field(rec)
                if i < N_PREVIEW:
                    previews.append(describe_record(rec))
                if text_field and isinstance(rec.get(text_field), str):
                    lengths.append(len(rec[text_field]))

            if not previews:
                raise RuntimeError("stream acildi ama hic kayit gelmedi")

            avg = sum(lengths) / len(lengths) if lengths else 0

            return {
                "repo": repo,
                "config_used": use_config,
                "configs_available": configs,
                "fields": sorted(previews[0].keys()),
                "text_field_guess": text_field,
                "records_measured": len(lengths),
                "avg_chars_per_record": round(avg, 1),
                "previews": previews,
            }

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            print(f"    BASARISIZ: {last_error}")

    raise RuntimeError(last_error or "bilinmeyen hata")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, failures = {}, {}

    for label, repo_ids, config in SOURCES:
        print(f"\n{'='*70}\n[{label}]")
        try:
            results[label] = probe_one(label, repo_ids, config)
            r = results[label]
            print(f"  OK  alanlar={r['fields']}")
            print(f"      metin alani tahmini='{r['text_field_guess']}'  "
                  f"ort={r['avg_chars_per_record']} karakter/kayit")
            first = r["previews"][0].get(r["text_field_guess"], {})
            print(f"      ornek: {first.get('preview', '')[:150]}")
        except Exception:
            failures[label] = traceback.format_exc(limit=2)
            print(f"  >>> {label} PROBE EDILEMEDI")

    OUT_FILE.write_text(
        json.dumps({"ok": results, "failed": list(failures)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'='*70}")
    print(f"BASARILI ({len(results)}): {list(results)}")
    print(f"BASARISIZ ({len(failures)}): {list(failures)}")
    for label, tb in failures.items():
        print(f"\n--- {label} ---\n{tb}")
    print(f"\nRapor: {OUT_FILE}")

    if failures:
        print("\nDIKKAT: Basarisiz kaynaklar var. Cekmeye baslamadan once cozulmeli.")


if __name__ == "__main__":
    main()
