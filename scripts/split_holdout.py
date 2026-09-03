"""
Egitim / Held-out Ayrimi
========================
Korpusu ikiye boler. Held-out seti FIZIKSEL olarak ayrilir -- kopyalanmaz,
egitim tarafindan cikarilir. Boylece eğitimde yanlislikla kullanilmasi
mumkun olmaz.

Neden gerekli: fertility ve compression olcumlerini egitim verisi uzerinde
yaparsak kendi tokenizer'imiz yapay olarak iyi gorunur, cunku tam o
satirlardaki birlesmeleri ogrenmistir. Rakip tokenizer'lar (BERTurk, XLM-R,
Cosmos) korpusumuzu zaten hic gormedi; olcumun adil olmasi icin bizimkinin de
gormemis olmasi gerekiyor.

Secim RASTGELE DEGIL, hash tabanli: blake2b(satir) % M == 0 -> held-out.
Boylece ayrim tekrar uretilebilir; script'i yeniden calistirsan ayni satirlar
ayni tarafa duser. random.seed'e ve onu saklamaya gerek kalmaz.

Kaynak etiketi HER IKI tarafta da korunuyor:
  - held-out'ta sart (register bazli degerlendirme icin)
  - egitimde de lazim (RAM tavani olcumunde temsili alt kume almak icin)

Girdi : data/shards/*.txt + data/state.json
Cikti : data/train/<kaynak>_NNNN.txt
        data/holdout/<kaynak>.txt
        data/split_report.md

Kullanim:
    python scripts/split_holdout.py
    python scripts/split_holdout.py --data /content/drive/MyDrive/tr-tokenizer-data
"""

import argparse
import json
import time
from collections import defaultdict
from hashlib import blake2b
from pathlib import Path

HOLDOUT_LINES = 100_000     # hedef toplam held-out satiri
MIN_PER_SOURCE = 2_000      # kucuk kaynaklar da degerlendirmede temsil edilsin
SHARD_LINES = 500_000       # egitim shard'i basina satir (collect_corpus ile ayni)
WRITE_BUFFER = 1 << 20      # 1 MB; Drive'da cok sayida kucuk yazma yavas


def holdout_key(s):
    """
    Held-out secimi icin hash. person=b"holdout" ile dedup anahtarindan
    BAGIMSIZ: ayni satir dedup'ta ve burada iliskisiz sayilar uretir.
    """
    return int.from_bytes(
        blake2b(s.encode("utf-8"), digest_size=8, person=b"holdout").digest(), "big"
    )


def plan_split(state):
    """
    Kaynak basina "kacta bir satir held-out'a gitsin" degerini hesaplar.

    Oran her kaynak icin ayni (HOLDOUT_LINES / toplam), bu yuzden orantili
    temsil kendiliginden cikar. Ama kucuk kaynaklarda bu cok az satir birakir
    (vitamin'de ~745), fertility olcumu icin yetersiz kalir -- o yuzden
    MIN_PER_SOURCE tabani var.
    """
    total = sum(v.get("lines", 0) for v in state.values())
    if not total:
        raise SystemExit("state.json'da satir sayisi yok.")

    plan = {}
    for name, v in state.items():
        lines = v.get("lines", 0)
        if lines == 0:
            continue
        want = max(MIN_PER_SOURCE, round(lines * HOLDOUT_LINES / total))
        # Tavan: bir kaynagin en fazla %10'u held-out'a gidebilir.
        # Tavan olmasa, MIN_PER_SOURCE'tan kucuk bir kaynakta want == lines olur,
        # modulo 1'e duser ve "x % 1 == 0" hep dogru oldugu icin o kaynagin
        # TAMAMI held-out'a gider, egitime hic veri kalmaz.
        want = min(want, max(1, lines // 10))
        plan[name] = {"lines": lines, "want": want,
                      "modulo": max(2, round(lines / want))}
    return plan


class ShardWriter:
    """SHARD_LINES satirda bir yeni dosyaya gecer."""

    def __init__(self, out_dir, name):
        self.out_dir, self.name = out_dir, name
        self.index = self.count = 0
        self.fh = None

    def write(self, line):
        if self.fh is None or self.count >= SHARD_LINES:
            self.close()
            self.index += 1
            self.fh = open(self.out_dir / f"{self.name}_{self.index:04d}.txt",
                           "w", encoding="utf-8", buffering=WRITE_BUFFER)
            self.count = 0
        self.fh.write(line + "\n")
        self.count += 1

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="korpus klasoru")
    args = ap.parse_args()

    data_dir = Path(args.data)
    shard_dir = data_dir / "shards"
    train_dir = data_dir / "train"
    holdout_dir = data_dir / "holdout"
    state_file = data_dir / "state.json"

    if not state_file.exists():
        raise SystemExit(f"{state_file} yok. Once collect_corpus.py calistirilmali.")
    shards = sorted(shard_dir.glob("*.txt"))
    if not shards:
        raise SystemExit(f"Shard bulunamadi: {shard_dir}")

    train_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_split(json.loads(state_file.read_text(encoding="utf-8")))

    print("Ayrim plani (state.json'a gore):\n")
    print(f"{'kaynak':10} {'toplam satir':>14} {'held-out hedef':>16} {'kacta bir':>11}")
    for name, p in sorted(plan.items()):
        print(f"{name:10} {p['lines']:>14,} {p['want']:>16,} {p['modulo']:>11,}")
    print()

    writers = {}
    holdout_fh = {}
    counts = defaultdict(lambda: {"train": 0, "holdout": 0,
                                  "train_chars": 0, "holdout_chars": 0})

    total = 0
    t0 = t_log = time.time()

    for path in shards:
        source = path.stem.rsplit("_", 1)[0]     # forum_0003.txt -> forum
        if source not in plan:
            print(f"UYARI: {path.name} state.json'da yok, atlaniyor")
            continue

        modulo = plan[source]["modulo"]
        if source not in writers:
            writers[source] = ShardWriter(train_dir, source)
            holdout_fh[source] = open(holdout_dir / f"{source}.txt", "w",
                                      encoding="utf-8", buffering=WRITE_BUFFER)

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                total += 1
                c = counts[source]

                if holdout_key(line) % modulo == 0:
                    holdout_fh[source].write(line + "\n")
                    c["holdout"] += 1
                    c["holdout_chars"] += len(line)
                else:
                    writers[source].write(line)
                    c["train"] += 1
                    c["train_chars"] += len(line)

                if time.time() - t_log >= 30:
                    t_log = time.time()
                    print(f"  {total:>12,} satir islendi "
                          f"({total / (time.time() - t0):,.0f}/sn)  {path.name}")

    for w in writers.values():
        w.close()
    for f in holdout_fh.values():
        f.close()

    # ── Rapor ──
    tr_lines = sum(c["train"] for c in counts.values())
    ho_lines = sum(c["holdout"] for c in counts.values())
    tr_chars = sum(c["train_chars"] for c in counts.values())
    ho_chars = sum(c["holdout_chars"] for c in counts.values())

    rows = ["# Egitim / Held-out Ayrimi", "",
            f"- Egitim: **{tr_lines:,}** satir, {tr_chars:,} karakter "
            f"({tr_chars / 1e9:.2f} GB)",
            f"- Held-out: **{ho_lines:,}** satir, {ho_chars:,} karakter "
            f"({ho_chars / 1e6:.1f} MB)",
            f"- Held-out payi: **%{100 * ho_lines / max(total, 1):.2f}**", "",
            "Secim hash tabanli ve tekrar uretilebilir "
            "(`blake2b(satir, person=b\"holdout\") % M == 0`).", "",
            "| kaynak | egitim satir | held-out satir | held-out hedef | held-out % |",
            "|---|---:|---:|---:|---:|"]
    for name in sorted(counts):
        c = counts[name]
        tot = c["train"] + c["holdout"]
        rows.append(f"| {name} | {c['train']:,} | {c['holdout']:,} | "
                    f"{plan[name]['want']:,} | "
                    f"{100 * c['holdout'] / max(tot, 1):.2f} |")

    report = data_dir / "split_report.md"
    report.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"  egitim   : {tr_lines:>12,} satir  {tr_chars / 1e9:>6.2f} GB")
    print(f"  held-out : {ho_lines:>12,} satir  {ho_chars / 1e6:>6.1f} MB  "
          f"(%{100 * ho_lines / max(total, 1):.2f})")
    print("=" * 74)
    print(f"{'kaynak':10} {'egitim':>13} {'held-out':>11} {'hedef':>9}")
    for name in sorted(counts):
        c = counts[name]
        print(f"{name:10} {c['train']:>13,} {c['holdout']:>11,} {plan[name]['want']:>9,}")
    print("=" * 74)
    print(f"  sure: {(time.time() - t0) / 60:.1f} dk")
    print(f"\nEgitim  : {train_dir}")
    print(f"Held-out: {holdout_dir}")
    print(f"Rapor   : {report}")
    print("\nOrijinal shards/ klasoru DOKUNULMADI. Ayrim dogrulandiktan sonra silebilirsin.")


if __name__ == "__main__":
    main()
