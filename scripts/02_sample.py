#!/usr/bin/env python3
"""
02_sample.py — build the annotation set from data/raw/corpus.jsonl

Why this exists: the raw scrape is dominated by a few huge international
videos (300–400 English comments each). A random sample would mostly be
English reaction chatter, not the Nepali / Romanized / code-mixed crisis
text this project is about. So we sample deliberately.

What it does
  1. Loads the raw corpus, drops rows shorter than MIN_CHARS.
  2. Drops rows whose parent video title matches EXCLUDE_TITLE_TERMS
     (e.g. the tribute-song video).
  3. Caps each parent video at MAX_PER_VIDEO rows so no one source dominates.
  4. Takes ALL remaining Nepali-script rows (devanagari, romanized_nepali,
     mixed) — that's the scarce, valuable part — up to the target.
  5. Fills the rest with English rows up to ENGLISH_FRAC of the total,
     so the classifier still learns a real "not Nepali / not relevant" class.
  6. Shuffles with a fixed seed and writes:
       data/interim/to_annotate.csv     <- open in Excel/Sheets and label
       data/interim/sample_report.txt   <- counts, sanity check

Run
    python scripts/02_sample.py
    python scripts/02_sample.py --target 1800 --english-frac 0.2 --max-per-video 25
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "corpus.jsonl"
OUT_DIR = ROOT / "data" / "interim"
OUT_CSV = OUT_DIR / "to_annotate.csv"
OUT_REPORT = OUT_DIR / "sample_report.txt"

SEED = 42
MIN_CHARS = 15
NEPALI_SCRIPTS = {"devanagari", "romanized_nepali", "mixed"}

# Any parent video whose title contains one of these (case-insensitive)
# is dropped entirely. Add to this list as you find more off-topic sources.
EXCLUDE_TITLE_TERMS = [
    "song",
    "गीत",        # song
    "भजन",        # devotional hymn
]

# Columns you will fill in during annotation. Left blank here on purpose.
ANNOTATION_COLUMNS = ["label", "sub_label", "location_mentioned", "notes"]

OUTPUT_COLUMNS = [
    "id", "text", "script", "platform", "kind", "timestamp",
    "source_url", "parent_title", "parent_url", "query",
] + ANNOTATION_COLUMNS


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_excluded(row: dict) -> bool:
    title = (row.get("parent_title") or "").lower()
    return any(term.lower() in title for term in EXCLUDE_TITLE_TERMS)


def cap_per_video(rows: list[dict], cap: int, rng: random.Random) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("parent_url") or r.get("source_url")].append(r)
    kept = []
    for items in groups.values():
        rng.shuffle(items)
        kept.extend(items[:cap])
    rng.shuffle(kept)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a stratified annotation sample.")
    ap.add_argument("--target", type=int, default=1800, help="total rows to sample")
    ap.add_argument("--english-frac", type=float, default=0.20,
                    help="max share of the sample that may be English (0–1)")
    ap.add_argument("--max-per-video", type=int, default=25,
                    help="max rows any single parent video may contribute")
    args = ap.parse_args()

    if not RAW.exists():
        raise SystemExit(f"raw corpus not found: {RAW}\nrun scripts/01_scrape.py first")

    rng = random.Random(SEED)

    # 1–2. load, length filter, title exclusions
    all_rows = load_rows(RAW)
    rows = [r for r in all_rows if len(r.get("text", "")) >= MIN_CHARS]
    n_short = len(all_rows) - len(rows)
    excluded = [r for r in rows if is_excluded(r)]
    rows = [r for r in rows if not is_excluded(r)]

    # 3. per-video cap, applied separately to each pool
    nepali_pool = cap_per_video([r for r in rows if r["script"] in NEPALI_SCRIPTS],
                                args.max_per_video, rng)
    english_pool = cap_per_video([r for r in rows if r["script"] == "english"],
                                 args.max_per_video, rng)

    # 4–5. budget: Nepali first, English fills the remainder up to its cap
    english_cap = int(args.target * args.english_frac)
    nepali_take = min(len(nepali_pool), args.target - english_cap)
    english_take = min(len(english_pool), args.target - nepali_take, english_cap)

    sample = nepali_pool[:nepali_take] + english_pool[:english_take]
    rng.shuffle(sample)

    # 6. write CSV with blank annotation columns
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sample:
            w.writerow({**r, **{c: "" for c in ANNOTATION_COLUMNS}})

    # report
    lines = [
        f"raw rows:                     {len(all_rows)}",
        f"dropped (< {MIN_CHARS} chars):        {n_short}",
        f"dropped (excluded titles):    {len(excluded)}",
        f"nepali pool after video cap:  {len(nepali_pool)}",
        f"english pool after video cap: {len(english_pool)}",
        "",
        f"SAMPLE: {len(sample)} rows -> {OUT_CSV}",
        f"  nepali-script: {nepali_take} ({nepali_take / len(sample):.0%})",
        f"  english:       {english_take} ({english_take / len(sample):.0%})",
        "",
        "by script:",
        *[f"  {s:18s} {c}" for s, c in Counter(r["script"] for r in sample).most_common()],
        "",
        "top 10 contributing videos:",
        *[f"  {c:3d}  {t[:72]}" for t, c in
          Counter(r.get("parent_title", "") for r in sample).most_common(10)],
    ]
    if excluded:
        lines += ["", "excluded videos:",
                  *[f"  {c:3d}  {t[:72]}" for t, c in
                    Counter(r.get("parent_title", "") for r in excluded).most_common()]]

    report = "\n".join(lines)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n-> {OUT_CSV}\n-> {OUT_REPORT}")


if __name__ == "__main__":
    main()