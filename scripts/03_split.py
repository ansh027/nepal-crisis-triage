#!/usr/bin/env python3
"""
03_split.py — split the labeled annotation CSV into train / dev / test.

Input
    data/interim/to_annotate.csv
        every row must have `label` filled; every `disaster` row must
        also have `sub_label`. The script refuses to run otherwise.

Output
    data/final/train.csv   70%
    data/final/dev.csv     15%
    data/final/test.csv    15%
    data/final/README.md   auto-generated dataset card — edit before release

Why stratified
    A plain random split can leave dev or test with zero rows of a rare
    class (rescue_needed has only 9 rows in this corpus). Stratifying on
    the combined label+sub_label key guarantees every class that has at
    least 3 rows is present in all three splits.

Usage
    pip install scikit-learn
    python scripts/03_split.py
    python scripts/03_split.py --train 0.7 --dev 0.15 --test 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    sys.exit("scikit-learn is required:  pip install scikit-learn")

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data" / "interim" / "to_annotate.csv"
OUT_DIR = ROOT / "data" / "final"

COLUMNS = [
    "id", "text", "script", "platform", "kind", "timestamp", "source_url",
    "parent_title", "parent_url", "query", "label", "sub_label",
    "location_mentioned", "notes",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load_rows(path: Path) -> list[dict]:
    """Read the CSV. utf-8-sig strips the BOM that Excel's CSV-UTF-8 adds."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"missing columns: {missing}\nfound: {reader.fieldnames}")
        rows = []
        for r in reader:
            if not (r.get("id") or "").strip():     #skip Excel's trailing blank lines
                continue
            r["label"] = (r.get("label") or "").strip()
            r["sub_label"] = (r.get("sub_label") or "").strip()
            rows.append(r)
        return rows


def validate(rows: list[dict]) -> None:
    unlabeled = [r["id"] for r in rows if not r["label"]]
    if unlabeled:
        sys.exit(f"{len(unlabeled)} rows have no label — first: {unlabeled[0]}")
    no_sub = [r["id"] for r in rows if r["label"] == "disaster" and not r["sub_label"]]
    if no_sub:
        sys.exit(f"{len(no_sub)} disaster rows have no sub_label — first: {no_sub[0]}")
    bad_labels = {r["label"] for r in rows} - {"disaster", "not_disaster"}
    if bad_labels:
        sys.exit(f"unexpected label values: {sorted(bad_labels)}")


def strat_key(r: dict) -> str:
    """Combined class key used for stratification and reporting."""
    return f"disaster/{r['sub_label']}" if r["label"] == "disaster" else "not_disaster"


def split(rows: list[dict], train: float, dev: float, seed: int):
    keys = [strat_key(r) for r in rows]
    counts = Counter(keys)

    # sklearn stratify needs >=2 members per class per split; classes with
    # fewer than 3 rows can't be spread across three splits — send them all
    # to train and warn, rather than crash.
    tiny = {k for k, n in counts.items() if n < 3}
    if tiny:
        print(f"warning: classes with <3 rows go entirely to train: {sorted(tiny)}")
    splittable = [r for r in rows if strat_key(r) not in tiny]
    tiny_rows = [r for r in rows if strat_key(r) in tiny]

    keys_s = [strat_key(r) for r in splittable]
    tr, rest = train_test_split(
        splittable, train_size=train, stratify=keys_s, random_state=seed
    )
    keys_rest = [strat_key(r) for r in rest]
    dev_share = dev / (1.0 - train)          # dev's share of the remainder
    dv, te = train_test_split(
        rest, train_size=dev_share, stratify=keys_rest, random_state=seed
    )
    return tr + tiny_rows, dv, te


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_readme(path: Path, all_rows, tr, dv, te) -> None:
    keys = sorted(Counter(strat_key(r) for r in all_rows))
    c_all, c_tr, c_dv, c_te = (Counter(strat_key(r) for r in s)
                               for s in (all_rows, tr, dv, te))
    lines = [
        "# Nepali Crisis-Text Triage Dataset",
        "",
        f"{len(all_rows)} hand-labeled comments about the 2026 Nepal floods and "
        f"landslides (Rasuwa, Bhotekoshi, Melamchi, Kathmandu Valley and related "
        f"events), collected Aug–Sep 2026 from public YouTube comments.",
        "",
        "## Files",
        "",
        f"| file | rows |",
        f"|---|---|",
        f"| train.csv | {len(tr)} |",
        f"| dev.csv | {len(dv)} |",
        f"| test.csv | {len(te)} |",
        "",
        "## Labels",
        "",
        "`label`: `disaster` (text is about the event) or `not_disaster`.",
        "`sub_label` (disaster rows only): `rescue_needed`, `damage`, "
        "`resource_available`, `warning`, `other`.",
        "Full definitions and edge-case rulings: `docs/annotation_guidelines.md`.",
        "",
        "## Class distribution",
        "",
        "| class | total | train | dev | test |",
        "|---|---|---|---|---|",
    ]
    for k in keys:
        lines.append(f"| {k} | {c_all[k]} | {c_tr.get(k, 0)} | {c_dv.get(k, 0)} | {c_te.get(k, 0)} |")
    lines += [
        "",
        "## Script mix",
        "",
        "| script | rows |",
        "|---|---|",
    ]
    for s, n in Counter(r["script"] for r in all_rows).most_common():
        lines.append(f"| {s} | {n} |")
    lines += [
        "",
        "## Collection and privacy",
        "",
        "- Source: YouTube Data API v3 comment threads on videos matched by "
        "Nepali and English flood/landslide queries.",
        "- Sampling: all Nepali-script rows (Devanagari, Romanized, mixed) kept; "
        "English capped at ~20%; max 25 comments per video; tribute-song videos "
        "excluded.",
        "- PII removed before storage: phone numbers, emails, @handles, URLs "
        "replaced with placeholder tokens. Author IDs are one-way hashed.",
        "- Labeled by a single annotator following the written guidelines; "
        "ambiguous rows were flagged and resolved in a second pass.",
        "",
        "## Known limitations",
        "",
        "- `disaster/other` dominates: YouTube comment sections skew toward "
        "reaction, grief and political commentary. Actionable triage classes "
        "(rescue_needed, damage, resource_available, warning) are rare here "
        "because real-time coordination happens on other platforms.",
        "- Single-annotator labels; no inter-annotator agreement measured.",
        "- Single event cluster (Aug–Sep 2026); generalisation to other "
        "disasters is untested.",
        "",
        "## License",
        "",
        "TODO — choose before release (CC BY 4.0 is a reasonable default).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--dev", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if abs(a.train + a.dev + a.test - 1.0) > 1e-9:
        sys.exit("--train + --dev + --test must equal 1.0")
    if not IN_CSV.exists():
        sys.exit(f"not found: {IN_CSV}")

    rows = load_rows(IN_CSV)
    validate(rows)
    tr, dv, te = split(rows, a.train, a.dev, a.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "train.csv", tr)
    write_csv(OUT_DIR / "dev.csv", dv)
    write_csv(OUT_DIR / "test.csv", te)
    write_readme(OUT_DIR / "README.md", rows, tr, dv, te)

    print(f"total {len(rows)}  ->  train {len(tr)} / dev {len(dv)} / test {len(te)}")
    print()
    print(f"{'class':28s} {'total':>6s} {'train':>6s} {'dev':>5s} {'test':>5s}")
    c_all, c_tr, c_dv, c_te = (Counter(strat_key(r) for r in s) for s in (rows, tr, dv, te))
    for k in sorted(c_all):
        print(f"{k:28s} {c_all[k]:6d} {c_tr.get(k,0):6d} {c_dv.get(k,0):5d} {c_te.get(k,0):5d}")
    print()
    for name in ("train.csv", "dev.csv", "test.csv", "README.md"):
        print(f"-> {OUT_DIR / name}")


if __name__ == "__main__":
    main()