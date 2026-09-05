# Nepali Crisis-Text Triage

A hand-labeled dataset and baseline classifier for triaging Nepali-language
social media text during disaster response — built around the August–September
2026 Rasuwa and Bhotekoshi floods in Nepal.

## Why this exists

When disasters hit Nepal, official coordination (the BIPAD portal, the NEOC
hotline) struggles to keep up with the volume of unstructured reporting that
shows up on social media in real time — rescue requests, damage reports,
warnings, offers of help, all mixed in with reaction and political commentary.
Existing NLP tools for this kind of triage are built for English or
high-resource languages; Nepali — especially the Romanized and code-mixed
Nepali-English that actually dominates real usage — is underserved.

This project builds a small, carefully labeled dataset to test whether a
transformer-based classifier can separate disaster-relevant text from noise,
and where it holds up or breaks down across script types (Devanagari,
Romanized Nepali, code-mixed, English).

## What's here

- **A scraper** (`scripts/01_scrape.py`) that collects YouTube comments on
  Nepal flood/landslide videos via the official Data API, strips PII (phone
  numbers, emails, handles, URLs) before anything is stored, and tags each
  comment's script (Devanagari / Romanized Nepali / English / mixed).
- **A stratified sampler** (`scripts/02_sample.py`) that builds a balanced
  annotation set from the raw scrape — capping any single video's
  contribution and prioritizing Nepali-script content so the sample isn't
  dominated by a few large international news videos.
- **Annotation guidelines** (`docs/annotation_guidelines.md`) — the label
  schema and the edge-case rulings accumulated during labeling, so the
  dataset's decisions are documented and reproducible, not implicit.
- **1,131 hand-labeled rows** (`data/final/`), split 70/15/15 into
  train/dev/test, stratified so every class appears in every split.
- **A split script** (`scripts/03_split.py`) that regenerates the splits
  and a dataset card (`data/final/README.md`) from the labeled CSV.

## Labels

Two-step scheme, applied by hand to every row:

1. **`label`** — `disaster` (text is about the flood/landslide event in
   some way) or `not_disaster` (off-topic: generic praise, unrelated
   arguments, prayer-chain templates, spam).
2. **`sub_label`** (disaster rows only) — `rescue_needed`, `damage`,
   `resource_available`, `warning`, or `other` (reaction, political
   commentary, grief, speculation about cause — anything disaster-related
   that isn't a structured report or request).

Full definitions and every edge-case ruling are in
`docs/annotation_guidelines.md`.

## Dataset at a glance

| class | rows |
|---|---|
| disaster / other | 755 |
| not_disaster | 309 |
| disaster / warning | 22 |
| disaster / damage | 22 |
| disaster / resource_available | 14 |
| disaster / rescue_needed | 9 |
| **total** | **1,131** |

Full breakdown, per-split counts, and script mix: `data/final/README.md`.

## Honest limitations

- **`other` dominates the disaster class.** YouTube comment sections skew
  heavily toward reaction, grief, and political commentary — not
  structured triage content. Real rescue coordination happens on
  Facebook/WhatsApp in the moment, not in YouTube comments days later.
  This is a genuine finding about the platform, not a labeling artifact.
- **Rare classes are too small to train a 5-way classifier reliably**
  (`rescue_needed` has 9 rows total). The realistic task is a binary
  disaster/not_disaster classifier, with the sub_label distribution
  reported as a descriptive finding rather than a trained target.
- **Single annotator, single event cluster.** No inter-annotator
  agreement was measured, and the dataset covers one flood event —
  generalization to other disasters or annotators is untested.
- **English is capped at ~20% of the sample by design**, to keep the
  dataset centered on Nepali-script and code-mixed text, which is the
  actual research gap this project targets.

## Status

- [x] Scraper, sampler, annotation guidelines
- [x] 1,131 rows labeled, reviewed, and split
- [ ] Baseline fine-tuned classifier (NepaliBERT / RoBERTa) + per-script F1
- [ ] Location NER + geocoding demo
- [ ] Public release with license and dataset card finalized

## Setup

```bash
pip install -r requirements.txt
```

Scraping requires a YouTube Data API v3 key in a `.env` file — see the
docstring at the top of `scripts/01_scrape.py` for the exact variables.

## Repository structure

```
scripts/          numbered pipeline: scrape -> sample -> split -> (train)
docs/             annotation guidelines
data/raw/         scraped corpus (not tracked in git)
data/interim/     working annotation file (not tracked in git)
data/final/       released dataset: train/dev/test + dataset README
```

## License

TBD before public release — likely CC BY 4.0 for the dataset, MIT for code.

## Acknowledgments

Built as part of ongoing disaster-response NLP work for Nepal, informed by
gaps identified in Nepal's existing disaster information systems (BIPAD,
NDRRMA) and by prior open-source civic response tools such as the
Rasuwa–Bhotekoshi Flood Rescue Portal.