# Nepali Crisis-Text Triage Dataset

1131 hand-labeled comments about the 2026 Nepal floods and landslides (Rasuwa, Bhotekoshi, Melamchi, Kathmandu Valley and related events), collected Aug–Sep 2026 from public YouTube comments.

## Files

| file | rows |
|---|---|
| train.csv | 791 |
| dev.csv | 169 |
| test.csv | 171 |

## Labels

`label`: `disaster` (text is about the event) or `not_disaster`.
`sub_label` (disaster rows only): `rescue_needed`, `damage`, `resource_available`, `warning`, `other`.
Full definitions and edge-case rulings: `docs/annotation_guidelines.md`.

## Class distribution

| class | total | train | dev | test |
|---|---|---|---|---|
| disaster/damage | 22 | 15 | 3 | 4 |
| disaster/other | 755 | 528 | 113 | 114 |
| disaster/rescue_needed | 9 | 6 | 2 | 1 |
| disaster/resource_available | 14 | 10 | 2 | 2 |
| disaster/warning | 22 | 16 | 3 | 3 |
| not_disaster | 309 | 216 | 46 | 47 |

## Script mix

| script | rows |
|---|---|
| devanagari | 412 |
| english | 360 |
| romanized_nepali | 268 |
| mixed | 91 |

## Collection and privacy

- Source: YouTube Data API v3 comment threads on videos matched by Nepali and English flood/landslide queries.
- Sampling: all Nepali-script rows (Devanagari, Romanized, mixed) kept; English capped at ~20%; max 25 comments per video; tribute-song videos excluded.
- PII removed before storage: phone numbers, emails, @handles, URLs replaced with placeholder tokens. Author IDs are one-way hashed.
- Labeled by a single annotator following the written guidelines; ambiguous rows were flagged and resolved in a second pass.

## Known limitations

- `disaster/other` dominates: YouTube comment sections skew toward reaction, grief and political commentary. Actionable triage classes (rescue_needed, damage, resource_available, warning) are rare here because real-time coordination happens on other platforms.
- Single-annotator labels; no inter-annotator agreement measured.
- Single event cluster (Aug–Sep 2026); generalisation to other disasters is untested.

## License

TODO — choose before release (CC BY 4.0 is a reasonable default).
