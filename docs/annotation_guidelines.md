# Annotation Guidelines — Nepali Crisis-Text Triage

One page. Read it once before you start, keep it open in a tab while you
label, don't re-derive rules mid-session — write new edge cases down here
instead (see "Edge case log" at the bottom).

## What you're labeling

Each row is one comment/post scraped from YouTube (and optionally Reddit/
Facebook) about the 2026 Nepal floods and landslides. You're deciding what
kind of content it is, the way a triage system would have to decide in
real time.

## Step 1 — `label` (every row gets this one)

| Value | Meaning |
|---|---|
| `disaster` | The text is *about* the flood/landslide event — a report, reaction, request, warning, or factual claim connected to it. |
| `not_disaster` | Off-topic: song praise, unrelated politics, spam, generic "nice video," jokes with no disaster content. |

**Default to `disaster` if genuinely unsure and the text at least references
the event** (e.g. "so sad 😢" on a flood video is borderline — see edge
case rule below). Default to `not_disaster` only when the text has nothing
to do with the flood at all.

## Step 2 — `sub_label` (only if `label` = disaster)

Pick exactly one. If two seem to fit, pick the one that would matter more
to a responder deciding where to send help.

| Value | Meaning | Example shape |
|---|---|---|
| `rescue_needed` | Someone trapped, missing, or in active danger; a call for help for a specific person/place | "मामा अझै भेटिएका छैनन्" (uncle still not found) |
| `damage` | Reports property/infrastructure/casualty damage, no active rescue ask | "पुल बगायो", "140 dead" |
| `resource_available` | Offering help, aid, shelter, donations, volunteer coordination | "हामी Rasuwa मा राहत लैजाँदैछौं" |
| `warning` | Forecast, alert, or caution about ongoing/future risk | "अझै खतरा छ", flood warning links |
| `other` | Clearly disaster-related but doesn't fit the above — political commentary on the response, blame, speculation about cause, general grief | "सरकार के गर्दैछ?", "लासमाथि राजनीति" |

## Step 3 — `location_mentioned` (optional, fill if present)

Free text. Copy the place name exactly as written, even if Romanized
oddly (e.g. `Rasuwa`, `भोटेकोशी`, `Melamchi`). Leave blank if no place is
named. Don't guess or geocode by hand — that's a later automated step.

## Step 4 — `notes` (optional)

Anything you want future-you to know: ambiguous call, sarcasm you're
not sure about, possible misinformation, code-mixing you found interesting.
Not required per row.

## Hard rules

- **One pass, don't overthink.** If a row takes more than ~10 seconds,
  make your best call, add a one-word note, move on. You can revisit
  flagged rows at the end.
- **Judge the text alone**, not the video it came from — a comment on an
  international news video can still be `rescue_needed` if it quotes
  someone's situation, and a comment on a clearly disaster-related video
  can still be `not_disaster` if it's just "great reporting."
- **Sarcasm/dark humor about the disaster** (common in the political
  commentary) still counts as `disaster` / usually `other` — it's
  disaster-adjacent discourse, not off-topic.
- **Don't infer language proficiency or identity** about commenters.
  You're labeling text content only.

## Edge case rulings (add to this list as you hit new ones — don't relitigate)

- Generic sympathy with no other content ("😢😢", "so sad", "RIP") →
  `disaster` / `other` if clearly on a disaster-context post; if the video
  itself is ambiguous, use `not_disaster`.
- Short blessings/expressions of hope directly on a disaster video (not a
  copy-paste religious template) → `disaster` / `other`. Distinguish from
  generic prayer-chain templates (which stay `not_disaster`) by whether the
  text could equally appear on any random video, or whether it responds to
  this specific situation.
- News-anchor-style factual recap comments (people pasting death tolls
  from articles) → `disaster` / `damage`.
- Comments blaming a political figure for the response → `disaster` /
  `other`.
- Commentary praising or criticizing a video/commentator's explanation of
  the disaster's cause (technical/analytical accuracy) → `disaster` /
  `other`, same as commentary on the government's response — both are
  disaster-adjacent discourse.
- Comments that are just a username tag or emoji-only → should already be
  filtered by length, but if one slips through, `not_disaster`.

## Labeling in the CSV

Open `data/interim/to_annotate.csv` in Excel/Google Sheets/LibreOffice.
Fill `label`, `sub_label`, `location_mentioned`, `notes` per row. Save as
CSV (not xlsx) when done, or periodically — don't lose an afternoon's work
to a forgotten save.

When finished (or when you stop for the day), that file becomes the input
to `scripts/03_split.py`, which turns it into train/dev/test sets.
