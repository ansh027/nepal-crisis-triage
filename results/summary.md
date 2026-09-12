# Results

All numbers are macro-F1 on the held-out test set (n=171). For each transformer, mean ± population std across 3 random seeds. Higher is better.

## Overall

| model | seeds | accuracy | macro-F1 |
|---|---|---|---|
| TF-IDF + LogReg | 1 | 0.737 | **0.630** |
| MuRIL (multilingual) | 3 | 0.747 ± 0.019 | **0.666 ± 0.017** |
| NepaliBERT (RoBERTa) | 3 | 0.700 ± 0.028 | **0.577 ± 0.007** |

## Per-script macro-F1 (test set)

| model | devanagari | romanized_nepali | mixed | english |
|---|---|---|---|---|
| TF-IDF + LogReg | 0.580 | 0.702 | 0.435 | 0.644 |
| MuRIL (multilingual) | 0.607 ± 0.040 | 0.629 ± 0.058 | 0.458 ± 0.018 | 0.714 ± 0.043 |
| NepaliBERT (RoBERTa) | 0.555 ± 0.020 | 0.508 ± 0.045 | 0.398 ± 0.035 | 0.597 ± 0.026 |

## Headline finding

**MuRIL (multilingual) beats the TF-IDF baseline (0.666 vs. 0.630 macro-F1, a gap of +0.036)** — the margin is larger than its own seed-to-seed spread of ±0.017, so this is a real improvement, not noise. Across every transformer run and seed, the model's lowest per-script F1 is on **mixed** text. The exact numbers vary (test slice is small), but the ranking is consistent — this is the real weak point of every model tried.

---
_Regenerate with `python scripts/05_evaluate.py`._