"""Evaluate a trained model on data/final/test.csv: macro-F1 overall + per-script slice, confusion matrix."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

FINAL_DIR = Path(__file__).resolve().parent.parent / "data" / "final"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-path", default=FINAL_DIR / "test.csv")
    parser.add_argument("--out", default=RESULTS_DIR / "metrics.json")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_path)

    raise NotImplementedError(
        "load model from model_dir, run predictions, compute overall + per-script macro-F1, "
        "save confusion matrix to results/figures/, dump metrics to --out"
    )


if __name__ == "__main__":
    main()
