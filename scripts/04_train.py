"""Fine-tune NepaliBERT / RoBERTa on data/final/{train,dev}.csv. Intended to run on Colab."""

import argparse
from pathlib import Path

import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

FINAL_DIR = Path(__file__).resolve().parent.parent / "data" / "final"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Rajan/NepaliBERT")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", default=MODELS_DIR)
    args = parser.parse_args()

    train_df = pd.read_csv(FINAL_DIR / "train.csv")
    dev_df = pd.read_csv(FINAL_DIR / "dev.csv")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    labels = sorted(train_df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}

    raise NotImplementedError("wire up Dataset objects, Trainer, and save to out_dir")


if __name__ == "__main__":
    main()
