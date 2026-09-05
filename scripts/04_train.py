#!/usr/bin/env python3
"""
04_train.py — train a disaster/not_disaster classifier and report per-script F1.

Two backends:
  --model tfidf                 TF-IDF + logistic regression. Runs on CPU in
                                seconds. Run this FIRST — it's the floor every
                                transformer has to beat.
  --model <huggingface id>      fine-tune a transformer. Needs a GPU (Colab T4
                                is fine). Verified IDs:
                                  IRIISNEPAL/RoBERTa_Nepali_110M   Nepali-only
                                  IRIISNEPAL/BERT_Nepali_110M      Nepali-only
                                  google/muril-base-cased          multilingual, trained
                                                                   on transliterated Indic
                                  xlm-roberta-base                 multilingual

Tasks:
  --task binary   disaster vs not_disaster            (default)
  --task three    not_disaster / other / actionable   (actionable = damage +
                  warning + rescue_needed + resource_available, ~67 rows)

Outputs (results/<run_name>/):
  metrics.json      overall + per-script precision/recall/F1 on test
  predictions.csv   test rows with gold and predicted labels
  model/            saved transformer (skipped for tfidf)

Usage:
  pip install -r requirements.txt   # + transformers torch datasets for HF models
  python scripts/04_train.py --model tfidf
  python scripts/04_train.py --model google/muril-base-cased --epochs 4
  python scripts/04_train.py --model IRIISNEPAL/RoBERTa_Nepali_110M --task three

Colab:
  !git clone https://github.com/ansh027/nepal-crisis-triage.git
  %cd nepal-crisis-triage
  !pip install -q transformers datasets accelerate scikit-learn
  !python scripts/04_train.py --model google/muril-base-cased
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "final"
RESULTS = ROOT / "results"

ACTIONABLE = {"damage", "warning", "rescue_needed", "resource_available"}
SCRIPTS = ["devanagari", "romanized_nepali", "mixed", "english", "other"]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_split(name: str) -> list[dict]:
    path = DATA / f"{name}.csv"
    if not path.exists():
        sys.exit(f"missing {path} — run scripts/03_split.py first")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("id") or "").strip()]


def target(row: dict, task: str) -> str:
    if task == "binary":
        return row["label"]
    # three-way
    if row["label"] == "not_disaster":
        return "not_disaster"
    return "actionable" if row["sub_label"] in ACTIONABLE else "other"


def prepare(task: str):
    splits = {s: load_split(s) for s in ("train", "dev", "test")}
    labels = sorted({target(r, task) for r in splits["train"]})
    label2id = {l: i for i, l in enumerate(labels)}
    for rows in splits.values():
        for r in rows:
            r["y"] = label2id[target(r, task)]
    return splits, labels, label2id


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def evaluate(rows: list[dict], preds: list[int], labels: list[str]) -> dict:
    gold = [r["y"] for r in rows]
    p, r_, f, s = precision_recall_fscore_support(gold, preds, labels=range(len(labels)),
                                                  zero_division=0)
    out = {
        "n": len(rows),
        "accuracy": float(np.mean(np.array(gold) == np.array(preds))),
        "macro_f1": float(f1_score(gold, preds, average="macro", zero_division=0)),
        "per_class": {labels[i]: {"precision": float(p[i]), "recall": float(r_[i]),
                                  "f1": float(f[i]), "support": int(s[i])}
                      for i in range(len(labels))},
        "confusion_matrix": confusion_matrix(gold, preds, labels=range(len(labels))).tolist(),
        "per_script": {},
    }
    for sc in SCRIPTS:
        idx = [i for i, r in enumerate(rows) if r["script"] == sc]
        if not idx:
            continue
        g = [gold[i] for i in idx]
        pr = [preds[i] for i in idx]
        out["per_script"][sc] = {
            "n": len(idx),
            "accuracy": float(np.mean(np.array(g) == np.array(pr))),
            "macro_f1": float(f1_score(g, pr, average="macro", zero_division=0)),
        }
    return out


def print_report(m: dict, labels: list[str]) -> None:
    print(f"\nTEST  n={m['n']}  acc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}")
    print(f"{'class':20s} {'P':>6s} {'R':>6s} {'F1':>6s} {'n':>5s}")
    for l in labels:
        c = m["per_class"][l]
        print(f"{l:20s} {c['precision']:6.3f} {c['recall']:6.3f} {c['f1']:6.3f} {c['support']:5d}")
    print(f"\n{'script':20s} {'n':>5s} {'acc':>6s} {'macroF1':>8s}")
    for sc, v in m["per_script"].items():
        print(f"{sc:20s} {v['n']:5d} {v['accuracy']:6.3f} {v['macro_f1']:8.3f}")


# --------------------------------------------------------------------------
# backend 1: tf-idf + logistic regression
# --------------------------------------------------------------------------

def run_tfidf(splits, labels, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union

    # word n-grams catch vocabulary; char n-grams catch Romanized spelling
    # variation (badhi / baadi / badi) and Devanagari morphology.
    vec = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    )
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                             random_state=seed)
    pipe = make_pipeline(vec, clf)

    xtr = [r["text"] for r in splits["train"]]
    ytr = [r["y"] for r in splits["train"]]
    pipe.fit(xtr, ytr)

    dev_pred = pipe.predict([r["text"] for r in splits["dev"]])
    dev_f1 = f1_score([r["y"] for r in splits["dev"]], dev_pred, average="macro")
    print(f"dev macro-F1: {dev_f1:.3f}")

    test_pred = pipe.predict([r["text"] for r in splits["test"]]).tolist()
    return test_pred, {"dev_macro_f1": float(dev_f1)}, None


# --------------------------------------------------------------------------
# backend 2: huggingface transformer
# --------------------------------------------------------------------------

def run_transformer(splits, labels, args):
    try:
        import torch
        from datasets import Dataset
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  DataCollatorWithPadding, Trainer, TrainingArguments)
    except ImportError:
        sys.exit("pip install transformers datasets accelerate torch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found — this will be very slow. Use Colab.")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels))

    def to_ds(rows):
        return Dataset.from_dict({"text": [r["text"] for r in rows],
                                  "labels": [r["y"] for r in rows]})

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds = {k: to_ds(v).map(tokenize, batched=True, remove_columns=["text"])
          for k, v in splits.items()}

    # class weights to counter the imbalance (not_disaster ~27%, actionable ~6%)
    counts = Counter(r["y"] for r in splits["train"])
    total = sum(counts.values())
    weights = torch.tensor([total / (len(labels) * counts[i]) for i in range(len(labels))],
                           dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            y = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                out.logits, y, weight=weights.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def compute_metrics(ev):
        preds = ev.predictions.argmax(-1)
        return {"macro_f1": f1_score(ev.label_ids, preds, average="macro")}

    out_dir = RESULTS / args.run_name
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=20,
        seed=args.seed,
        fp16=(device == "cuda"),
        report_to="none",
    )
    trainer = WeightedTrainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["dev"],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    dev_metrics = trainer.evaluate(ds["dev"])
    test_logits = trainer.predict(ds["test"]).predictions
    test_pred = test_logits.argmax(-1).tolist()

    trainer.save_model(str(out_dir / "model"))
    tok.save_pretrained(str(out_dir / "model"))
    return test_pred, {"dev_macro_f1": float(dev_metrics["eval_macro_f1"])}, out_dir


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="'tfidf' or a HuggingFace model id")
    ap.add_argument("--task", choices=["binary", "three"], default="binary")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.run_name = args.run_name or f"{args.task}_{args.model.replace('/', '__')}_s{args.seed}"

    splits, labels, label2id = prepare(args.task)
    print(f"task={args.task}  model={args.model}  labels={labels}")
    print("train:", dict(Counter(target(r, args.task) for r in splits['train'])))

    t0 = time.time()
    if args.model == "tfidf":
        test_pred, extra, _ = run_tfidf(splits, labels, args.seed)
    else:
        test_pred, extra, _ = run_transformer(splits, labels, args)
    elapsed = time.time() - t0

    m = evaluate(splits["test"], test_pred, labels)
    m.update(extra)
    m.update({"model": args.model, "task": args.task, "seed": args.seed,
              "labels": labels, "train_seconds": round(elapsed, 1)})
    print_report(m, labels)

    out_dir = RESULTS / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "script", "gold", "pred", "text"])
        for r, p in zip(splits["test"], test_pred):
            w.writerow([r["id"], r["script"], labels[r["y"]], labels[p], r["text"]])

    print(f"\n-> {out_dir / 'metrics.json'}\n-> {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()