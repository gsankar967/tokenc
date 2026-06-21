"""
Train the query-aware keep/drop token classifier (LLMLingua-2 recipe).

Model-agnostic via AutoModelForTokenClassification:
  * default backbone: distilbert-base-uncased  (small bidirectional encoder — the
    right architecture for keep/drop; each token sees both directions)
  * stronger options: bert-base-uncased, microsoft/deberta-v3-small,
    answerdotai/ModernBERT-base  (just pass --backbone)

Input format per example:  "Query: <q> Context: <w1 w2 ...>"
  - query/prefix tokens get label -100 (ignored in loss)
  - context word tokens get the keep/drop label (first sub-token labeled)

Run:
    # quick pipeline+timing check (offline pseudo-labels)
    ./.venv/bin/python distill.py --offline --n 80
    ./.venv/bin/python train_compressor.py --smoke

    # real run after distilling from Claude
    ./.venv/bin/python train_compressor.py --backbone distilbert-base-uncased --epochs 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DEFAULT = Path(__file__).resolve().parent / "compressor_model"


def load_rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def make_dataset(rows, tokenizer, max_len: int):
    def encode(row):
        prefix = ["Query:"] + row["query"].split() + ["Context:"]
        words = prefix + row["words"]
        wlabels = [-100] * len(prefix) + row["labels"]
        enc = tokenizer(
            words, is_split_into_words=True, truncation=True, max_length=max_len,
        )
        word_ids = enc.word_ids()
        labels, prev = [], None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev:
                labels.append(wlabels[wid])      # label first sub-token of a word
            else:
                labels.append(-100)              # ignore continuation sub-tokens
            prev = wid
        enc["labels"] = labels
        return enc

    return Dataset.from_list([encode(r) for r in rows])


def keep_f1(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    p, y = preds[mask], labels[mask]
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = float((p == y).mean()) if len(y) else 0.0
    return {"keep_precision": prec, "keep_recall": rec, "keep_f1": f1, "token_acc": acc}


class WeightedTrainer(Trainer):
    """Token-classification with a class-weighted loss — the KEEP class is rare
    (~2%, since usually one sentence answers the query), so we up-weight it."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = None if self._class_weights is None else self._class_weights.to(logits.device)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1),
            weight=w, ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="distilbert-base-uncased")
    ap.add_argument("--keep-weight", type=float, default=8.0,
                    help="loss weight for the rare KEEP class")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on a tiny subset to validate pipeline + timing")
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Backbone: {args.backbone} | device: {device}")

    train_rows = load_rows(DATA_DIR / "distill_train.jsonl")
    val_rows = load_rows(DATA_DIR / "distill_val.jsonl")
    if args.smoke:
        train_rows, val_rows = train_rows[:16], val_rows[:8]
        args.epochs = 1.0

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:                       # Qwen et al. have no pad token
        tok.pad_token = tok.eos_token
    model = AutoModelForTokenClassification.from_pretrained(
        args.backbone, num_labels=2,
        id2label={0: "DROP", 1: "KEEP"}, label2id={"DROP": 0, "KEEP": 1},
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    ds_train = make_dataset(train_rows, tok, args.max_len)
    ds_val = make_dataset(val_rows, tok, args.max_len)
    collator = DataCollatorForTokenClassification(tok)

    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        fp16=False, bf16=False,
    )
    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=collator, compute_metrics=keep_f1,
        class_weights=torch.tensor([1.0, args.keep_weight]),
    )

    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    metrics = trainer.evaluate()
    print(f"\nTrained in {dt:.1f}s on {len(train_rows)} examples ({device}).")
    print(f"Val keep-F1 {metrics['eval_keep_f1']:.3f} | "
          f"precision {metrics['eval_keep_precision']:.3f} | "
          f"recall {metrics['eval_keep_recall']:.3f} | "
          f"token-acc {metrics['eval_token_acc']:.3f}")

    if not args.smoke:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        report = {
            "backbone": args.backbone, "device": device,
            "train_examples": len(train_rows), "train_seconds": round(dt, 1),
            "keep_f1": metrics["eval_keep_f1"],
            "keep_precision": metrics["eval_keep_precision"],
            "keep_recall": metrics["eval_keep_recall"],
            "token_acc": metrics["eval_token_acc"],
        }
        (Path(args.out) / "metrics.json").write_text(json.dumps(report, indent=2))
        print(f"Saved compressor + metrics to {args.out}")
    else:
        per_ex = dt / max(1, len(train_rows))
        print(f"\n[smoke] ~{per_ex:.2f}s/example/epoch on {device}. "
              f"For 240 ex × 3 epochs ≈ {per_ex*240*3/60:.1f} min.")


if __name__ == "__main__":
    main()
