"""
Distill Claude's compression judgment into per-token keep/drop labels.

For each (context, query) we show Claude the context as a numbered list of
sentences and ask which sentences are *needed* to answer the query. Those become
the keep=1 sentences; everything else is drop=0. We emit word-level labels
(propagated to sub-word tokens at train time) — this is the LLMLingua-2 setup,
with Claude as the teacher.

Output: data/distill_train.jsonl, data/distill_val.jsonl
Each line: {"query": str, "words": [str, ...], "labels": [0/1, ...]}

Run:
    export ANTHROPIC_API_KEY=sk-...
    ./.venv/bin/python distill.py --n 240 --teacher claude-haiku-4-5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import anthropic

import tokenc as tc

DATA_DIR = Path(__file__).resolve().parent / "data"
TEACHER_DEFAULT = "claude-haiku-4-5"   # cheap+fast; pass --teacher claude-sonnet-4-6 for sharper labels

_TEACHER_SYS = (
    "You are a context compressor. You are given a QUERY and a numbered list of "
    "SENTENCES. Choose the minimal set of sentences strictly required to answer "
    "the QUERY. Return ONLY a JSON array of integer indices (e.g. [2,5]); no prose."
)


def teacher_select(client, sentences: list[str], query: str, model: str) -> set[int]:
    listing = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
    user = f"QUERY: {query}\n\nSENTENCES:\n{listing}"
    key = f"teacher::{model}::{_TEACHER_SYS}::{user}"
    hit = tc._CACHE.get(key)
    if hit is not None:
        raw = hit["raw"]
    else:
        resp = client.messages.create(
            model=model, max_tokens=200, system=_TEACHER_SYS,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        tc._CACHE.set(key, {"raw": raw})
    m = re.search(r"\[[\d,\s]*\]", raw)
    if not m:
        return set()
    try:
        idxs = json.loads(m.group(0))
    except json.JSONDecodeError:
        return set()
    return {int(i) for i in idxs if 0 <= int(i) < len(sentences)}


def build_examples(n: int, seed: int):
    """Varied (context, query) pairs for training — mixes lexical & semantic
    modes and doc counts, with seeds far from the eval seeds so we never train on
    test items. The semantic half is what teaches the student to beat BM25."""
    r = random.Random(seed)
    out, i = [], 0
    while len(out) < n:
        nd = r.choice([4, 6, 8, 10])
        nf = r.choice([2, 3, 4, 5])
        mode = "semantic" if i % 2 == 0 else "lexical"
        bench = tc.make_benchmark(n_examples=8, n_docs=nd, n_filler=nf,
                                  seed=1000 + i, mode=mode)
        for ex in bench:
            out.append(ex)
            if len(out) >= n:
                break
        i += 1
    return out


def pseudo_select(sentences: list[str], query: str, coverage: float = 0.4) -> set[int]:
    """Offline pseudo-teacher (BM25 top-k to a coverage budget). Lets us validate
    the training pipeline end-to-end with no API key. NOT used for the real run."""
    tokd = [tc.tokenize(s) for s in sentences]
    bm25 = tc.BM25(tokd)
    q = tc.tokenize(query)
    order = sorted(range(len(sentences)), key=lambda i: bm25.score(q, i), reverse=True)
    budget = max(1, int(round(sum(tc.estimate_tokens(s) for s in sentences) * coverage)))
    kept, used = set(), 0
    for i in order:
        t = tc.estimate_tokens(sentences[i])
        if kept and used + t > budget:
            continue
        kept.add(i); used += t
        if used >= budget:
            break
    return kept


def to_labeled_row(ex, kept_idx: set[int]):
    sents = tc.split_sentences(ex.context)
    words, labels = [], []
    for i, s in enumerate(sents):
        keep = 1 if i in kept_idx else 0
        for w in s.split():
            words.append(w)
            labels.append(keep)
    return {"query": ex.question, "words": words, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240, help="number of training examples")
    ap.add_argument("--teacher", default=TEACHER_DEFAULT)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--offline", action="store_true",
                    help="use BM25 pseudo-labels (no API) to validate the pipeline")
    args = ap.parse_args()

    client = None
    if not args.offline:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Set ANTHROPIC_API_KEY first:  export ANTHROPIC_API_KEY=sk-...  "
                             "(or pass --offline to dry-run with BM25 pseudo-labels)")
        client = anthropic.Anthropic()

    DATA_DIR.mkdir(exist_ok=True)
    examples = build_examples(args.n, args.seed)

    rows, kept_tot, tok_tot = [], 0, 0
    for j, ex in enumerate(examples):
        sents = tc.split_sentences(ex.context)
        if args.offline:
            kept = pseudo_select(sents, ex.question)
        else:
            kept = teacher_select(client, sents, ex.question, args.teacher)
        # Safety net: the teacher should keep the sentence holding the gold value.
        for i, s in enumerate(sents):
            if tc._norm(ex.gold) in tc._norm(s):
                kept.add(i)
        row = to_labeled_row(ex, kept)
        rows.append(row)
        kept_tot += sum(row["labels"])
        tok_tot += len(row["labels"])
        if (j + 1) % 20 == 0:
            print(f"  labeled {j+1}/{len(examples)} "
                  f"(keep-rate so far {kept_tot/max(1,tok_tot)*100:.0f}%)")

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]

    (DATA_DIR / "distill_train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train))
    (DATA_DIR / "distill_val.jsonl").write_text(
        "\n".join(json.dumps(r) for r in val))
    print(f"\nWrote {len(train)} train / {len(val)} val rows to {DATA_DIR}")
    print(f"Overall teacher keep-rate: {kept_tot/max(1,tok_tot)*100:.1f}% "
          f"(this is the compression target the student learns)")


if __name__ == "__main__":
    main()
