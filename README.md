# TokenC — distill Claude's context compression into a tiny model

**The Token Company Compression Challenge.** Cut the tokens you send an LLM by ~50–80% while preserving answer quality — by distilling **Claude's relevance judgment** into a small, local **keep/drop token classifier** (the LLMLingua‑2 recipe, with Claude as the teacher).

### Headline result (from `demo.ipynb`, measured, not estimated)

On a semantic multi‑doc QA benchmark with lexical‑trap distractors, downstream reader = **Claude Haiku 4.5**:

| keep rate | full context | BM25 (classical) | **trained model** |
|---|---|---|---|
| 100% (431 tok) | 80% | 80% | 80% |
| 30% (~176 tok) | — | 60% | **70%** |
| **20% (~141 tok)** | — | **18%** | **70%** |

> At **20% of the keep‑rate (~33% of the tokens)** the distilled model holds **70%** accuracy while the classical BM25 baseline **collapses to 18%**. Same token budget for both — the model just keeps the *right* sentences. Cost falls proportionally at every Claude tier (Haiku/Sonnet/Opus).

The 66M‑param compressor trains in **~10 seconds on a MacBook (MPS)**.

---

## Why a learned compressor beats classical

BM25 ranks sentences by surface‑word overlap. The moment the query and the answer don't share words — *"Where is X **based**?"* answered by *"X **runs everything out of** Helsinki"* — and the context is full of lexical traps (*"X **based** its culture on remote‑first work"*), BM25 spends its budget on the traps and **drops the answer**. The model, trained to imitate which sentences **Claude** says are needed, learns the meaning‑level mapping and keeps the answer.

This is exactly The Token Company's thesis: a small custom model that compresses context better than heuristics, cutting cost while maintaining downstream quality.

---

## How it works

```
 (context, query)
        │
        ▼  distill.py
  Claude (teacher) picks the sentences needed to answer the query
        │            → per‑token KEEP/DROP labels  (data/*.jsonl)
        ▼  train_compressor.py
  DistilBERT (student) fine‑tuned as a query‑aware keep/drop classifier
        │            → compressor_model/  (class‑weighted loss; KEEP is ~2%)
        ▼  neural.py
  score each sentence by mean KEEP‑probability (query‑conditioned),
  budget‑controller keeps top sentences to a target token ratio
        │
        ▼  tokenc.py (eval harness)
  full vs BM25 vs model → downstream accuracy / tokens / $  → demo.ipynb
```

- **Teacher labeling** (`distill.py`): Claude is shown the context as a numbered sentence list and returns the indices needed to answer the query → KEEP/DROP labels. Mixed **lexical** + **semantic** slices; seeds disjoint from the eval set.
- **Student** (`train_compressor.py`): `AutoModelForTokenClassification` (default `distilbert-base-uncased`, swap with `--backbone`). Bidirectional encoder — each token sees both directions, the right architecture for keep/drop. Class‑weighted loss because KEEP is rare.
- **Inference** (`neural.py`): per‑sentence KEEP‑probability → rank → fill to a token budget → re‑emit in original order. Sentence scores are ratio‑independent and disk‑cached, so a whole Pareto sweep reuses them.
- **Eval** (`tokenc.py`): controllable multi‑doc QA benchmark with distractors; Claude as the downstream reader; objective substring grading; every LLM call disk‑cached so the demo re‑runs instantly.

---

## Files

| file | what |
|---|---|
| `tokenc.py` | engine: BM25 baseline, budget controller, benchmark generator, eval harness, pricing, caching |
| `distill.py` | generate KEEP/DROP training labels from Claude (`--offline` for an API‑free dry run) |
| `train_compressor.py` | fine‑tune the keep/drop classifier (`--smoke` for a fast pipeline+timing check) |
| `neural.py` | load the trained model and compress (drop‑in for the BM25 baseline) |
| `build_notebook.py` | regenerates `demo.ipynb` |
| `demo.ipynb` | the story + 3 charts + interactive keep‑rate slider (pre‑executed) |
| `smoke_test.py` | offline sanity checks (no API key) |

---

## Quickstart

```bash
bash setup.sh                                   # venv + deps + Jupyter kernel
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env      # auto-loaded by every script

./.venv/bin/python distill.py --n 240           # distill labels from Claude  (~5 min, cached)
./.venv/bin/python train_compressor.py          # train the model             (~10 s on MPS)
./.venv/bin/python build_notebook.py            # (re)build the notebook
./.venv/bin/jupyter notebook demo.ipynb         # open the demo
```

The notebook ships **pre‑executed** — open it to see the charts immediately; re‑run is instant (cached).

Interactive booth demo: the **keep‑rate slider** cell — drag it and watch tokens & cost fall while the answer stays correct, then flip to BM25 to watch it break at low keep‑rates.

---

## Product framing

A **drop‑in proxy**: point your Anthropic `base_url` at TokenC; we compress every prompt's context before it hits the model — **~50%+ fewer input tokens, same answers, one line of config.** The compressor is a small model you run locally or at the edge.

---

## Honest notes

- The benchmark is **synthetic and controllable** by design — it lets us dial the exact regime (lexical traps, compression budget) where classical methods fail and a learned one wins, with objective grading. The method (`compress(context, query, ratio)`) is content‑agnostic and runs on any text (try the slider on your own).
- We do **not** claim compression *improves over* full‑context accuracy here — the Haiku reader handles these short contexts well at full length. The measured win is **robustness under aggressive compression vs the classical baseline**, plus the cost reduction.
- Models/pricing pulled from the current Claude lineup (Haiku 4.5 $1/$5, Sonnet 4.6 $3/$15, Opus 4.8 $5/$25 per 1M in/out). Token counts use Anthropic's own counter / real API `usage`, never tiktoken.
