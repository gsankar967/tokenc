"""
TokenC — query-aware context compression for LLMs.

The pitch: send fewer tokens to the model while preserving (and often improving)
answer quality. This module is the engine + an eval harness that *proves* the
claim with a token-reduction-vs-downstream-quality curve on a controllable
multi-doc QA benchmark with distractors.

Design notes
------------
* The default compressor is **query-aware extractive selection** (a compact
  BM25 ranker). It is fast, dependency-free, deterministic, and — crucially —
  removes distractor/irrelevant text, which is exactly what makes the downstream
  model *more* accurate on long noisy contexts ("lost in the middle").
* An optional **LLM densifier** (Haiku) is included as a second strategy.
* All LLM calls are cached to disk so the notebook re-runs instantly and cheaply
  during a live demo.

Nothing here uses tiktoken. Token counts come from Anthropic's own counter / the
real `usage` returned by the Messages API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional


def _pmap(fn, items, workers: int = 8):
    """Thread-pooled map that preserves order (LLM calls are IO-bound)."""
    if workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))

# ----------------------------------------------------------------------------
# Models & pricing  (USD per 1,000,000 tokens)  — current Claude lineup.
# ----------------------------------------------------------------------------
PRICING = {
    "claude-haiku-4-5":  {"in": 1.0,  "out": 5.0},
    "claude-sonnet-4-6": {"in": 3.0,  "out": 15.0},
    "claude-opus-4-8":   {"in": 5.0,  "out": 25.0},
}
DOWNSTREAM_MODEL = "claude-haiku-4-5"   # the "reader" the compressor feeds
COMPRESSOR_MODEL = "claude-haiku-4-5"   # used only by the LLM densifier strategy

CACHE_DIR = Path(__file__).resolve().parent / ".tokenc_cache"


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency). Lets every script + the notebook pick
    up ANTHROPIC_API_KEY from a local, gitignored .env without hardcoding it."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    envp = Path(__file__).resolve().parent / ".env"
    if not envp.exists():
        return
    for line in envp.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


# ----------------------------------------------------------------------------
# Tiny disk cache so a live demo never pays twice for the same call.
# ----------------------------------------------------------------------------
class DiskCache:
    def __init__(self, path: Path = CACHE_DIR):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.path / f"{h}.json"

    def get(self, key: str):
        f = self._file(key)
        if f.exists():
            return json.loads(f.read_text())
        return None

    def set(self, key: str, value) -> None:
        self._file(key).write_text(json.dumps(value))


_CACHE = DiskCache()


# ----------------------------------------------------------------------------
# Token counting.
#   * estimate_tokens : instant, offline, monotonic — used for the budget knob.
#   * count_tokens    : exact, via Anthropic's counter — used for headline numbers.
# ----------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Fast offline token estimate. ~chars/4, the standard rough heuristic."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def count_tokens(client, text: str, model: str = DOWNSTREAM_MODEL) -> int:
    """Exact prompt token count via Anthropic's token-counting endpoint (cached)."""
    key = f"count::{model}::{text}"
    hit = _CACHE.get(key)
    if hit is not None:
        return hit["input_tokens"]
    resp = client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    )
    _CACHE.set(key, {"input_tokens": resp.input_tokens})
    return resp.input_tokens


# ----------------------------------------------------------------------------
# Text utilities: sentence splitting + a minimal stemmer for robust matching.
# ----------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = set(
    "a an the of to in on at for and or but is are was were be been being "
    "this that these those with as by from it its their his her our your my "
    "what which who whom whose when where why how do does did has have had "
    "will would can could should may might into about over under than then".split()
)


def split_sentences(text: str) -> list[str]:
    """Split into sentence-ish units, also breaking on hard newlines."""
    units: list[str] = []
    for block in re.split(r"\n{2,}", text.strip()):
        block = block.strip()
        if not block:
            continue
        parts = _SENT_SPLIT.split(block.replace("\n", " "))
        units.extend(p.strip() for p in parts if p.strip())
    return units


def _stem(tok: str) -> str:
    for suf in ("ing", "edly", "ed", "ly", "es", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


# ----------------------------------------------------------------------------
# BM25 — the query-aware ranker at the heart of extractive compression.
# ----------------------------------------------------------------------------
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = max(1, len(docs))
        self.avgdl = (sum(len(d) for d in docs) / self.N) or 1.0
        df: dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        # Standard BM25 idf with +1 smoothing (always positive).
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def score(self, query_terms: list[str], idx: int) -> float:
        doc = self.docs[idx]
        if not doc:
            return 0.0
        freq: dict[str, int] = {}
        for t in doc:
            freq[t] = freq.get(t, 0) + 1
        dl = len(doc)
        s = 0.0
        for t in query_terms:
            if t not in freq:
                continue
            f = freq[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
        return s


# ----------------------------------------------------------------------------
# The compression result object.
# ----------------------------------------------------------------------------
@dataclass
class Compressed:
    text: str
    query: str
    strategy: str
    orig_tokens: int
    new_tokens: int
    n_units_total: int
    n_units_kept: int

    @property
    def ratio(self) -> float:
        return self.new_tokens / self.orig_tokens if self.orig_tokens else 1.0

    @property
    def reduction(self) -> float:
        return 1.0 - self.ratio

    def summary(self) -> str:
        return (
            f"[{self.strategy}] {self.orig_tokens}→{self.new_tokens} tokens "
            f"({self.reduction*100:.0f}% reduction, "
            f"{self.n_units_kept}/{self.n_units_total} units kept)"
        )


# ----------------------------------------------------------------------------
# Strategy 1: query-aware extractive compression (the workhorse).
# ----------------------------------------------------------------------------
def compress_extractive(
    context: str, query: str, target_ratio: float = 0.5, token_fn=estimate_tokens
) -> Compressed:
    """Keep the highest-BM25 sentences (vs the query) until we hit the token
    budget, then re-emit them in original order so the text stays coherent."""
    units = split_sentences(context)
    orig_tokens = token_fn(context)
    if not units:
        return Compressed(context, query, "extractive", orig_tokens, orig_tokens, 0, 0)

    tokd = [tokenize(u) for u in units]
    bm25 = BM25(tokd)
    qterms = tokenize(query)
    scored = sorted(
        range(len(units)), key=lambda i: bm25.score(qterms, i), reverse=True
    )

    budget = max(1, int(round(orig_tokens * target_ratio)))
    kept: set[int] = set()
    used = 0
    for i in scored:
        t = token_fn(units[i])
        if kept and used + t > budget:
            continue
        kept.add(i)
        used += t
        if used >= budget:
            break

    text = " ".join(units[i] for i in sorted(kept))
    return Compressed(
        text, query, "extractive", orig_tokens, token_fn(text), len(units), len(kept)
    )


# ----------------------------------------------------------------------------
# Strategy 2: LLM densifier (Haiku rewrites the pre-filtered context into dense
# query-relevant facts). Optional — costs tokens/latency but compresses harder.
# ----------------------------------------------------------------------------
def densify(client, text: str, query: str, budget_tokens: int,
            model: str = COMPRESSOR_MODEL) -> str:
    """Abstractive last mile: have a small LLM rewrite already-relevant text into
    dense, query-focused facts. Preserves values verbatim; output is cached."""
    sys = (
        "You compress context for a downstream model. Given a QUERY and TEXT, "
        "output only the facts from TEXT needed to answer the QUERY, as terse "
        "bullet points. Preserve names, numbers, and exact values verbatim. "
        f"Stay under roughly {budget_tokens} tokens. Output nothing else."
    )
    user = f"QUERY: {query}\n\nTEXT:\n{text}"
    key = f"densify::{model}::{sys}::{user}"
    hit = _CACHE.get(key)
    if hit is not None:
        return hit["text"]
    resp = client.messages.create(
        model=model, max_tokens=min(2048, budget_tokens + 200),
        system=sys, messages=[{"role": "user", "content": user}],
    )
    out = "".join(b.text for b in resp.content if b.type == "text").strip()
    _CACHE.set(key, {"text": out})
    return out


def compress_llm(client, context, query, target_ratio=0.3,
                 model=COMPRESSOR_MODEL, token_fn=estimate_tokens) -> Compressed:
    """Extract (BM25) then rephrase (densify). The neural extract+rephrase hybrid
    lives in neural.NeuralCompressor.compress_hybrid."""
    orig = token_fn(context)
    pre = compress_extractive(context, query, min(1.0, target_ratio * 2), token_fn)
    text = densify(client, pre.text, query, max(20, int(round(orig * target_ratio))), model)
    return Compressed(text, query, "llm", orig, token_fn(text),
                      pre.n_units_total, pre.n_units_total)


def compress(context, query, target_ratio=0.5, strategy="extractive", client=None,
             token_fn=estimate_tokens) -> Compressed:
    """Dispatcher. `target_ratio` is the fraction of original tokens to keep."""
    if strategy == "extractive":
        return compress_extractive(context, query, target_ratio, token_fn)
    if strategy == "llm":
        assert client is not None, "LLM strategy needs an Anthropic client"
        return compress_llm(client, context, query, target_ratio, token_fn=token_fn)
    raise ValueError(f"unknown strategy: {strategy}")


# ----------------------------------------------------------------------------
# Downstream reader: ask the model to answer using the (compressed) context.
# Returns the answer plus REAL token usage from the API.
# ----------------------------------------------------------------------------
_ANSWER_SYS = (
    "Answer the question using ONLY the provided context. Reply with just the "
    "answer in as few words as possible — no explanation. If the answer is not "
    "in the context, reply exactly: unknown"
)


def ask(client, context: str, question: str, model: str = DOWNSTREAM_MODEL):
    user = f"Context:\n{context}\n\nQuestion: {question}"
    key = f"ask::{model}::{_ANSWER_SYS}::{user}"
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    resp = client.messages.create(
        model=model, max_tokens=64, system=_ANSWER_SYS,
        messages=[{"role": "user", "content": user}],
    )
    out = {
        "answer": "".join(b.text for b in resp.content if b.type == "text").strip(),
        "in_tokens": resp.usage.input_tokens,
        "out_tokens": resp.usage.output_tokens,
    }
    _CACHE.set(key, out)
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def graded_correct(answer: str, gold: str) -> bool:
    return _norm(gold) in _norm(answer)


# ----------------------------------------------------------------------------
# Synthetic multi-doc QA benchmark with distractors.
#
# Each example asks for one entity's attribute. The gold doc (placed in the
# MIDDLE, to trigger lost-in-the-middle) holds the answer; the other docs are
# plausible distractors about *other* entities with the *same* attribute type.
# Fully offline + deterministic. Values are distinctive single tokens for
# objective substring grading.
# ----------------------------------------------------------------------------
_CITIES = ["Helsinki", "Reykjavik", "Wellington", "Montevideo", "Ljubljana",
           "Tallinn", "Brisbane", "Nagoya", "Calgary", "Porto", "Lyon", "Bergen"]
_PRODUCTS = ["Zentyx", "Orbify", "Lumina", "Kestrel", "Vantyr", "Nimbus",
             "Quasar", "Halcyon", "Vextra", "Pyrite", "Solara", "Tessaract"]
_SURNAMES = ["Kovalenko", "Adeyemi", "Nakashima", "Bjornsson", "Okafor",
             "Vasquez", "Lindqvist", "Mahmood", "Petrova", "Caldwell"]
_FIRST = ["Maya", "Tomas", "Aiko", "Ravi", "Lena", "Diego", "Nora", "Soren"]
_PREFIX = ["Aero", "Nova", "Quant", "Helix", "Vertex", "Cobalt", "Lumen",
           "Strato", "Iron", "Delphi", "Onyx", "Pyra", "Zephyr", "Astra"]
_SUFFIX = ["dyne", "works", "labs", "tech", "systems", "core", "ware", "wave"]
_FILLER = [
    "The company operates across several international markets.",
    "It maintains a distributed engineering organization.",
    "Industry analysts have covered its recent strategy shifts.",
    "The firm partners with a range of enterprise customers.",
    "Its supply chain spans multiple regions.",
    "Employees describe a fast-moving internal culture.",
]

_ATTRS = {
    "founding year": lambda r: str(r.randint(1971, 2019)),
    "headquarters city": lambda r: r.choice(_CITIES),
    "flagship product": lambda r: r.choice(_PRODUCTS),
    "chief executive": lambda r: r.choice(_SURNAMES),   # single-token value → clean grading
}

# LEXICAL mode: query + fact share the attribute words → BM25 is near-optimal.
_Q_LEXICAL = "What is {e}'s {attr}?"
_FACT_LEXICAL = {
    "founding year": "{e}'s founding year is {v}.",
    "headquarters city": "{e}'s headquarters city is {v}.",
    "flagship product": "{e}'s flagship product is {v}.",
    "chief executive": "{e}'s chief executive is {v}.",
}

# SEMANTIC mode: query uses synonyms and the fact avoids the query's words, so
# lexical overlap no longer identifies the answer — this is where a model that
# learned Claude's relevance judgment beats BM25.
_Q_SEMANTIC = {
    "founding year": "In which year did {e} begin operations?",
    "headquarters city": "Where is {e} based?",
    "flagship product": "What does {e} mainly sell?",
    "chief executive": "Who leads {e}?",
}
_FACT_SEMANTIC = {
    "founding year": "{e} opened its doors back in {v}.",
    "headquarters city": "{e} runs everything out of {v}.",
    "flagship product": "{e}'s biggest moneymaker is {v}.",
    "chief executive": "{e} is run day to day by {v}.",
}

# Lexical-trap filler for the SEMANTIC slice: every distractor sentence echoes the
# query's keyword ("based", "sell", "leads", "operations/year") but carries NO
# answer value. BM25 ranks these ABOVE the real answer sentence (which avoids the
# query's words) and drops the answer under compression — exactly the failure mode
# a model that learned Claude's meaning-level judgment avoids.
_SOFT_TRAP_FILLER = {
    "founding year": [
        "{e} expanded its operations again this past year.",
        "{e} will begin a new hiring round later this year.",
        "{e} reviews operations at the start of every year.",
        "{e} restructured operations earlier in the year.",
        "{e} plans to begin overseas operations next year.",
    ],
    "headquarters city": [
        "{e} based its latest campaign on customer feedback.",
        "{e} has based its hiring on employee referrals.",
        "{e} based its rebrand on a minimalist philosophy.",
        "{e} keeps its culture based on remote-first work.",
        "{e} based this year's roadmap on user research.",
    ],
    "flagship product": [
        "{e} chose not to sell hardware to the public.",
        "{e} plans to sell merchandise at upcoming events.",
        "{e} did not sell any new lines this quarter.",
        "{e} will sell support contracts to enterprises.",
        "{e} declined to sell its smaller business unit.",
    ],
    "chief executive": [
        "{e} leads its sector in customer satisfaction.",
        "{e} leads a company-wide sustainability program.",
        "{e} leads weekly all-hands strategy meetings.",
        "{e} leads the market in net retention.",
        "{e} leads an internal mentorship initiative.",
    ],
}


def _company(r) -> str:
    return r.choice(_PREFIX) + r.choice(_SUFFIX).capitalize()


def _doc_for(r, entity, attr, value, n_filler, fact_tmpl, filler_tmpls=None) -> str:
    """One short document about `entity` stating the (attr, value) fact, padded
    with filler. In semantic mode the filler is keyword-matched lexical-trap
    sentences (see _SOFT_TRAP_FILLER); otherwise generic filler."""
    sents = [fact_tmpl[attr].format(e=entity, v=value)]
    for _ in range(n_filler):
        if filler_tmpls:
            sents.append(r.choice(filler_tmpls).format(e=entity))
        else:
            sents.append(f"{entity} update: {r.choice(_FILLER)}")
    r.shuffle(sents)
    return f"[{entity}] " + " ".join(sents)


@dataclass
class Example:
    question: str
    gold: str
    context: str
    n_docs: int
    mode: str = "lexical"


def make_benchmark(n_examples=40, n_docs=10, n_filler=3, seed=7,
                   mode="lexical") -> list[Example]:
    """Multi-doc QA with distractors. `mode`:
       * "lexical"  — query/fact share words (BM25-friendly)
       * "semantic" — synonym queries, words-disjoint facts (needs understanding)
    """
    fact_tmpl = _FACT_SEMANTIC if mode == "semantic" else _FACT_LEXICAL
    r = random.Random(seed if mode == "lexical" else seed + 9999)
    examples: list[Example] = []
    for _ in range(n_examples):
        attr = r.choice(list(_ATTRS))
        entities = []
        while len(entities) < n_docs:
            e = _company(r)
            if e not in entities:
                entities.append(e)
        gold_entity = entities[0]

        # Distinct gold value; every distractor value differs from the gold so a
        # wrong-document answer is gradeable as wrong.
        gold_value = _ATTRS[attr](r)
        values = [gold_value]
        for _ in range(n_docs - 1):
            v = _ATTRS[attr](r)
            while v == gold_value:
                v = _ATTRS[attr](r)
            values.append(v)

        pool = _SOFT_TRAP_FILLER[attr] if mode == "semantic" else None
        docs = [_doc_for(r, entities[i], attr, values[i], n_filler, fact_tmpl, pool)
                for i in range(n_docs)]

        # Gold document goes in the MIDDLE — worst case for the reader.
        rest = docs[1:]
        r.shuffle(rest)
        mid = len(rest) // 2
        ordered = rest[:mid] + [docs[0]] + rest[mid:]

        q_tmpl = _Q_SEMANTIC[attr] if mode == "semantic" else _Q_LEXICAL
        examples.append(
            Example(
                question=q_tmpl.format(e=gold_entity, attr=attr),
                gold=gold_value,
                context="\n\n".join(ordered),
                n_docs=n_docs,
                mode=mode,
            )
        )
    return examples


# ----------------------------------------------------------------------------
# Eval harness.
# ----------------------------------------------------------------------------
@dataclass
class RatioResult:
    ratio_target: float
    accuracy: float
    avg_in_tokens: float
    n: int


def run_ratio_sweep(client, examples, ratios, strategy="extractive",
                    model=DOWNSTREAM_MODEL, progress=None, workers=8,
                    compress_fn=None) -> list[RatioResult]:
    """For each target ratio, compress every example, ask the reader, grade.
    `compress_fn(context, query, ratio) -> str` overrides the built-in strategy
    (used to drop in the trained neural compressor)."""
    results = []
    for ri, ratio in enumerate(ratios):
        def run_one(ex, ratio=ratio):
            if ratio >= 0.999:
                ctx = ex.context
            elif compress_fn is not None:
                ctx = compress_fn(ex.context, ex.question, ratio)
            else:
                ctx = compress(ex.context, ex.question, ratio, strategy,
                               client=client).text
            out = ask(client, ctx, ex.question, model)
            return graded_correct(out["answer"], ex.gold), out["in_tokens"]

        rows = _pmap(run_one, examples, workers)
        n = len(examples)
        correct = sum(c for c, _ in rows)
        in_toks = sum(t for _, t in rows)
        results.append(RatioResult(ratio, correct / n, in_toks / n, n))
        if progress:
            progress(ri + 1, len(ratios), results[-1])
    return results


def run_length_sweep(client, doc_counts, conditions=("full", "compressed"),
                     ratio=0.4, n_examples=30, model=DOWNSTREAM_MODEL,
                     n_filler=3, seed=11, mode="lexical", progress=None,
                     workers=8, compress_fn=None):
    """Accuracy vs context length, full vs compressed — the lost-in-the-middle
    demonstration. Returns {condition: {"acc": [...], "tok": [...]}}."""
    out = {c: {"acc": [], "tok": []} for c in conditions}
    for di, nd in enumerate(doc_counts):
        bench = make_benchmark(n_examples, nd, n_filler, seed, mode=mode)
        for cond in conditions:
            def run_one(ex, cond=cond):
                if cond == "full":
                    ctx = ex.context
                elif compress_fn is not None:
                    ctx = compress_fn(ex.context, ex.question, ratio)
                else:
                    ctx = compress(ex.context, ex.question, ratio, client=client).text
                res = ask(client, ctx, ex.question, model)
                return graded_correct(res["answer"], ex.gold), res["in_tokens"]

            rows = _pmap(run_one, bench, workers)
            out[cond]["acc"].append(sum(c for c, _ in rows) / len(bench))
            out[cond]["tok"].append(sum(t for _, t in rows) / len(bench))
        if progress:
            progress(di + 1, len(doc_counts), nd)
    return out


# ----------------------------------------------------------------------------
# Cost helpers.
# ----------------------------------------------------------------------------
def cost_per_million_requests(avg_in_tokens: float, model: str) -> float:
    """Input-side $ for 1,000,000 requests at this prompt size."""
    return avg_in_tokens * PRICING[model]["in"]  # (tok * $/1e6tok) * 1e6 req
