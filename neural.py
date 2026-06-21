"""
Inference for the trained keep/drop compressor.

Loads the fine-tuned token classifier and turns it into a drop-in compressor with
the same interface as the BM25 baseline (`tokenc.compress`). It scores each
sentence by the model's mean KEEP-probability (query-conditioned), then the same
budget controller keeps top sentences to a target token ratio and re-emits them
in original order.

Sentence scores are independent of the target ratio, so we compute them once per
(context, query) and cache to disk — every ratio in the Pareto sweep reuses them.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

import tokenc as tc


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class NeuralCompressor:
    def __init__(self, model_dir="compressor_model", device=None,
                 max_len=320, batch_size=16):
        self.model_dir = str(model_dir)
        self.device = device or _device()
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = (
            AutoModelForTokenClassification.from_pretrained(self.model_dir)
            .to(self.device).eval()
        )
        self.max_len = max_len
        self.batch_size = batch_size
        # KEEP class index (robust to label order)
        self.keep_id = self.model.config.label2id.get("KEEP", 1)
        self._tag = hashlib.sha256(self.model_dir.encode()).hexdigest()[:8]

    @torch.no_grad()
    def _score_batch(self, sentences, query):
        prefix = ["Query:"] + query.split() + ["Context:"]
        plen = len(prefix)
        batch_words = [prefix + s.split() for s in sentences]
        enc = self.tok(
            batch_words, is_split_into_words=True, truncation=True,
            max_length=self.max_len, padding=True, return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits
        keep_prob = F.softmax(logits, dim=-1)[..., self.keep_id]  # (B, T)
        scores = []
        for i in range(len(sentences)):
            wids = enc.word_ids(batch_index=i)
            vals = [keep_prob[i, t].item()
                    for t, w in enumerate(wids) if w is not None and w >= plen]
            scores.append(sum(vals) / len(vals) if vals else 0.0)
        return scores

    def score_sentences(self, context, query):
        sents = tc.split_sentences(context)
        key = f"neural::{self._tag}::{query}::{hashlib.sha256(context.encode()).hexdigest()}"
        hit = tc._CACHE.get(key)
        if hit is not None and len(hit["scores"]) == len(sents):
            return sents, hit["scores"]
        scores = []
        for i in range(0, len(sents), self.batch_size):
            scores.extend(self._score_batch(sents[i:i + self.batch_size], query))
        tc._CACHE.set(key, {"scores": scores})
        return sents, scores

    def compress(self, context, query, target_ratio=0.5, token_fn=tc.estimate_tokens):
        sents, scores = self.score_sentences(context, query)
        orig = token_fn(context)
        if not sents:
            return tc.Compressed(context, query, "neural", orig, orig, 0, 0)
        budget = max(1, int(round(orig * target_ratio)))
        order = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
        kept, used = set(), 0
        for i in order:
            t = token_fn(sents[i])
            if kept and used + t > budget:
                continue
            kept.add(i); used += t
            if used >= budget:
                break
        text = " ".join(sents[i] for i in sorted(kept))
        return tc.Compressed(text, query, "neural", orig, token_fn(text),
                             len(sents), len(kept))
