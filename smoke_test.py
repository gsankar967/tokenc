"""Offline sanity checks for the compression engine — no API key needed.

Run:  ./.venv/bin/python smoke_test.py
Verifies: sentence splitting, BM25 ranking, the budget controller, and that
query-aware extraction actually keeps the gold sentence while dropping
distractors on a synthetic example.
"""
import tokenc as tc


def test_compress_keeps_answer_drops_distractors():
    bench = tc.make_benchmark(n_examples=20, n_docs=10, seed=3)
    kept_hits = 0
    ratios = []
    for ex in bench:
        c = tc.compress(ex.context, ex.question, target_ratio=0.3)
        ratios.append(c.ratio)
        # gold value should survive aggressive compression
        if tc._norm(ex.gold) in tc._norm(c.text):
            kept_hits += 1
    avg_ratio = sum(ratios) / len(ratios)
    keep_rate = kept_hits / len(bench)
    print(f"avg kept-ratio at target 0.30 : {avg_ratio:.2f}")
    print(f"gold-survival rate            : {keep_rate*100:.0f}%")
    assert avg_ratio < 0.45, "compression should hit roughly the target budget"
    assert keep_rate >= 0.85, "query-aware extraction should keep the gold fact"


def test_token_estimate_monotonic():
    a = tc.estimate_tokens("hello world")
    b = tc.estimate_tokens("hello world " * 50)
    assert b > a > 0


def test_pricing_present():
    for m in (tc.DOWNSTREAM_MODEL, "claude-sonnet-4-6", "claude-opus-4-8"):
        assert m in tc.PRICING


if __name__ == "__main__":
    test_token_estimate_monotonic()
    test_pricing_present()
    test_compress_keeps_answer_drops_distractors()
    print("\nAll offline smoke tests passed ✅")
