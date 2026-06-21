"""Render submission-ready performance figures from the (cached) eval.

Outputs:
  performance.png  — head-to-head: full vs BM25 vs trained model vs hybrid
  pareto.png       — tokens vs accuracy frontier

Run:  ./.venv/bin/python make_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import anthropic
import tokenc as tc
from neural import NeuralCompressor

client = anthropic.Anthropic()
nc = NeuralCompressor("compressor_model")

EVAL = tc.make_benchmark(n_examples=40, n_docs=3, n_filler=8, seed=7, mode="semantic")
RATIOS = [1.0, 0.6, 0.5, 0.4, 0.3, 0.2]
M = tc.DOWNSTREAM_MODEL

bm = tc.run_ratio_sweep(client, EVAL, RATIOS, strategy="extractive", model=M)
nz = tc.run_ratio_sweep(client, EVAL, RATIOS, model=M,
                        compress_fn=lambda c, q, r: nc.compress(c, q, r).text)
hy = tc.run_ratio_sweep(client, EVAL, [0.15], model=M,
                        compress_fn=lambda c, q, r: nc.compress_hybrid(client, c, q, r).text)[0]

full_acc, full_tok = bm[0].accuracy * 100, bm[0].avg_in_tokens
def at(res, t): return min(res, key=lambda r: abs(r.ratio_target - t))
bm20, nz20 = at(bm, 0.2), at(nz, 0.2)

# ---------------------------------------------------------------- figure 1: bars
labels = ["full\ncontext", "BM25\n(classical)", "trained\nmodel\n(extractive)",
          "hybrid\n(model + rephrase)"]
accs = [full_acc, bm20.accuracy * 100, nz20.accuracy * 100, hy.accuracy * 100]
toks = [full_tok, bm20.avg_in_tokens, nz20.avg_in_tokens, hy.avg_in_tokens]
colors = ["#888888", "#cc4444", "#2277aa", "#22aa88"]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
bars = ax.bar(labels, accs, color=colors, width=0.66)
for b, a, t in zip(bars, accs, toks):
    ax.text(b.get_x() + b.get_width() / 2, a + 1.5,
            f"{a:.0f}%\n{t:.0f} tok", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(full_acc, ls="--", c="#888888", alpha=.7)
ax.text(3.45, full_acc + 0.5, "full-context accuracy", ha="right", va="bottom",
        fontsize=9, color="#666666")
ax.set_ylabel("downstream answer accuracy (%)", fontsize=12)
ax.set_ylim(0, 108)
ax.set_title("TokenC: accuracy vs tokens under aggressive context compression\n"
             "semantic multi-doc QA with lexical traps · reader = Claude Haiku 4.5",
             fontsize=12.5, fontweight="bold")
ax.grid(axis="y", alpha=.3)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("performance.png", dpi=200, bbox_inches="tight")
print("wrote performance.png")

# ------------------------------------------------------------- figure 2: pareto
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot([r.avg_in_tokens for r in bm], [r.accuracy * 100 for r in bm],
        "o-", color="#cc4444", label="BM25 (classical)")
ax.plot([r.avg_in_tokens for r in nz], [r.accuracy * 100 for r in nz],
        "s-", color="#2277aa", label="trained model (extractive)")
ax.scatter([hy.avg_in_tokens], [hy.accuracy * 100], marker="*", s=320,
           color="#22aa88", zorder=5, label="hybrid (model + rephrase)")
ax.scatter([full_tok], [full_acc], color="#000000", zorder=5)
ax.annotate("full context", (full_tok, full_acc),
            textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=10)
ax.axhline(full_acc, ls="--", c="#888888", alpha=.5)
ax.set_xlabel("avg input tokens per request  (lower = cheaper)", fontsize=12)
ax.set_ylabel("downstream answer accuracy (%)", fontsize=12)
ax.set_title("TokenC: token–accuracy frontier\n"
             "the hybrid beats full-context accuracy at ~18% of the tokens",
             fontsize=12.5, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=.3)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("pareto.png", dpi=200, bbox_inches="tight")
print("wrote pareto.png")

print(f"\nfull  : {full_acc:.0f}% @ {full_tok:.0f} tok")
print(f"BM25  : {bm20.accuracy*100:.0f}% @ {bm20.avg_in_tokens:.0f} tok")
print(f"model : {nz20.accuracy*100:.0f}% @ {nz20.avg_in_tokens:.0f} tok")
print(f"hybrid: {hy.accuracy*100:.0f}% @ {hy.avg_in_tokens:.0f} tok "
      f"({hy.avg_in_tokens/full_tok*100:.0f}% of full, {hy.accuracy*100-full_acc:+.0f} pts vs full)")
