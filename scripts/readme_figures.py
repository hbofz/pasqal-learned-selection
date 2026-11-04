"""The README's figure, as a PNG in docs/figures/. Run from the repository root:

    MPLBACKEND=Agg OMP_NUM_THREADS=1 .venv/bin/python scripts/readme_figures.py

Every number is computed from the tracked records in results/eval/; nothing is typed in by hand.
- tradeoff.png: the 520 fresh problems (seeds 1020 to 1039), exact piece solver. fresh4c_* and fresh5b_* records.

Gap is 100 * (best - size) / best, where best is the largest set any record found on that problem (the exact solver's
answer, proven optimal on almost every problem). The script prints the pooled means it plots, so they can be checked
against the README's table.
"""
import datetime as dt
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learned_selection.instances import FRESH_SEEDS, HELD_OUT_SETTINGS, TRAIN_SETTINGS
from learned_selection.market import EPISODES

OUT = Path("docs/figures")
EVAL = Path("results/eval")
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1f1e1c", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white"})

GLS, LIN, DGIN = "gls", "lin-h2h4+sage-h5", "dgin-h2h4h5"
BLD, S8, BANDIT, GREEDY = "bld-h3+dgin-h4+dgin-h5", "bld-h3+sage-h4+dsage-h5+mcts-c0.5", "bandit-r", "greedy-mindeg"
NAMES = {GLS: "Pasqal's rule (GLS)", LIN: "Linear scorer + value network", DGIN: "Self-corrected graph network",
         BLD: "Learned builder", S8: "Builder + look-ahead search", BANDIT: "Adaptive search width",
         GREEDY: "Greedy (no quantum runs)"}
POOLS = {**{(f, json.dumps(p, sort_keys=True)): "seen" for f, p in TRAIN_SETTINGS},
         **{(f, json.dumps(p, sort_keys=True)): "unseen" for f, p in HELD_OUT_SETTINGS}}
TITLES = {"seen": "Graph types seen in training", "unseen": "Graph types never seen in training"}


def read(*patterns) -> list[dict]:
    return [json.loads(line) for pat in patterns for path in sorted(glob.glob(str(EVAL / pat)))
            for line in open(path) if line.strip()]


def inst(r) -> tuple:
    return r["family"], json.dumps(r["params"], sort_keys=True), r["n"], r["seed"]


def pool(r) -> str:
    return POOLS[inst(r)[:2]]


def gap(r, best) -> float:
    b = best[inst(r)]
    return 100 * (b - r["size"]) / b


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# Fresh problems: one configuration per method, as in the progress update (widths 4,2,10; look-ahead at budget 100;
# the adaptive width is training seed 0's model at c = 0.01, which bench reads as l = 10).
fresh = [r for r in read("fresh4c_*.jsonl", "fresh5b_*.jsonl") if r["seed"] in FRESH_SEEDS]
best = defaultdict(int)
for r in fresh:
    best[inst(r)] = max(best[inst(r)], r["size"])
bandit0 = read("fresh5b_s0.jsonl")
CONFIG = {GLS: 10, LIN: 10, DGIN: 10, BLD: 10, S8: 100}
picked = defaultdict(dict)                       # (method, pool) -> instance -> record, one record per instance
for r in fresh:
    if r["method"] == GREEDY or (CONFIG.get(r["method"]) == r["l"] and (r["k"], r["s"], r["restarts"]) == (4, 2, 1)):
        picked[r["method"], pool(r)][inst(r)] = r
for r in bandit0:
    if r["l"] == 10:
        picked[BANDIT, pool(r)][inst(r)] = r
stats = {}
for (m, p), recs in picked.items():
    stats[m, p] = (mean(r["quantum_runs"] for r in recs.values()), mean(gap(r, best) for r in recs.values()),
                   mean(r["pasqal_runs"] for r in recs.values()), len(recs))
print("fresh problems: method, pool, problems, gap, quantum runs, pieces sent")
for (m, p), (runs, g, sent, count) in sorted(stats.items()):
    print(f"  {m:36s} {p:6s} {count:4d} {g:6.2f} {runs:7.1f} {sent:6.1f}")

# 1. Quality against quantum runs.
MARK = {GLS: "o", LIN: "s", DGIN: "P", BLD: "^", S8: "*", BANDIT: "v", GREEDY: "D"}
COLOR = {GLS: ORANGE, GREEDY: AQUA}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
for ax, p, top in ((axes[0], "seen", 16), (axes[1], "unseen", 6.6)):
    for m in MARK:
        runs, g = (0.0, stats[m, p][1]) if m == GREEDY else stats[m, p][:2]
        ax.scatter(runs, g, s=170 if m == S8 else 64, marker=MARK[m], color=COLOR.get(m, BLUE), edgecolor="white",
                   linewidth=1.2, zorder=3, label=NAMES[m])
    ax.set_xlim(-20, 340)
    ax.set_ylim(0, top)
    ax.set_xlabel("Mean quantum runs per problem")
    ax.set_title(TITLES[p], fontsize=9.5, color=INK)
axes[0].set_ylabel("Mean gap to the best answer (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "tradeoff.png")
