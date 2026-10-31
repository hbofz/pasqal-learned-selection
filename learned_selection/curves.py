"""Compare methods on curves of gap against quantum runs instead of at a single width. A learned hook changes
piece sizes, and so run counts, so one width alone can favour either side.

    uv run python -m learned_selection.curves results/eval/*.jsonl --out results/plots
    uv run python -m learned_selection.curves results/eval/val.jsonl --out results/plots-val --split val

Only rows on the split's seeds are read: test seeds by default, validation seeds with --split val, selection seeds
with --split sel, fresh seeds with --split fresh.

Against GLS's lower envelope (the best gap any GLS configuration reaches for its runs):
- d_gap is a method's mean gap minus the envelope's gap at the method's mean runs. Below 0 is better.
- run_ratio is the method's mean runs over the runs the envelope needs to reach the method's mean gap. Below 1 is
  better.
Both are None where the method falls outside the envelope's range, and run_ratio is None where the method makes no
runs. Such cells are counted by the side the method falls on.
- d_clamp is d_gap with the envelope held flat beyond its ends, so it is defined wherever there is an envelope.
"""
import argparse, json, math
from pathlib import Path

import numpy as np

from .bench import (BEAM_METHODS, GNN_KINDS, ORACLE_METHODS, gap_pct, known_method, parse_method, search_of, tilt_of,
                    track_of)
from .instances import (FRESH_SEEDS, HELD_OUT_SETTINGS, SEL_SEEDS, SEL_SIZES, TEST_SEEDS, TEST_SIZES, TRAIN_SETTINGS,
                        VAL_SEEDS)

HEADLINE_CONFIG = (4, 2, 10, 1)   # k, s, l, restarts
RUN_KEYS = ("quantum_runs", "pasqal_runs")


def _rank(method: str) -> tuple:
    return BEAM_METHODS.index(method) if method in BEAM_METHODS else len(BEAM_METHODS), method


def learned_in(rows) -> list[str]:
    """The known methods in `rows` that are neither GLS, oracle nor classical, in BEAM_METHODS order, then sorted."""
    return sorted((m for m in {r["method"] for r in rows}
                   if known_method(m) and not m.startswith("gls") and track_of(m) == "structure"), key=_rank)


def _cell(r) -> tuple:
    return r["family"], json.dumps(r["params"], sort_keys=True), r["n"]


def _config(r) -> tuple:
    return r["method"], r["k"], r["s"], r["l"], r["restarts"]


def bootstrap_ci(values, reps=2000, seed=0) -> tuple[float, float]:
    """95% percentile interval of the mean; (None, None) for no values."""
    v = np.asarray(values, dtype=float)
    if not len(v):
        return None, None
    means = v[np.random.default_rng(seed).integers(0, len(v), (reps, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def load(paths) -> list[dict]:
    """Records from JSONL files, with best_known, proven and gap_pct recomputed per instance across all files. Two
    records of one instance and config that differ in size or quantum runs raise ValueError; exact repeats pass."""
    rows = [json.loads(line) for p in paths for line in Path(p).read_text().splitlines() if line.strip()]
    seen = {}
    for r in (r for r in rows if r.get("k") is not None):
        key = (*_cell(r), r["seed"], *_config(r))
        first = seen.setdefault(key, r)
        if (first["size"], first["quantum_runs"]) != (r["size"], r["quantum_runs"]):
            raise ValueError(f"records disagree for {key}: size {first['size']} and {r['size']}, "
                             f"quantum runs {first['quantum_runs']} and {r['quantum_runs']}")
    best, proven = {}, {}
    for r in rows:
        i = (*_cell(r), r["seed"])
        if not r["exploded"]:
            best[i] = max(best.get(i, 0), r["size"])
        if r["method"] == "exact":
            proven[i] = r["proven"]
    for r in rows:
        i = (*_cell(r), r["seed"])
        r.update(best_known=best.get(i), proven=proven.get(i),
                 gap_pct=None if r["exploded"] else gap_pct(r["size"], best[i]))
    return rows


def envelope(points) -> list[tuple[float, float]]:
    """Lower-left Pareto front of (runs, gap) points, sorted by runs: each kept point has a lower gap than every
    point with fewer runs."""
    env = []
    for runs, gap in sorted(points):
        if not env or gap < env[-1][1]:
            env.append((runs, gap))
    return env


def _interp(pts, x):
    """Linear interpolation through (x, y) points sorted by x; None outside their range."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] or pts):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return None


def gap_at(env, runs):
    return _interp([(math.log10(r), g) for r, g in env], math.log10(runs))


def runs_at(env, gap):
    log_runs = _interp([(g, math.log10(r)) for r, g in reversed(env)], gap)
    return None if log_runs is None else 10 ** log_runs


def _tables(rows) -> dict:
    """{cell: {config: {seed: record}}}"""
    out = {}
    for r in rows:
        out.setdefault(_cell(r), {}).setdefault(_config(r), {})[r["seed"]] = r
    return out


def _prepare(table, cfg, baseline, runs_key):
    """Runs and gaps, shape (1 + baseline configs, instances), over the instances where the config and every
    baseline config finished; also how many of the config's instances that leaves out."""
    cfgs = [cfg, *(c for c in table if c[0] == baseline)]
    seeds = sorted(table[cfg])
    used = [i for i in seeds if all(i in table[c] and not table[c][i]["exploded"] for c in cfgs)]
    R = np.array([[table[c][i][runs_key] for i in used] for c in cfgs], dtype=float).reshape(len(cfgs), len(used))
    G = np.array([[table[c][i]["gap_pct"] for i in used] for c in cfgs], dtype=float).reshape(len(cfgs), len(used))
    return R, G, len(seeds) - len(used)


def _stats(runs, gaps):
    """((d_gap, side), (run_ratio, side), d_clamp) from per-config means, the method first and the baseline configs
    after it. An undefined statistic is None, and its side says where the method falls: "beats" for fewer runs than
    the envelope's cheapest point (d_gap), or for a lower gap than its best point or no runs at all (run_ratio);
    "expensive" for the other direction. With no envelope both are None and have no side. d_clamp is the gap minus
    the envelope's, held flat beyond its ends (the first point's for no runs), and None only with no envelope."""
    env = envelope((r, g) for r, g in zip(runs[1:], gaps[1:]) if r > 0)   # 0 runs has no place on a log scale
    if not env:
        return (None, None), (None, None), None
    g = gap_at(env, runs[0]) if runs[0] > 0 else None
    clamp = gaps[0] - (gap_at(env, min(max(runs[0], env[0][0]), env[-1][0])) if runs[0] > 0 else env[0][1])
    r = runs_at(env, gaps[0]) if runs[0] > 0 else None    # a method with no runs has no ratio, not a ratio of 0
    return ((gaps[0] - g, None) if g is not None else (None, "expensive" if runs[0] > env[-1][0] else "beats"),
            (runs[0] / r, None) if r is not None else (None, "expensive" if gaps[0] > env[0][1] else "beats"),
            clamp)


def _geomean(v):
    return float(np.exp(np.mean(np.log(v))))


def _gain(prepared, reps, seed) -> dict:
    """d_gap averaged and run_ratio geometrically averaged over the cells where each is defined, each with a 95%
    bootstrap CI, and d_clamp averaged the same way over the cells that have an envelope. A resample draws instances
    with replacement within each cell and averages over the cells defined in it; a resample with no defined cell
    counts in *_none. *_cells counts the defined cells, and *_beats and *_expensive the undefined ones by the side of
    the envelope the method falls on."""
    rng = np.random.default_rng(seed)
    point, boot = [], []
    for R, G in prepared:
        n = R.shape[1]
        point.append(_stats(R.mean(1).tolist(), G.mean(1).tolist()) if n else ((None, None), (None, None), None))
        if n:
            idx = rng.integers(0, n, (reps, n))
            Rb, Gb = R[:, idx].mean(2).T.tolist(), G[:, idx].mean(2).T.tolist()
            boot.append([_stats(a, b) for a, b in zip(Rb, Gb)])
    out = {}
    mean = lambda v: float(np.mean(v))
    for j, (name, average) in enumerate((("d_gap", mean), ("run_ratio", _geomean), ("d_clamp", mean))):
        value = (lambda s: s[2]) if j == 2 else (lambda s, j=j: s[j][0])
        values = [value(p) for p in point if value(p) is not None]
        resamples = ([value(cell[b]) for cell in boot if value(cell[b]) is not None] for b in range(reps))
        stats = [average(v) for v in resamples if v]
        lo, hi = np.percentile(stats, [2.5, 97.5]).tolist() if stats else (None, None)
        out.update({name: average(values) if values else None, f"{name}_lo": lo, f"{name}_hi": hi})
        if j < 2:
            sides = [p[j][1] for p in point]
            out.update({f"{name}_none": reps - len(stats), f"{name}_cells": len(values),
                        f"{name}_beats": sides.count("beats"), f"{name}_expensive": sides.count("expensive")})
    return out


def curve_gain(rows, method, baseline="gls", runs_key="quantum_runs", reps=2000, seed=0) -> list[dict]:
    """d_gap and run_ratio of every config of `method` against the baseline's envelope, one row per cell and config.
    Uses only instances where the config and every baseline config finished; `dropped` counts the rest."""
    out = []
    for (family, params, n), table in _tables(rows).items():
        for cfg in [c for c in table if c[0] == method]:
            R, G, dropped = _prepare(table, cfg, baseline, runs_key)
            out.append(dict(family=family, params=json.loads(params), n=n, method=method, k=cfg[1], s=cfg[2],
                            l=cfg[3], restarts=cfg[4], baseline=baseline, runs_key=runs_key, instances=R.shape[1],
                            dropped=dropped, **_gain([(R, G)], reps, seed)))
    return out


def _pools(seeds) -> list[tuple[str, list[tuple]]]:
    """The (name, cells) pools of a seed set: see `headline`."""
    pools = ([("validation", TRAIN_SETTINGS, (100,))] if seeds == VAL_SEEDS else
             [("selection", TRAIN_SETTINGS, SEL_SIZES)] if seeds == SEL_SEEDS else
             [("training", TRAIN_SETTINGS, TEST_SIZES), ("held-out", HELD_OUT_SETTINGS, TEST_SIZES)])
    return [(pool, [(f, json.dumps(p, sort_keys=True), n) for f, p in settings for n in sizes])
            for pool, settings, sizes in pools]


def headline(rows, runs_key, reps=2000, seed=0, seeds=TEST_SEEDS, baseline="gls", methods=None,
             configs: dict[str, tuple] | None = None) -> list[dict]:
    """Each learned method in `rows` (or each of `methods`) against `baseline` (which is skipped as a method), at its
    own (k, s, l, restarts) in `configs` (default HEADLINE_CONFIG), pooled over cells on `seeds` only. On test seeds
    there are two pools, the training settings and the held-out settings, at the test sizes; settings at BIG_N are
    left to the per-setting tables. On VAL_SEEDS the one pool is the validation grid: the training settings at
    n=100. On SEL_SEEDS it is the selection grid: the training settings at SEL_SIZES."""
    tables = _tables([r for r in rows if r["seed"] in seeds])
    out = []
    for pool, cells in _pools(seeds):
        for method in (m for m in (learned_in(rows) if methods is None else methods) if m != baseline):
            cfg = (method, *(configs or {}).get(method, HEADLINE_CONFIG))
            prepared = [_prepare(tables[c], cfg, baseline, runs_key) for c in cells if cfg in tables.get(c, {})]
            if prepared:
                out.append(dict(pool=pool, method=method, baseline=baseline, runs_key=runs_key, cells=len(prepared),
                                instances=sum(R.shape[1] for R, _, _ in prepared),
                                dropped=sum(d for _, _, d in prepared),
                                **_gain([(R, G) for R, G, _ in prepared], reps, seed)))
    return out


PAIRED_MARGIN = 0.25   # gap points a method may add and still save runs (the bar's curve trades this for 20% of runs)


def _paired_cell(table, cfg, base, runs_key):
    """(log of method runs over baseline runs, method gap minus baseline gap), one entry per instance where both
    finished; the log is nan where either made no runs. None where there is no such instance."""
    m, b = table.get(cfg, {}), table.get(base, {})
    used = [i for i in sorted(m) if i in b and not m[i]["exploded"] and not b[i]["exploded"]]
    if not used:
        return None
    logr = [math.log(m[i][runs_key] / b[i][runs_key]) if m[i][runs_key] and b[i][runs_key] else math.nan for i in used]
    return np.array(logr), np.array([m[i]["gap_pct"] - b[i]["gap_pct"] for i in used])


def paired_gain(rows, method, baseline, runs_key="quantum_runs", config=HEADLINE_CONFIG, base_config=None,
                reps=2000, seed=0, seeds=TEST_SEEDS) -> list[dict]:
    """The method at `config` against the baseline at `base_config` (default `config`) at equal width, one row per
    pool as `headline` builds them. In each cell, over the instances where both finished: R, the geometric mean of
    the method's runs over the baseline's (an instance where either made no runs is left out and counted in
    `no_runs`), and D, the mean of the method's gap minus the baseline's. run_ratio is the geometric mean of R over
    cells and d_gap the mean of D, each with a 95% bootstrap CI that draws instances with replacement within each
    cell. The verdict is "costs gap" if the d_gap CI's lower end lies above PAIRED_MARGIN; else, if the run_ratio CI
    lies below 1, "saves runs" when the d_gap CI's upper end is below PAIRED_MARGIN and "saves runs, gap unresolved"
    when it is not; else "no clear change"."""
    tables = _tables([r for r in rows if r["seed"] in seeds])
    cfg, base = (method, *config), (baseline, *(base_config or config))
    rng = np.random.default_rng(seed)
    ci = lambda v: np.percentile(v, [2.5, 97.5]).tolist() if len(v) else (None, None)
    out = []
    for pool, cells in _pools(seeds):
        pts, boots, instances, no_runs = [], [], 0, 0
        for cell in cells:
            if (got := _paired_cell(tables.get(cell, {}), cfg, base, runs_key)) is None:
                continue
            logr, diff = got
            ok = ~np.isnan(logr)
            idx = rng.integers(0, len(diff), (reps, len(diff)))
            kept = ok[idx].sum(1)
            pts.append((logr[ok].mean() if ok.any() else math.nan, diff.mean()))
            boots.append((np.where(kept > 0, np.where(ok, logr, 0.0)[idx].sum(1) / np.maximum(kept, 1), math.nan),
                          diff[idx].mean(1)))
            instances, no_runs = instances + len(diff), no_runs + int((~ok).sum())
        if not pts:
            continue
        L, D = np.array(pts).T
        LB, DB = (np.array(x) for x in zip(*boots))      # cells x reps
        defined = np.isfinite(LB)       # a resample with no run-making instance in a cell leaves that cell out
        some = defined.any(0)
        r_lo, r_hi = ci(np.exp(np.where(defined, LB, 0.0).sum(0)[some] / defined.sum(0)[some]))
        d_lo, d_hi = ci(DB.mean(0))
        saves = r_hi is not None and r_hi < 1
        verdict = ("costs gap" if d_lo > PAIRED_MARGIN else
                   "saves runs" if saves and d_hi < PAIRED_MARGIN else
                   "saves runs, gap unresolved" if saves else "no clear change")
        out.append(dict(pool=pool, method=method, baseline=baseline, runs_key=runs_key, cells=len(pts),
                        instances=instances, no_runs=no_runs,
                        run_ratio=float(np.exp(L[np.isfinite(L)].mean())) if np.isfinite(L).any() else None,
                        run_ratio_lo=r_lo, run_ratio_hi=r_hi,
                        d_gap=float(D.mean()), d_gap_lo=d_lo, d_gap_hi=d_hi, verdict=verdict))
    return out


def reward_gain(rows, method, base_method, cost, config, base_config, seeds=TEST_SEEDS, runs_key="quantum_runs",
                reps=2000, seed=0) -> list[dict]:
    """The method at `config` against `base_method` at `base_config`, one row per pool as `headline` builds them.
    Per instance where both finished, the method's `size - cost * runs` minus the baseline's, averaged within each
    cell and then over cells, with a 95% bootstrap CI that draws instances with replacement within each cell. The
    verdict is "gain" if the CI lies above 0, "worse" if below 0, else "no gain"."""
    tables = _tables([r for r in rows if r["seed"] in seeds])
    cfg, base = (method, *config), (base_method, *base_config)
    reward = lambda r: r["size"] - cost * r[runs_key]
    rng = np.random.default_rng(seed)
    out = []
    for pool, cells in _pools(seeds):
        diffs = []
        for cell in cells:
            m, b = tables.get(cell, {}).get(cfg, {}), tables.get(cell, {}).get(base, {})
            used = [i for i in sorted(m) if i in b and not m[i]["exploded"] and not b[i]["exploded"]]
            if used:
                diffs.append(np.array([reward(m[i]) - reward(b[i]) for i in used]))
        if not diffs:
            continue
        boot = np.mean([d[rng.integers(0, len(d), (reps, len(d)))].mean(1) for d in diffs], axis=0)
        lo, hi = np.percentile(boot, [2.5, 97.5]).tolist()
        out.append(dict(pool=pool, method=method, baseline=base_method, cost=cost, runs_key=runs_key, cells=len(diffs),
                        instances=sum(len(d) for d in diffs), gain=float(np.mean([d.mean() for d in diffs])),
                        lo=lo, hi=hi, verdict="gain" if lo > 0 else "worse" if hi < 0 else "no gain"))
    return out


def summarize(rows, reps=2000, seed=0, seeds=TEST_SEEDS) -> list[dict]:
    """One row per cell and config on `seeds`: mean gap with its CI and mean run counts over finished instances,
    the share of counted solves over 30 atoms, and how many instances exploded."""
    groups = {}
    for r in (r for r in rows if r["seed"] in seeds):
        groups.setdefault((*_cell(r), *_config(r)), []).append(r)
    out = []
    for (family, params, n, method, k, s, l, restarts), recs in groups.items():
        ok = [r for r in recs if not r["exploded"]]
        mean = lambda key: float(np.mean([r[key] for r in ok])) if ok else None
        lo, hi = bootstrap_ci([r["gap_pct"] for r in ok], reps, seed)
        solves = sum(r["quantum_runs"] for r in ok)
        out.append(dict(family=family, params=json.loads(params), n=n, method=method, k=k, s=s, l=l,
                        restarts=restarts, gap=mean("gap_pct"), gap_lo=lo, gap_hi=hi,
                        quantum_runs=mean("quantum_runs"), pasqal_runs=mean("pasqal_runs"),
                        over_30_share=sum(r["over_30"] for r in ok) / solves if solves else None,
                        exploded=len(recs) - len(ok), instances=len(recs)))
    return out


def paired(rows, method, baseline="gls", reps=2000, seed=0, seeds=TEST_SEEDS) -> list[dict]:
    """Per cell and widths at 1 restart, on `seeds`: the mean per-instance gap difference (method minus baseline)
    with its CI, and the mean difference in quantum runs. Instances where either side exploded are dropped and
    counted."""
    out = []
    for (family, params, n), table in _tables([r for r in rows if r["restarts"] == 1 and r["seed"] in seeds]).items():
        for cfg in [c for c in table if c[0] == method]:
            m, base = table[cfg], table.get((baseline, *cfg[1:]), {})
            both = [i for i in m if i in base]
            ok = [i for i in both if not m[i]["exploded"] and not base[i]["exploded"]]
            diffs = [m[i]["gap_pct"] - base[i]["gap_pct"] for i in ok]
            lo, hi = bootstrap_ci(diffs, reps, seed)
            runs = [m[i]["quantum_runs"] - base[i]["quantum_runs"] for i in ok]
            out.append(dict(family=family, params=json.loads(params), n=n, k=cfg[1], s=cfg[2], l=cfg[3],
                            method=method, baseline=baseline, diff=float(np.mean(diffs)) if ok else None, lo=lo,
                            hi=hi, runs_diff=float(np.mean(runs)) if ok else None, instances=len(ok),
                            dropped=len(both) - len(ok)))
    return out


# Colour follows the model kind, never its position: GLS variants and kinds take fixed slots of a palette whose
# adjacent pairs stay distinguishable under colour-vision deficiency (checked in this order: gls, lin, lgb, cma,
# gin, sage, gat, gls-screen, az, against the light surface only, adjacent pairs: worst CVD delta E 9.1, worst
# normal-vision delta E 20.8). az (#882255) is a ninth, off-palette colour chosen because its worst pair with each
# of the other eight is delta E 13.8 under CVD and 21.5 in normal vision. Yellow, aqua and magenta sit under 3:1
# contrast, so those lines lean on the legend. The hook set is the marker, a mix of kinds is dash-dot, and a legend
# names every line, so colour is never the only code. Orange and red are close, so lin and gin lean on marker and
# legend.
COLORS = {"gls": "#2a78d6", "lin": "#eb6834", "lgb": "#1baf7a", "cma": "#eda100", "gin": "#e34948",
          "sage": "#6250d6", "gat": "#008300", "gls-screen": "#e87ba4", "gls-paper": "#a3a29a",
          "gls-code": "#6e6d68", "exact": "#0b0b0b", "az": "#882255"}
COLORS.update({"d" + k: COLORS[k] for k in ("lin", "gin", "sage", "gat")})   # told apart by their dotted line
COLORS.update(bgin=COLORS["gin"], pg=COLORS["az"], bld=COLORS["az"], gfn=COLORS["az"])   # these share az's colour
COLORS["bandit"] = "#8c5a2b"      # brown: a selector among methods, not a model kind
MARKERS = {"h2": "o", "h4": "s", "h2h4": "^", "h5": "D", "h2h4h5": "P", "h3h4h5": "X", "mcts": "*"}
BANDIT_MARKERS = {"bandit-i": "v", "bandit-r": "<"}
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
RUN_LABELS = {"quantum_runs": "quantum runs", "pasqal_runs": "piece-solver runs"}
GNN_COLORS = {COLORS[k] for k in GNN_KINDS}


def _style(method: str) -> tuple[str, str, str]:
    """(colour, marker, line style) of a beam method. A composed name takes the colour of the kind that provides
    its last hook, or of pg, bld or gfn when it holds one, and the marker of its joined hook set (a star for tree
    search). Its line is dashed with a tilt, else dash-dot when it mixes kinds, dotted when that kind is a dlin,
    dgin, dsage or dgat, and solid otherwise. A bandit has its own colour and marker, and a solid line."""
    if method.startswith("gls"):
        return COLORS[method], "o", "-"
    if method in BANDIT_MARKERS:
        return COLORS["bandit"], BANDIT_MARKERS[method], "-"
    if method.startswith("cma-"):
        return COLORS["cma"], MARKERS["h2"], "-"
    parts = parse_method(method)
    kind = next((k for k, _ in parts if k in ("pg", "bld", "gfn")), None) or max(parts, key=lambda p: p[1])[0]
    marker = MARKERS["mcts" if search_of(method) is not None else "".join(sorted({h for _, h in parts}))]
    style = ("--" if tilt_of(method) is not None else "-." if len({k for k, _ in parts}) > 1
             else ":" if kind[0] == "d" else "-")
    return COLORS[kind], marker, style


def plot(summary, out_dir, runs_key="quantum_runs", methods=None) -> list[Path]:
    """One PNG per cell: mean gap (with 95% CI bars) against mean runs on a log scale, one line per beam method
    through its widths at 1 restart, and one dashed line per method and larger restart count. Only `methods` are
    drawn when given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = {}
    for r in summary:
        if (known_method(r["method"]) and track_of(r["method"]) != "classical"
                and (methods is None or r["method"] in methods) and r["gap"] is not None and r[runs_key]):
            cells.setdefault((r["family"], json.dumps(r["params"], sort_keys=True), r["n"]), []).append(r)
    paths = []
    for (family, params, n), recs in cells.items():
        series = {}
        for r in recs:
            series.setdefault((r["method"], r["restarts"]), []).append(r)
        fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150, facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        for (method, restarts), pts in sorted(series.items(), key=lambda kv: (_rank(kv[0][0]), kv[0][1])):
            pts.sort(key=lambda r: r[runs_key])
            color, marker, style = _style(method)
            single = restarts == 1
            hollow = not single or color in GNN_COLORS    # GNN kinds are hollow, so lin and gin differ by more than hue
            x, y = [r[runs_key] for r in pts], [r["gap"] for r in pts]
            ax.errorbar(x, y, yerr=[[r["gap"] - r["gap_lo"] for r in pts], [r["gap_hi"] - r["gap"] for r in pts]],
                        fmt="none", ecolor=color, elinewidth=0.9, alpha=0.5)
            ax.plot(x, y, color=color, lw=1.6, ls=style if single else "--", marker=marker, ms=6,
                    mfc=SURFACE if hollow else color, mec=color if hollow else SURFACE, mew=1.4 if hollow else 1.2,
                    label=method if single else f"{method}, {restarts} restarts")
        ax.set_xscale("log")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelsize=8, which="both")
        ax.set_xlabel(f"mean {RUN_LABELS[runs_key]} per instance (log scale)", color=INK_2, fontsize=9)
        ax.set_ylabel("mean gap to best known (%)", color=INK_2, fontsize=9)
        setting = ", ".join(f"{k}={v}" for k, v in sorted(json.loads(params).items()))
        ax.set_title(f"{family} ({setting}), n={n}", color=INK, fontsize=11, loc="left", pad=18)
        ax.text(0, 1.015, "bars: 95% bootstrap CI over instances; lower left is better", transform=ax.transAxes,
                color=INK_2, fontsize=7.5, va="bottom")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, labelcolor=INK_2,
                  handlelength=3)
        fig.tight_layout()
        path = out_dir / f"{family}_{setting.replace(', ', '_')}_n{n}_{runs_key}.png"
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths


def _num(x, fmt=".2f") -> str:
    return "-" if x is None else format(x, fmt)


def _ci(x, lo, hi, fmt=".2f") -> str:
    return "-" if x is None else f"{x:{fmt}} [{_num(lo, fmt)}, {_num(hi, fmt)}]"


def _print_headline(rows, runs_key, seeds, baseline, methods=None, word="learned", configs=None) -> None:
    print(f"\nHeadline, {RUN_LABELS[runs_key]}: {word} methods at their own widths (default (4, 2, 10), 1 restart), "
          f"against {baseline}; d_gap is the mean over defined cells, run_ratio the geometric mean, d_clamp the mean "
          "with the envelope held flat past its ends")
    print("  def: defined cells; beat: undefined cells that beat the envelope; exp: undefined cells outside it on the "
          "expensive side; none: resamples with no defined cell")
    table = headline(rows, runs_key, seeds=seeds, baseline=baseline, methods=methods, configs=configs)
    if not table:
        print("  no headline cells on these seeds in these files")
        return
    print(f"  {'pool':11}{'method':18}{'cells':>6}{'inst':>6}{'drop':>6}  {'d_clamp [95% CI]':24}"
          f"{'d_gap [95% CI]':24}{'def':>4}{'beat':>5}{'exp':>4}{'none':>6}  {'run_ratio [95% CI]':24}{'def':>4}"
          f"{'beat':>5}{'exp':>4}{'none':>6}")
    for h in table:
        print(f"  {h['pool']:11}{h['method']:18}{h['cells']:>6}{h['instances']:>6}{h['dropped']:>6}  "
              f"{_ci(h['d_clamp'], h['d_clamp_lo'], h['d_clamp_hi']):24}"
              f"{_ci(h['d_gap'], h['d_gap_lo'], h['d_gap_hi']):24}{h['d_gap_cells']:>4}{h['d_gap_beats']:>5}"
              f"{h['d_gap_expensive']:>4}{h['d_gap_none']:>6}  "
              f"{_ci(h['run_ratio'], h['run_ratio_lo'], h['run_ratio_hi'], '.3f'):24}{h['run_ratio_cells']:>4}"
              f"{h['run_ratio_beats']:>5}{h['run_ratio_expensive']:>4}{h['run_ratio_none']:>6}")


def _print_settings(tables, summary, baseline, runs_key, cells, configs=None, methods=None) -> None:
    configs = configs or {}
    print(f"\nPer setting against {baseline}, {RUN_LABELS[runs_key]}: each method at its own widths (default (4, 2, 10), "
          "1 restart); over 30 is the share of counted solves over 30 atoms; none counts resamples with "
          "d_gap/run_ratio undefined")
    stats = {(_cell(s), s["method"]): s for s in summary
             if (s["k"], s["s"], s["l"], s["restarts"]) == tuple(configs.get(s["method"], HEADLINE_CONFIG))}
    for cell in cells:
        print(f"{cell[0]} {cell[1]} n={cell[2]}")
        print(f"    {'method':18}{'inst':>5}{'expl':>5}  {'gap % [95% CI]':22}{'q runs':>8}{'p runs':>8}"
              f"{'over 30':>9}  {'d_gap [95% CI]':24}{'run_ratio [95% CI]':24}{'drop':>5}{'none':>10}")
        for m in sorted({m for c, m in stats if c == cell and (methods is None or m in methods)}, key=_rank):
            s = stats[cell, m]
            R, G, dropped = _prepare(tables[cell], (m, *configs.get(m, HEADLINE_CONFIG)), baseline, runs_key)
            g = _gain([(R, G)], 2000, 0)
            print(f"    {m:18}{s['instances']:>5}{s['exploded']:>5}  {_ci(s['gap'], s['gap_lo'], s['gap_hi']):22}"
                  f"{_num(s['quantum_runs'], '.1f'):>8}{_num(s['pasqal_runs'], '.1f'):>8}"
                  f"{_num(s['over_30_share'], '.1%'):>9}  "
                  f"{_ci(g['d_gap'], g['d_gap_lo'], g['d_gap_hi']):24}"
                  f"{_ci(g['run_ratio'], g['run_ratio_lo'], g['run_ratio_hi'], '.3f'):24}"
                  f"{dropped:>5}{str(g['d_gap_none']) + '/' + str(g['run_ratio_none']):>10}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True, help="directory for the plots")
    ap.add_argument("--split", choices=["val", "test", "sel", "fresh"], default="test",
                    help="seeds to read: validation seeds pooled over the val grid, test seeds, selection seeds, or "
                         "fresh seeds (pooled as the test seeds are)")
    ap.add_argument("--baselines", default="gls,lin-h2h4",
                    help="comma list: the methods the headline and per-setting tables compare against")
    ap.add_argument("--plot-methods", help="comma list of methods to draw; default all")
    ap.add_argument("--methods", help="comma list of methods the headline and per-setting tables show; default all")
    ap.add_argument("--config", action="append", default=[], metavar="METHOD=k,s,l",
                    help="widths a method is read at in the tables, 1 restart; repeatable; default 4,2,10")
    a = ap.parse_args(argv)
    seeds = {"val": VAL_SEEDS, "test": TEST_SEEDS, "sel": SEL_SEEDS, "fresh": FRESH_SEEDS}[a.split]
    configs = {m: (*map(int, w.split(",")), 1) for m, _, w in (c.partition("=") for c in a.config)}
    methods = a.methods and a.methods.split(",")
    rows = [r for r in load(a.paths) if r["seed"] in seeds]
    summary = summarize(rows, seeds=seeds)
    baselines = a.baselines.split(",")
    for baseline in baselines:
        for runs_key in RUN_KEYS:
            _print_headline(rows, runs_key, seeds, baseline, methods, configs=configs)
    if a.split == "val":
        print("\noracle references (exact value pruning, not a structure-track method)")
        for runs_key in RUN_KEYS:
            _print_headline(rows, runs_key, seeds, "gls", ORACLE_METHODS, "oracle")
    tables = _tables(rows)
    code_finished = {}
    for r in rows:
        if r["method"] == "gls-code":
            code_finished[_cell(r)] = code_finished.get(_cell(r), True) and not r["exploded"]
    for baseline, cells in [(b, sorted(tables)) for b in baselines] + [
            ("gls-code", sorted(c for c, ok in code_finished.items() if ok))]:
        for runs_key in RUN_KEYS:
            _print_settings(tables, summary, baseline, runs_key, cells, configs, methods)
    plot_methods = a.plot_methods and a.plot_methods.split(",")
    paths = [p for runs_key in RUN_KEYS for p in plot(summary, a.out, runs_key, plot_methods)]
    print(f"\nwrote {len(paths)} plots to {a.out}")


if __name__ == "__main__":
    main()
