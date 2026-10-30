"""Training data for the learned models: the states the beam visits, and exact labels for them.

Growth hook: one row per vertex, labelled 1 if it is in one proven maximum independent set of the state.
Candidate-choice hook: one row per candidate piece, labelled with the size of the best independent set of the
state that commits to one of the piece's sets (the optimal continuation, Q*).

    uv run python -m learned_selection.dataset --split train --workers 12 --out cache/train.npz
"""
import argparse, json, os, random, subprocess, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np

from . import instances
from .engine import _remove_closed, code_prune, exact_piece_solver, gls_generator, run_beam
from .features import PIECE_FEATURES, VERTEX_FEATURES, piece_features, vertex_features
from .instances import TEST_SEEDS, make
from .solvers import exact_mis, first_maximum_set, top_maximal_sets

HOOKS = ("h2", "h4")
DEFERRED_TIME_LIMIT_S = 120.0


def collect_states(G: nx.Graph, k: int = 4, s: int = 2, l: int = 10, max_states: int = 12,
                   seed: int = 0) -> list[nx.Graph]:
    """The distinct graphs (by node set) the beam hands to the generator on G, in visit order; the first is G
    itself. When there are more than max_states, G plus a seeded random sample of the rest, still in visit order."""
    generate, seen = gls_generator(), {}

    def record(g: nx.Graph, k: int):
        seen.setdefault(frozenset(g), g)
        return generate(g, k)

    run_beam(G, record, exact_piece_solver, k, s, code_prune(k, l, count_small=True))
    states = list(seen.values())
    if len(states) > max_states:
        rest = sorted(random.Random(seed).sample(range(1, len(states)), max_states - 1))
        states = [states[0]] + [states[i] for i in rest]
    return states


def q_values(g: nx.Graph, maps: list[dict], s: int = 2, small: int = 5, time_limit_s: float = 20.0,
             workers: int = 1) -> list[int] | None:
    """For each candidate, the size of the best independent set of g that commits to one of the piece's sets
    (the engine's own choice of sets: the first maximum set for a piece of `small` nodes or fewer, else the top
    s maximal sets) and finishes the rest exactly. One solve per distinct vertex set. None if any is unproven."""
    best: dict[frozenset, int] = {}
    for m in maps:
        key = frozenset(m)
        if key in best:
            continue
        piece = g.subgraph(m)
        sets = [first_maximum_set(piece)] if len(key) <= small else top_maximal_sets(piece, s)
        total = 0
        for S in sets:
            rest, proven = exact_mis(_remove_closed(g, S), time_limit_s, workers)
            if not proven:
                return None
            total = max(total, len(S) + len(rest))
        best[key] = total
    return [best[frozenset(m)] for m in maps]


def mis_membership(g: nx.Graph, time_limit_s: float = 20.0, workers: int = 1) -> dict | None:
    """Each vertex mapped to 1 if it is in the maximum independent set found, else 0. None if unproven."""
    S, proven = exact_mis(g, time_limit_s, workers)
    return {v: int(v in S) for v in g} if proven else None


def label_state(g: nx.Graph, time_limit_s: float, workers: int) -> tuple[tuple | None, tuple | None]:
    """(X, y) for the growth hook and for the candidate-choice hook of one state; None for a hook whose
    labels are unproven. The candidates are every piece GLS grows on g."""
    member = mis_membership(g, time_limit_s, workers)
    growth = None
    if member is not None:
        nodes, X = vertex_features(g)
        growth = X, np.array([member[v] for v in nodes], dtype=np.int64)
    maps = gls_generator()(g, len(g))
    q = q_values(g, maps, time_limit_s=time_limit_s, workers=workers)
    choice = None if q is None else (piece_features(g, maps), np.array(q, dtype=np.int64))
    return growth, choice


def label_instance(family: str, params: dict, n: int, seed: int, max_states: int = 12,
                   time_limit_s: float = 20.0, big: int = 100) -> dict:
    """Label the states of one instance. States of at most `big` nodes are labelled here; larger ones come back
    in `deferred` to be labelled with more time and more workers. `is_first` marks the full input graph."""
    if seed in TEST_SEEDS:
        raise ValueError(f"seed {seed} is a test seed; it must not be used for training data")
    start = time.process_time()
    G = make(family, n, seed, **params)
    out = dict(h2=[], h4=[], deferred=[], dropped=Counter())
    for g in collect_states(G, max_states=max_states, seed=seed):
        first = set(g) == set(G)
        if len(g) > big:
            out["deferred"].append((g, first))
            continue
        for hook, labels in zip(HOOKS, label_state(g, time_limit_s, 1)):
            if labels is None:
                out["dropped"][(hook, first)] += 1
            else:
                out[hook].append((*labels, len(g), first))
    out["cpu_s"] = time.process_time() - start
    return out


def _setting_name(family: str, params: dict) -> str:
    return f"{family}({','.join(f'{k}={v}' for k, v in params.items())})"


def _label_job(job: tuple) -> dict:
    return label_instance(*job)


def _arrays(parts: list[tuple], width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack (X, y, state size) parts into X, y, group ids and weights. One group per state, one row per label."""
    if not parts:
        return np.zeros((0, width)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    rows = [len(y) for _, y, _ in parts]
    return (np.vstack([X for X, _, _ in parts]).astype(np.float64), np.concatenate([y for _, y, _ in parts]),
            np.repeat(np.arange(len(parts), dtype=np.int64), rows),
            np.repeat([float(size) for _, _, size in parts], rows))


def _commit() -> str:
    return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=Path(__file__).parent, text=True,
                          capture_output=True, check=True).stdout.strip()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    seeds = instances.TRAIN_SEEDS if a.split == "train" else instances.VAL_SEEDS
    jobs = [(fam, params, n, seed) for fam, params in instances.TRAIN_SETTINGS for n in instances.TRAIN_SIZES
            for seed in seeds]
    parts = {hook: [] for hook in HOOKS}               # (X, y, state size) per labelled state
    labelled = Counter()                               # (setting, n, hook) -> first states with labels
    dropped = Counter()                                # (setting, n, hook, first) -> states without labels
    deferred, cpu_s = [], 0.0

    def take(job, r):
        nonlocal cpu_s
        tag = (_setting_name(job[0], job[1]), job[2])
        for hook in HOOKS:
            for X, y, size, first in r[hook]:
                parts[hook].append((X, y, size))
                labelled[(*tag, hook)] += first
        for (hook, first), count in r["dropped"].items():
            dropped[(*tag, hook, first)] += count
        deferred.extend((tag, g, first) for g, first in r["deferred"])
        cpu_s += r["cpu_s"]

    with Pool(a.workers) as pool:
        for i, (job, r) in enumerate(zip(jobs, pool.imap(_label_job, jobs)), 1):
            take(job, r)
            if i % 20 == 0 or i == len(jobs):
                print(f"pass 1: {i}/{len(jobs)} instances", flush=True)
    for i, (tag, g, first) in enumerate(deferred, 1):
        start = time.process_time()
        for hook, labels in zip(HOOKS, label_state(g, DEFERRED_TIME_LIMIT_S, os.cpu_count())):
            if labels is None:
                dropped[(*tag, hook, first)] += 1
            else:
                parts[hook].append((*labels, len(g)))
                labelled[(*tag, hook)] += first
        cpu_s += time.process_time() - start
        print(f"pass 2: {i}/{len(deferred)} deferred states", flush=True)

    arrays, counts = {}, {}
    for hook, width in zip(HOOKS, (len(VERTEX_FEATURES), len(PIECE_FEATURES))):
        d = hook[1:]
        arrays[f"X{d}"], arrays[f"y{d}"], arrays[f"g{d}"], arrays[f"w{d}"] = _arrays(parts[hook], width)
        counts[hook] = dict(states=len(parts[hook]), rows=len(arrays[f"y{d}"]))
    meta = dict(split=a.split, instances=len(jobs), counts=counts, deferred_states=len(deferred),
                dropped=[dict(setting=s, n=n, hook=h, first=bool(f), count=c) for (s, n, h, f), c in dropped.items()],
                label_cpu_s=cpu_s, commit=_commit())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, meta=json.dumps(meta), **arrays)

    print(f"{len(jobs)} instances; " + "; ".join(f"{h}: {c['states']} states, {c['rows']} rows"
                                                  for h, c in counts.items()))
    per_setting = Counter((_setting_name(f, p), n) for f, p, n, _ in jobs)
    print(f"{'setting':32}{'n':>5}{'instances':>11}{'h2 first':>10}{'h4 first':>10}")
    for (name, n), total in per_setting.items():
        print(f"{name:32}{n:>5}{total:>11}{labelled[(name, n, 'h2')]:>10}{labelled[(name, n, 'h4')]:>10}")
    print(f"label cpu hours: {cpu_s / 3600:.2f}")


if __name__ == "__main__":
    main()
