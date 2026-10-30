"""Labelled graphs for the graph networks: the states the beam visits, with exact labels and exact child values.

Each record is one state graph (nodes renumbered 0..n-1 in `list(g)` order), the vertices of every candidate
piece, the exact maximum-independent-set membership of the state, the best continuation Q* of every candidate,
and every child (what is left after committing to one of a candidate's sets) with its exact independence number.

    uv run python -m learned_selection.graphdata --split train --workers 12 --out cache/train_graphs.pkl
    uv run python -m learned_selection.graphdata --summary cache/train_graphs.pkl [more.pkl ...]
"""
import argparse, os, pickle, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np

from . import instances
from .bench import shard
from .dataset import (DEFERRED_TIME_LIMIT_S, HOOKS, _commit, _setting_name, children_values, collect_states,
                      mis_membership)
from .engine import gls_generator
from .instances import TEST_SEEDS, make


def state_record(g: nx.Graph, first: bool, time_limit_s: float, workers: int) -> dict:
    """One labelled state. Node indices are positions in list(g). member, alpha and q are None when unproven;
    children holds the children proven before the first unproven solve."""
    nodes = list(g)
    at = {v: i for i, v in enumerate(nodes)}
    member = mis_membership(g, time_limit_s, workers)
    maps = gls_generator()(g, len(g))
    q, children = children_values(g, maps, time_limit_s=time_limit_s, workers=workers)
    return dict(
        n=len(g),
        edges=np.array([(at[u], at[v]) for u, v in g.edges()], dtype=np.int32).reshape(-1, 2),
        first=first,
        member=None if member is None else np.array([member[v] for v in nodes], dtype=np.int8),
        alpha=None if member is None else sum(member.values()),
        pieces=[np.array([at[v] for v in m], dtype=np.int16) for m in maps],
        q=None if q is None else np.array(q, dtype=np.int64),
        children=[(np.array([at[v] for v in h], dtype=np.int16), alpha, s) for h, alpha, s in children],
    )


def graph_of(rec: dict) -> nx.Graph:
    """The state graph of a record: nodes 0..n-1 in order, weight 1 on every node, edges in the stored order."""
    h = nx.Graph()
    h.add_nodes_from((i, {"weight": 1}) for i in range(rec["n"]))
    h.add_edges_from(map(tuple, rec["edges"].tolist()))
    return h


def label_graphs(family: str, params: dict, n: int, seed: int, max_states: int = 12,
                 time_limit_s: float = 60.0, big: int = 100) -> dict:
    """Label the states of one instance. States of at most `big` nodes are labelled here; larger ones come back
    in `deferred` as (graph, first) to be labelled with more time and more workers. A state with unproven
    labels keeps its record (the unproven fields are None) and is counted in `dropped` by (hook, first)."""
    if seed in TEST_SEEDS:
        raise ValueError(f"seed {seed} is a test seed; it must not be used for training data")
    start = time.process_time()
    G = make(family, n, seed, **params)
    out = dict(states=[], deferred=[], dropped=Counter())
    for g in collect_states(G, max_states=max_states, seed=seed):
        first = set(g) == set(G)
        if len(g) > big:
            out["deferred"].append((g, first))
            continue
        rec = state_record(g, first, time_limit_s, 1)
        out["states"].append(rec)
        _count_dropped(out["dropped"], (), rec)
    out["cpu_s"] = time.process_time() - start
    return out


def _count_dropped(dropped: Counter, tag: tuple, rec: dict) -> None:
    """Count each hook of the state whose labels are unproven, under (*tag, hook, first)."""
    for hook, label in zip(HOOKS, (rec["member"], rec["q"])):
        if label is None:
            dropped[(*tag, hook, rec["first"])] += 1


def _label_job(job: tuple) -> dict:
    return label_graphs(*job)


def _defer_job(args: tuple) -> tuple[dict, float]:
    """Label one deferred state; the CPU time is measured here, in the worker that spent it."""
    g, first = args
    start = time.process_time()
    rec = state_record(g, first, DEFERRED_TIME_LIMIT_S, 8)
    return rec, time.process_time() - start


def hook_counts(states: list[dict]) -> dict:
    """Per hook, the labelled states and their rows, read from the records: h2 has one row per vertex of a state
    with a proven membership, h4 one row per candidate of a state with proven Q*."""
    return {hook: dict(states=sum(r[key] is not None for r in states),
                       rows=sum(r[size] if size == "n" else len(r[size]) for r in states if r[key] is not None))
            for hook, key, size in zip(HOOKS, ("member", "q"), ("n", "pieces"))}


def summarize(metas: list[dict], states: list[dict]) -> None:
    counts = hook_counts(states)
    lost = Counter()
    for m in metas:
        for d in m["dropped"]:
            lost[d["hook"]] += d["count"]
    per_setting = Counter()
    for m in metas:
        for d in m["settings"]:
            per_setting[(d["setting"], d["n"])] += d["instances"]
    first = Counter((r["setting"], r["instance_n"], hook) for r in states if r["first"]
                    for hook, key in zip(HOOKS, ("member", "q")) if r[key] is not None)
    print(f"{sum(m['instances'] for m in metas)} instances; {len(states)} states")
    for hook in HOOKS:
        c = counts[hook]
        total = c["states"] + lost[hook]
        print(f"{hook}: {c['states']} states labelled ({c['rows']} rows), {lost[hook]} dropped "
              f"({lost[hook] / total if total else 0:.1%})")
    print(f"children: {sum(len(r['children']) for r in states)}")
    print(f"{'setting':32}{'n':>5}{'instances':>11}{'h2 first':>10}{'h4 first':>10}")
    for (name, n), total in per_setting.items():
        print(f"{name:32}{n:>5}{total:>11}{first[(name, n, 'h2')]:>10}{first[(name, n, 'h4')]:>10}")
    print(f"label cpu hours: {sum(m['label_cpu_s'] for m in metas) / 3600:.2f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"])
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out")
    ap.add_argument("--shard", help="I/N: run every N-th job starting at I")
    ap.add_argument("--summary", nargs="+", metavar="PATH", help="print the summary of these files and exit")
    a = ap.parse_args(argv)
    if a.summary:
        loaded = [pickle.load(open(p, "rb")) for p in a.summary]
        summarize([d["meta"] for d in loaded], [r for d in loaded for r in d["states"]])
        return
    if a.split is None or a.workers is None or a.out is None:
        ap.error("--split, --workers and --out are required unless --summary is given")
    commit = _commit()                                 # read now, so a git failure cannot cost hours of labels
    seeds = instances.TRAIN_SEEDS if a.split == "train" else instances.VAL_SEEDS
    jobs = shard([(fam, params, n, seed) for fam, params in instances.TRAIN_SETTINGS for n in instances.TRAIN_SIZES
                  for seed in seeds], a.shard)
    states, dropped, deferred, cpu_s = [], Counter(), [], 0.0   # dropped: (setting, n, hook, first) -> count

    def add(job, rec):
        rec.update(setting=_setting_name(job[0], job[1]), instance_n=job[2], seed=job[3])
        states.append(rec)

    with Pool(a.workers) as pool:
        for i, (job, r) in enumerate(zip(jobs, pool.imap(_label_job, jobs)), 1):
            tag = (_setting_name(job[0], job[1]), job[2])
            for rec in r["states"]:
                add(job, rec)
            for (hook, first), count in r["dropped"].items():
                dropped[(*tag, hook, first)] += count
            deferred.extend((job, g, first) for g, first in r["deferred"])
            cpu_s += r["cpu_s"]
            if i % 20 == 0 or i == len(jobs):
                print(f"pass 1: {i}/{len(jobs)} instances", flush=True)
    if deferred:
        with Pool(max(1, (os.cpu_count() or 1) // 8)) as pool:
            for i, ((job, _, _), (rec, cpu)) in enumerate(
                    zip(deferred, pool.imap(_defer_job, [(g, f) for _, g, f in deferred])), 1):
                add(job, rec)
                _count_dropped(dropped, (_setting_name(job[0], job[1]), job[2]), rec)
                cpu_s += cpu
                print(f"pass 2: {i}/{len(deferred)} deferred states", flush=True)

    per_setting = Counter((_setting_name(f, p), n) for f, p, n, _ in jobs)
    meta = dict(split=a.split, shard=a.shard, instances=len(jobs), states=len(states),
                counts=hook_counts(states),
                settings=[dict(setting=s, n=n, instances=c) for (s, n), c in per_setting.items()],
                deferred_states=len(deferred),
                dropped=[dict(setting=s, n=n, hook=h, first=bool(f), count=c) for (s, n, h, f), c in dropped.items()],
                label_cpu_s=cpu_s, commit=commit)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(dict(meta=meta, states=states), f)
    summarize([meta], states)


if __name__ == "__main__":
    main()
