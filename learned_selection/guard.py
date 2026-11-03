"""Check that earlier result records and label pickles still reproduce with the current code, before they are reused.

    uv run python -m learned_selection.guard --records 'results/eval/test_shard*.jsonl' --methods lin-h2 \
        --stride 52 --count 10 --labels 'cache/train_graphs*.pkl' --label-count 3 --models models --workers 2
"""
import argparse, json, pickle, sys
from functools import partial
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .bench import _instance_key, grid, run_instance
from .dataset import _setting_name
from .graphdata import label_graphs
from .instances import TRAIN_SETTINGS

COMPARED = ("size", "quantum_runs", "pasqal_runs", "atoms", "exploded")
LABELLED = ("n", "edges", "member", "alpha", "pieces", "q", "children")


def _map(fn, jobs, workers):
    if workers <= 1:
        return list(map(fn, jobs))
    with Pool(workers) as pool:
        return pool.map(fn, jobs)


def _equal(a, b) -> bool:
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(map(_equal, a, b))
    return bool(np.array_equal(a, b))


def _rerun(inst, old, methods, models_dir):
    """(same, differ, missing, messages) of one instance: each method with old beam records on it is rerun at
    the widths found in them, one run per restart count, so exactly the old keys are rerun."""
    ik, same, differ, missing, msgs = _instance_key(*inst), 0, 0, 0, []
    for m in methods:
        keys = [k for k in old if k[:5] == (*ik, m)]
        if not keys:
            missing += 1
            continue
        for r in sorted({k[8] for k in keys}):
            rows = run_instance(*inst, [m], widths=sorted(k[5:8] for k in keys if k[8] == r), restarts=[r],
                                time_limit_s=old[keys[0]]["time_limit_s"], models_dir=models_dir)
            for row in rows:
                key = (*ik, m, row["k"], row["s"], row["l"], row["restarts"])
                bad = [f"{f} {old[key][f]} -> {row[f]}" for f in COMPARED if old[key][f] != row[f]]
                if bad:
                    differ += 1
                    msgs.append(f"differ {key}: {', '.join(bad)}")
                else:
                    same += 1
    return same, differ, missing, msgs


def records_guard(paths, methods, instances, workers, models_dir) -> tuple[int, int, int]:
    """(same, differ, missing): missing counts a method with no old record on an instance. Two records with one
    key that disagree on a compared field are a conflict and count as differ."""
    old, conflicts = {}, []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line) if line.strip() else {"k": None}
            if r["k"] is None:
                continue
            key = (*_instance_key(r["family"], r["params"], r["n"], r["seed"]), r["method"], r["k"], r["s"], r["l"],
                   r["restarts"])
            if key in old and any(old[key][f] != r[f] for f in COMPARED):
                conflicts.append(f"conflict {key}: {p} disagrees with an earlier record")
            old.setdefault(key, r)
    out = _map(partial(_rerun, old=old, methods=methods, models_dir=models_dir), instances, workers)
    for msg in conflicts + [m for o in out for m in o[3]]:
        print(msg)
    same, differ, missing = (sum(o[i] for o in out) for i in range(3))
    return same, differ + len(conflicts), missing


def _relabel(job):
    inst, old = job
    new = label_graphs(*inst)["states"]
    bad = [[f for f in LABELLED if not _equal(a[f], b[f])] for a, b in zip(old, new)]
    msgs = [f"differ {inst} state {i}: {', '.join(fs)}" for i, fs in enumerate(bad) if fs]
    return len(bad) - len(msgs), len(msgs) + abs(len(old) - len(new)), msgs


def _states(paths):
    return [r for p in paths for r in pickle.loads(Path(p).read_bytes())["states"] if r["n"] <= 100]


def labels_guard(paths, instances, workers) -> tuple[int, int]:
    """(same, differ) over the states of at most 100 nodes of each instance, relabelled with no policy; a state
    present in only one of the two counts as different."""
    old = {}
    for r in _states(paths):
        old.setdefault((r["setting"], r["instance_n"], r["seed"]), []).append(r)
    out = _map(_relabel, [(i, old.get((_setting_name(i[0], i[1]), i[2], i[3]), [])) for i in instances], workers)
    for msg in (m for o in out for m in o[2]):
        print(msg)
    return sum(o[0] for o in out), sum(o[1] for o in out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", default=[], metavar="GLOB")
    ap.add_argument("--grid", choices=["test", "fresh"], default="test")
    ap.add_argument("--seeds", help="inclusive range a-b: keep only the instances of these seeds, before stride and count")
    ap.add_argument("--methods", default="")
    ap.add_argument("--stride", type=int, default=52)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--labels", nargs="+", default=[], metavar="GLOB")
    ap.add_argument("--label-count", type=int, default=3)
    ap.add_argument("--models", default="models")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    same = differ = missing = 0
    empty = False                                  # a part that was asked for and compared nothing
    if a.records:
        insts = grid(a.grid)
        if a.seeds:
            lo, hi = map(int, a.seeds.split("-"))
            insts = [i for i in insts if lo <= i[3] <= hi]
        insts = insts[::a.stride][:a.count] + (grid("val")[:2] if a.grid == "test" else [])
        paths = sorted(p for g in a.records for p in glob(g))
        same, differ, missing = records_guard(paths, a.methods.split(","), insts, a.workers, a.models)
        empty = same == 0
    if a.labels:
        paths, by_name = sorted(p for g in a.labels for p in glob(g)), {_setting_name(*s): s for s in TRAIN_SETTINGS}
        seen = sorted({(r["setting"], r["instance_n"], r["seed"]) for r in _states(paths)})
        insts = [(*by_name[s], n, seed) for s, n, seed in seen if s in by_name]
        s, d = labels_guard(paths, insts[::max(1, len(insts) // a.label_count)][:a.label_count], a.workers)
        same, differ, empty = same + s, differ + d, empty or s == 0
    print(f"guard: {same} same, {differ} differ, {missing} missing")
    sys.exit(1 if differ or empty else 0)


if __name__ == "__main__":
    main()
