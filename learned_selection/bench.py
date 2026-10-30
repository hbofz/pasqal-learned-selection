"""Benchmark runner: every method on the same instances, one JSON record per (instance, method, widths, restarts).

    uv run python -m learned_selection.bench --grid test --workers 12 --out results/eval/test.jsonl
    uv run python -m learned_selection.bench --family er --params p=0.25 --n 100 --seeds 0-9 \\
        --methods gls-code,gls-paper,greedy-pasqal,greedy-mindeg,exact --widths 4,2,10 --restarts 1 --out results/er.jsonl

Records are appended as each instance finishes. A rerun skips every (instance, method) already in --out, so an
interrupted grid resumes; widths and restarts are not part of that check, so use a new --out to change them.
"""
import argparse, json, re, sys, time
from contextlib import nullcontext
from functools import cache, partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import networkx as nx

from . import dataset
from .engine import RunResult, _remove_closed, best_of_restarts, run_code_bounded, run_code_faithful, run_paper_faithful
from .instances import (BIG_N, BIG_SETTINGS, HELD_OUT_SETTINGS, SEL_SEEDS, SEL_SIZES, TEST_SEEDS, TEST_SIZES,
                        TRAIN_SETTINGS, VAL_SEEDS, make)
from .lattice import BLOCKADE_UM
from .models import h2_score, h4_rank, linear_value, load, load_theta, theta_score
from .search import run_mcts
from .solvers import exact_mis, is_independent, min_degree_greedy
from .vendored import weighted_greedy_independent_set

NEW_METHODS = ["gin-h2", "sage-h2", "gat-h2", "gin-h4", "sage-h4", "gat-h4", "gin-h2h4", "sage-h2h4", "gat-h2h4",
               "lin-h2+gin-h4", "lin-h2+sage-h4", "lin-h2+gat-h4", "lin-h5", "gin-h5", "sage-h5", "gat-h5",
               "lin-h2h4+lin-h5", "lin-h2h4+gin-h5", "lin-h2h4+sage-h5", "lin-h2h4+gat-h5", "gin-h2h4h5"]
# Value pruning by the true optimum of each remaining graph: a reference that computes independent sets outside the
# counted solves, so it is its own track. Validation grid only, and run only when named.
ORACLE_METHODS = ["exact-h5", "lin-h2h4+exact-h5"]
BEAM_METHODS = ["gls-code", "gls", "gls-paper", "gls-screen", "lin-h2", "lin-h4", "lin-h2h4", "lgb-h2", "lgb-h4",
                "lgb-h2h4", "cma-h2"] + NEW_METHODS + ORACLE_METHODS
CLASSICAL_METHODS = ["greedy-pasqal", "greedy-mindeg", "exact"]
KIND_EXT = {"lin": ".json", "lgb": ".txt", "gin": ".pt", "sage": ".pt", "gat": ".pt", "dlin": ".json", "dgin": ".pt",
            "dsage": ".pt", "dgat": ".pt", "az": ".pt"}
GNN_KINDS = ("gin", "sage", "gat", "dgin", "dsage", "dgat", "az")
LINEAR_KINDS = ("lin", "dlin")
SEARCH = re.compile(r"mcts(?:-c(\d+(?:\.\d+)?))?")
MINDEG_SEEDS = 20
SMALL = 5
COSTS = ("quantum_runs", "pasqal_runs", "atoms", "rounds", "classical_s", "solver_s", "max_atoms", "over_30",
         "atoms_all")


def track_of(method: str) -> str:
    if method in CLASSICAL_METHODS:
        return "classical"
    return "hybrid" if method == "gls-screen" else "oracle" if "exact" in method else "structure"


def gap_pct(size: int, best: int) -> float:
    return 0.0 if best == 0 else 100 * (best - size) / best


_commit = cache(dataset._commit)


def screen_rank(g: nx.Graph, maps: list[dict]) -> list[int]:
    """Hybrid screen: a greedy independent set of the candidate piece plus a greedy one of what it leaves behind."""
    def score(m):
        S = min_degree_greedy(g.subgraph(m), 0)
        return len(S) + len(min_degree_greedy(_remove_closed(g, S), 0))
    return [score(m) for m in maps]


@cache
def _gnn_loader():
    """load_gnn, with torch set to one thread, the first time a process needs a graph network."""
    # Imported here, not at the top: torch and lightgbm cannot share a process on macOS (OMP error or segfault), and
    # the linear and LightGBM methods should not need torch.
    import torch
    from .gnn import load_gnn
    torch.set_num_threads(1)
    return load_gnn


@cache
def _model(path: Path):
    """Each model file is read once per process."""
    if path.suffix == ".pt":
        return _gnn_loader()(path)
    return load_theta(path) if path.name.startswith("cma-") else load(path)


def parse_method(name: str) -> list[tuple[str, str]]:
    """"lin-h2h4+gin-h5" -> [("lin", "h2"), ("lin", "h4"), ("gin", "h5")]: one (kind, hook) per hook, in the order
    written. A trailing "mcts" or "mcts-c<float>" part is dropped, and then h2, h4 and h5 must all be given. Raises
    ValueError on an unknown kind, a malformed or repeated hook, lgb with h5 (there is no LightGBM value model), or
    exact with any hook but h5."""
    out, parts = [], name.split("+")
    searching = SEARCH.fullmatch(parts[-1]) is not None
    for part in parts[:-1] if searching else parts:
        kind, _, hooks = part.partition("-")
        if kind != "exact" and kind not in KIND_EXT:
            raise ValueError(f"{name!r}: unknown model kind {kind!r}")
        found = re.fullmatch(r"(h2)?(h4)?(h5)?", hooks)
        if not hooks or not found:
            raise ValueError(f"{name!r}: hooks must be one or more of h2, h4, h5 in that order, got {hooks!r}")
        for hook in filter(None, found.groups()):
            if (kind == "lgb" and hook == "h5") or (kind == "exact" and hook != "h5"):
                raise ValueError(f"{name!r}: no {kind} model for {hook}")
            if hook in (h for _, h in out):
                raise ValueError(f"{name!r}: {hook} given twice")
            out.append((kind, hook))
    if searching and {h for _, h in out} != {"h2", "h4", "h5"}:
        raise ValueError(f"{name!r}: tree search needs h2, h4 and h5 hooks")
    return out


def search_of(name: str) -> float | None:
    """The exploration constant of a tree-search name (1.0 for plain "mcts"), None for any other name."""
    found = SEARCH.fullmatch(name.split("+")[-1])
    return None if found is None else float(found[1] or 1.0)


def known_method(name: str) -> bool:
    if name in BEAM_METHODS or name in CLASSICAL_METHODS:
        return True
    try:
        return bool(parse_method(name))
    except ValueError:
        return False


def exact_value(graphs: list[nx.Graph]) -> list[int]:
    """The true independence number of each graph."""
    return [len(exact_mis(g, 10.0, workers=1)[0]) for g in graphs]


def _hook(kind: str, hook: str, d: Path):
    """The (run_code_bounded keyword, callable) for one hook of one model kind."""
    keyword = {"h2": "score", "h4": "rank", "h5": "value"}[hook]
    if kind == "exact":
        return keyword, exact_value
    model = _model(d / f"{kind}-{hook}{KIND_EXT[kind]}")
    if kind in GNN_KINDS:
        from . import gnn     # torch is already loaded: the model above is a .pt file
        head = {"h2": "vertex", "h4": "piece", "h5": "value"}[hook]
        if model.head != head:
            raise ValueError(f"{kind}-{hook}.pt has a {model.head!r} head, but hook {hook} needs a {head!r} head")
        return keyword, {"h2": gnn.gnn_score, "h4": gnn.gnn_rank, "h5": gnn.gnn_value}[hook](model)
    return keyword, {"h2": h2_score, "h4": h4_rank, "h5": linear_value}[hook](model)


def method_hooks(name: str, models_dir) -> dict:
    """The score, rank and value keywords of a composed name, tree-search names included."""
    d = Path(models_dir or "models")
    return dict(_hook(kind, hook, d) for kind, hook in parse_method(name))


def method_runner(name: str, models_dir, max_branches: int = 1_000) -> Callable[[nx.Graph, int, int, int, int], RunResult]:
    """run(graph, k, s, l, seed) for a beam method."""
    if not known_method(name) or name in CLASSICAL_METHODS:
        raise ValueError(f"unknown beam method {name!r}")
    if search_of(name) is not None:   # l is the budget of counted solves, and max_branches does not apply
        hooks = method_hooks(name, models_dir)
        return lambda G, k, s, l, seed: run_mcts(G, k, s, l, c=search_of(name), seed=seed, **hooks)
    if name == "gls-code":
        return lambda G, k, s, l, seed: run_code_faithful(G, k, s, l, seed=seed, max_branches=max_branches)
    if name == "gls-paper":
        return lambda G, k, s, l, seed: run_paper_faithful(G, k, s, l, seed=seed, max_branches=max_branches)
    d = Path(models_dir or "models")
    hooks = {}
    if name == "gls-screen":
        hooks["rank"] = screen_rank
    elif name == "cma-h2":
        hooks["score"] = theta_score(_model(d / "cma-h2.json"))
    elif name != "gls":
        hooks = method_hooks(name, models_dir)
    return lambda G, k, s, l, seed: run_code_bounded(G, k, s, l, seed=seed, max_branches=max_branches, **hooks)


def run_instance(family, params, n, seed, methods, widths, restarts, time_limit_s, models_dir=None,
                 max_branches=1_000, cpsat_workers=1) -> list[dict]:
    G = make(family, n, seed, **params)
    rows, proven = [], None

    def record(method, k, s, l, r, solution, costs):
        rows.append(dict(method=method, track=track_of(method), k=k, s=s, l=l, restarts=r,
                         piece_solver=None if k is None else "exact-top-s", solution=solution, **costs))

    def classical(method, solve, r):
        t = time.perf_counter()
        S = solve()
        record(method, None, None, None, r, S, dict(dict.fromkeys(COSTS, 0), classical_s=time.perf_counter() - t,
                                                    solver_s=0.0))

    for m in methods:
        if m == "greedy-pasqal":
            classical(m, lambda: frozenset(weighted_greedy_independent_set(G)), 1)
        elif m == "greedy-mindeg":
            classical(m, lambda: max((min_degree_greedy(G, i) for i in range(MINDEG_SEEDS)), key=len), MINDEG_SEEDS)
        elif m == "exact":
            def solve():
                nonlocal proven
                S, proven = exact_mis(G, time_limit_s, workers=cpsat_workers)
                return S
            classical(m, solve, 1)
        else:
            run = method_runner(m, models_dir, max_branches)
            for k, s, l in widths:
                for r in restarts if m.startswith("gls") else [1]:
                    try:
                        res = best_of_restarts(lambda i: run(G, k, s, l, i), r)
                    except RuntimeError as e:
                        if not str(e).startswith("beam exploded"):
                            raise
                        record(m, k, s, l, r, None, dict.fromkeys(COSTS))
                        continue
                    record(m, k, s, l, r, res.solution, dict(
                        quantum_runs=res.quantum_runs, pasqal_runs=res.pasqal_runs, atoms=res.atoms,
                        rounds=res.rounds, classical_s=res.classical_s, solver_s=res.solver_s,
                        max_atoms=max(res.run_sizes, default=0), over_30=sum(x > 30 for x in res.run_sizes),
                        atoms_all=sum(res.run_sizes)))

    best = max((len(r["solution"]) for r in rows if r["solution"] is not None), default=None)
    for r in rows:
        S = r.pop("solution")
        if S is not None and not is_independent(G, S):
            raise AssertionError(f"{r['method']} returned a non-independent set")
        r.update(family=family, params=params, n=n, p=params.get("p"), seed=seed, lattice="fresnel_61",
                 includes_classical_time=True, small=SMALL, blockade=BLOCKADE_UM, max_branches=max_branches,
                 time_limit_s=time_limit_s, commit=_commit(), size=None if S is None else len(S), best_known=best,
                 proven=proven, gap_pct=None if S is None else gap_pct(len(S), best), exploded=S is None)
    return rows


def parse_params(text: str) -> dict:
    """"p=0.25,m=2" -> {"p": 0.25, "m": 2}: a value that parses as an int is an int, otherwise a float."""
    out = {}
    for item in filter(None, text.split(",")):
        key, value = item.split("=")
        try:
            out[key] = int(value)
        except ValueError:
            out[key] = float(value)
    return out


def grid(name: str) -> list[tuple]:
    """(family, params, n, seed) of every instance in a named grid."""
    if name == "test":
        return [(f, p, n, s) for f, p in TRAIN_SETTINGS + HELD_OUT_SETTINGS for n in TEST_SIZES for s in TEST_SEEDS]
    if name == "val":
        return [(f, p, 100, s) for f, p in TRAIN_SETTINGS for s in VAL_SEEDS]
    if name == "sel":
        return [(f, p, n, s) for f, p in TRAIN_SETTINGS for n in SEL_SIZES for s in SEL_SEEDS]
    return [(f, p, BIG_N, s) for f, p in BIG_SETTINGS for s in TEST_SEEDS[:10]]


def shard(items: list, spec: str | None) -> list:
    """Keep the items at positions j with j % N == I for spec "I/N"; None keeps all of them."""
    if spec is None:
        return items
    try:
        i, n = map(int, spec.split("/"))
    except ValueError:
        raise ValueError(f"shard must look like I/N, got {spec!r}") from None
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"shard {spec!r}: need N >= 1 and 0 <= I < N")
    return items[i::n]


def _instance_key(family, params, n, seed) -> tuple:
    return family, json.dumps(params, sort_keys=True), n, seed


def _job(job: tuple, **kw) -> list[dict]:
    family, params, n, seed, methods = job
    return run_instance(family, params, n, seed, methods, **kw)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["test", "val", "sel", "big"])
    ap.add_argument("--family")
    ap.add_argument("--params", default="", help='e.g. "p=0.25" or "blocks=4,p_in=0.3,p_out=0.03"')
    ap.add_argument("--n", type=int)
    ap.add_argument("--seeds", help="inclusive range a-b")
    ap.add_argument("--methods", default=",".join([m for m in BEAM_METHODS if m not in ORACLE_METHODS]
                                                  + CLASSICAL_METHODS), help="the oracle methods run only when named")
    ap.add_argument("--widths", default="1,1,1;2,2,5;4,2,10;8,3,30", help="k,s,l triples separated by ;")
    ap.add_argument("--restarts", default="1,4", help="restart counts for the gls methods")
    ap.add_argument("--models", default="models")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cpsat-workers", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", help="I/N: run only the instances at positions j with j % N == I, e.g. 0/3")
    a = ap.parse_args(argv)

    methods = a.methods.split(",")
    unknown = {m for m in methods if not known_method(m)}
    if unknown:
        ap.error(f"unknown methods: {', '.join(sorted(unknown))}")
    if a.grid:
        instances = grid(a.grid)
    elif a.family and a.n and a.seeds:
        lo, hi = map(int, a.seeds.split("-"))
        instances = [(a.family, parse_params(a.params), a.n, s) for s in range(lo, hi + 1)]
    else:
        ap.error("give --grid, or --family with --n and --seeds")
    time_limit = 600.0 if a.grid == "big" else a.time_limit
    instances = shard(instances, a.shard)
    if a.shard:
        print(f"shard {a.shard}: {len(instances)} instances kept", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        text = out.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping line {i} of {out}: not valid JSON", file=sys.stderr)
                continue
            done.add((*_instance_key(r["family"], r["params"], r["n"], r["seed"]), r["method"]))
        if text and not text.endswith("\n"):     # a cut-off last line: start new records on a line of their own
            with out.open("a") as fh:
                fh.write("\n")
    jobs = []
    for f, p, n, s in instances:
        missing = [m for m in methods if (*_instance_key(f, p, n, s), m) not in done]
        if missing:
            jobs.append((f, p, n, s, missing))
    print(f"{len(jobs)} instances to run, {len(instances) - len(jobs)} already done in {out}", flush=True)

    run = partial(_job, widths=[tuple(map(int, w.split(","))) for w in a.widths.split(";")],
                  restarts=[int(r) for r in a.restarts.split(",")], time_limit_s=time_limit, models_dir=a.models,
                  cpsat_workers=a.cpsat_workers)
    written = 0
    with Pool(a.workers) if a.workers > 1 else nullcontext() as pool, out.open("a") as fh:
        for recs in pool.imap_unordered(run, jobs) if pool else map(run, jobs):
            fh.write("".join(json.dumps(r) + "\n" for r in recs))
            fh.flush()
            written += len(recs)
            r = recs[0]
            print(f"{r['family']} {json.dumps(r['params'], sort_keys=True)} n={r['n']} seed={r['seed']}: "
                  f"{len(recs)} records", flush=True)
    print(f"wrote {written} records to {out}")


if __name__ == "__main__":
    main()
