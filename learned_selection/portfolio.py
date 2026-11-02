"""The portfolio grid: every method on every (rebalance date, theta) market graph, one JSONL record per run.

    uv run python -m learned_selection.portfolio --out results/eval/port6.jsonl --workers 24 --stop-hours 12
    uv run python -m learned_selection.portfolio --solver cloud --dates 2009-01-02 --thetas 0.3,0.5 --methods gls --out E --cache C --calls L
    uv run python -m learned_selection.portfolio --solver cloud --out E --cache C --replay 1

Records hold tickers and statistics, never prices. Primary methods run first over all dates, so a stop by the clock can
only cost a secondary one. A rerun skips what is in --out.
"""
import argparse, json, sys, threading, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from .bench import _commit, method_runner
from .cloud import CloudPieceSolver, CountsCache, client, uniform_piece_solver, virtual_device
from .emuval import LABELS, _fresh_line, _records
from .engine import exact_piece_solver
from .market import THETAS, Market
from .solvers import exact_mis, is_independent, min_degree_greedy

S8, B8, L = "bld-h3+sage-h4+dsage-h5+mcts-c0.5", "bld-h3+dgin-h4+dgin-h5", "lin-h2h4+sage-h5"
RUNS = [("greedy-mindeg", None, None, None), ("exact", None, None, None), ("gls", 4, 2, 10), (S8, 4, 2, 100),
        ("bandit-r", 4, 2, 10), (B8, 4, 2, 10), (L, 4, 2, 10)]
MINDEG_SEEDS, EXACT_LIMIT_S = 20, 120.0
_market = None


def jobs(dates, thetas=THETAS, methods=None) -> list[tuple]:
    """(date, theta, method, k, s, l), method by method in RUNS' order, so the primary ones finish first."""
    return [(d, t, m, k, s, l) for m, k, s, l in RUNS if methods is None or m in methods for d in dates for t in thetas]


def _key(r) -> tuple:
    return r["date"], r["theta"], r["method"], r["k"], r["s"], r["l"]


def run_job(job, market, solve, label, models="models") -> dict:
    date, theta, method, k, s, l = job
    G, names = market.graph(date, theta)
    t = time.perf_counter()
    rec = dict(date=date, theta=theta, method=method, k=k, s=s, l=l, n=len(names), edges=G.number_of_edges(), piece_solver=None if k is None else label,
               commit=_commit(), proven=None, exploded=False, quantum_runs=0, pasqal_runs=0, max_atoms=0)
    if method == "greedy-mindeg":
        S = max((min_degree_greedy(G, i) for i in range(MINDEG_SEEDS)), key=len)
    elif method == "exact":
        S, rec["proven"] = exact_mis(G, EXACT_LIMIT_S, workers=1)
    else:
        try:
            r = method_runner(method, models, solve=solve)(G, k, s, l, 0)
        except RuntimeError as e:
            if not str(e).startswith("beam exploded"):
                raise
            return dict(rec, exploded=True, size=None, tickers=None, seconds=time.perf_counter() - t)
        S = r.solution
        rec.update(quantum_runs=r.quantum_runs, pasqal_runs=r.pasqal_runs, max_atoms=max(r.run_sizes, default=0))
    if not is_independent(G, S):
        raise AssertionError(f"{job} returned a non-independent set")
    return dict(rec, size=len(S), tickers=sorted(names[i] for i in S), seconds=time.perf_counter() - t)


def _init(folder):
    global _market
    _market = Market.load(folder)


def _exact_job(job, label, models):
    return run_job(job, _market, exact_piece_solver, label, models)


def run_cpu(todo, out, folder, workers, stop_at, label, models) -> int:
    failed = 0
    pool = ProcessPoolExecutor(workers, initializer=_init, initargs=(folder,))
    waiting = {}
    it = iter(todo)
    with open(out, "a") as f:
        try:
            while True:
                while len(waiting) < workers and time.time() < stop_at and (job := next(it, None)):
                    waiting[pool.submit(_exact_job, job, label, models)] = job
                if not waiting:
                    break
                done = next(as_completed(waiting))
                job = waiting.pop(done)
                try:
                    f.write(json.dumps(done.result()) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"FAILED {job}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    failed += 1
        finally:
            pool.shutdown(cancel_futures=True)
    left = sum(1 for _ in it)
    if left:
        print(f"CLOCK STOP: {left} jobs not started", flush=True)
    return failed


def run_threads(todo, out, market, solver, label, workers, calls, models) -> int:
    lock = threading.Lock()
    for job in {j[2]: j for j in todo if j[3] is not None}.values():
        method_runner(job[2], models)        # hooks load once, here, before the threads start

    def one(job) -> int:
        lines, ask = [], getattr(solver, "answer", None)

        def solve(piece, mapping, want):
            top, key = ask(piece, mapping, want) if ask else (solver(piece, mapping, want), None)
            lines.append(json.dumps(dict(date=job[0], theta=job[1], method=job[2], key=key, atoms=len(piece), returned=len(top), asked=want)) + "\n")
            return top
        try:
            rec = run_job(job, market, solve, label, models)
        except Exception as e:
            print(f"FAILED {job}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return 1
        with lock:
            if calls and lines:
                with open(calls, "a") as f:
                    f.write("".join(lines))
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return 0
    pool = ThreadPoolExecutor(workers)
    try:
        return sum(pool.map(one, todo))
    finally:
        pool.shutdown(cancel_futures=True)


REPLAY_FIELDS = ("size", "quantum_runs", "pasqal_runs", "max_atoms", "tickers")


def replay(out, cache_path, stride, market, models="models") -> tuple[int, int]:
    """Rerun every stride-th record from the cache alone. Returns (same, differ)."""
    solve = CloudPieceSolver(None, virtual_device(), cache=CountsCache(cache_path))
    same = differ = 0
    for r in _records(Path(out))[::stride]:
        try:
            new = run_job((r["date"], r["theta"], r["method"], r["k"], r["s"], r["l"]), market, solve, r["piece_solver"], models)
        except KeyError as e:
            print(f"DIFFER {_key(r)}: no cached answer {e}", flush=True)
            differ += 1
            continue
        if all(new[f] == r[f] for f in REPLAY_FIELDS):
            same += 1
        else:
            print(f"DIFFER {_key(r)}", flush=True)
            differ += 1
    return same, differ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=list(LABELS), default="exact")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="cache/market")
    ap.add_argument("--dates", help="comma list; default every rebalance date but the last, which only ends a holding")
    ap.add_argument("--thetas", help="comma list; default all of THETAS")
    ap.add_argument("--methods", help="comma list; default all of RUNS")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--models", default="models")
    ap.add_argument("--stop-hours", type=float, help="exact solver only: start no job after this long")
    ap.add_argument("--cache", help="cloud only: the counts cache")
    ap.add_argument("--calls", help="one line per piece solve, threads only")
    ap.add_argument("--replay", type=int, metavar="STRIDE", help="cloud only: rerun every STRIDE-th record from the cache, run nothing")
    a = ap.parse_args(argv)
    if a.solver == "cloud" and not a.cache:
        ap.error("--solver cloud needs --cache")
    market = Market.load(a.data)
    if a.replay:
        same, differ = replay(a.out, a.cache, a.replay, market, a.models)
        print(f"replay: {same} same, {differ} differ")
        return int(differ > 0)
    dates = a.dates.split(",") if a.dates else [str(d.date()) for d in market.rebalance_dates()[:-1]]
    thetas = tuple(map(float, a.thetas.split(","))) if a.thetas else THETAS
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {_key(r) for r in _records(out)}
    _fresh_line(out)
    todo = [j for j in jobs(dates, thetas, a.methods and a.methods.split(",")) if j not in done]
    print(f"{len(todo)} jobs to run, {len(done)} done", flush=True)
    if a.solver == "exact":
        failed = run_cpu(todo, a.out, a.data, a.workers, time.time() + (1e6 if a.stop_hours is None else a.stop_hours) * 3600, LABELS["exact"], a.models)
    else:
        solver = (CloudPieceSolver(client(), virtual_device(), cache=CountsCache(a.cache), tags=["plan6"]) if a.solver == "cloud"
                  else uniform_piece_solver())
        failed = run_threads(todo, a.out, market, solver, LABELS[a.solver], a.workers, a.calls, a.models)
    print(f"{failed} of {len(todo)} jobs failed", flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
