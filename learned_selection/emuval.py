"""Whole decompositions on the emulator: the test grid's best methods, each piece solved by a cloud emulator.

    uv run python -m learned_selection.emuval --solver cloud --seeds 1020-1025 --out results/ev/cloud.jsonl \\
        --cache results/ev/counts.jsonl --calls results/ev/calls.jsonl
    uv run python -m learned_selection.emuval --solver cloud --out results/ev/cloud.jsonl --cache results/ev/counts.jsonl --replay 10

One bench record per job, appended as the job finishes, so a rerun skips the jobs in --out and a kill loses only the
jobs in flight. A job that raises is reported and skipped; the exit code is 1 if any did. With --calls, each piece
solve also writes a line (family, params, n, seed, method, k, s, l, key, atoms, returned, asked), just before its
job's record: a kill between the two leaves lines without a record, and the rerun writes them again.
"""
import argparse, json, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .bench import _instance_key, method_runner, run_instance
from .cloud import CloudPieceSolver, CountsCache, client, uniform_piece_solver, virtual_device
from .engine import exact_piece_solver
from .instances import HELD_OUT_SETTINGS, TEST_SIZES, TRAIN_SETTINGS

RUNS = [("bld-h3+dgin-h4+dgin-h5", 4, 2, 10), ("bandit-r", 4, 2, 10), ("bandit-r", 4, 2, 9002),
        ("bld-h3+sage-h4+dsage-h5+mcts-c0.5", 4, 2, 100), ("lin-h2h4+sage-h5", 4, 2, 10), ("gls", 4, 2, 10)]
LABELS = {"cloud": "emu-mps", "uniform": "uniform-repair", "exact": "exact-top-s"}


def jobs(seeds, methods=None, settings=None, sizes=TEST_SIZES) -> list[tuple]:
    """(family, params, n, seed, method, k, s, l): seeds ascending, then settings x sizes, then RUNS."""
    settings = TRAIN_SETTINGS + HELD_OUT_SETTINGS if settings is None else settings
    return [(f, p, n, seed, m, k, s, l) for seed in sorted(seeds) for f, p in settings for n in sizes
            for m, k, s, l in RUNS if methods is None or m in methods]


def _key(family, params, n, seed, method, k, s, l) -> tuple:
    return (*_instance_key(family, params, n, seed), method, k, s, l)


def _fresh_line(path: Path) -> None:
    """After a cut-off last line, start the next record on a line of its own."""
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")


def _records(out: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(out.read_text().splitlines() if out.exists() else [], 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                print(f"skipping line {i} of {out}: not valid JSON", file=sys.stderr)
    return rows


def done_keys(out) -> set:
    out = Path(out)
    rows = _records(out)
    _fresh_line(out)
    return {_key(r["family"], r["params"], r["n"], r["seed"], r["method"], r["k"], r["s"], r["l"]) for r in rows}


def run_grid(todo, solve_for, label, out, workers, models="models", calls=None) -> int:
    """Run every job on `workers` threads and append its record to out. Returns the number of failed jobs."""
    out, lock = Path(out), threading.Lock()
    for p in filter(None, [out, calls and Path(calls)]):
        p.parent.mkdir(parents=True, exist_ok=True)
        _fresh_line(p)
    for m in dict.fromkeys(j[4] for j in todo):
        method_runner(m, models)                    # hooks load once, here, before the threads start

    def one(job) -> int:
        family, params, n, seed, method, k, s, l = job
        lines, solver = [], solve_for(job)
        ask = getattr(solver, "answer", None)

        def solve(piece, mapping, want):
            top, key = ask(piece, mapping, want) if ask else (solver(piece, mapping, want), None)
            lines.append(json.dumps(dict(family=family, params=params, n=n, seed=seed, method=method, k=k, s=s, l=l, key=key,
                                         atoms=len(piece), returned=len(top), asked=want)) + "\n")
            return top
        try:
            recs = run_instance(family, params, n, seed, [method], widths=[(k, s, l)], restarts=[1], time_limit_s=120.0,
                                models_dir=models, solve=solve, piece_solver=label)
        except Exception as e:
            print(f"FAILED {job}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return 1
        with lock:
            for path, text in ((calls, "".join(lines)), (out, "".join(json.dumps(r) + "\n" for r in recs))):
                if path and text:
                    with Path(path).open("a") as f:
                        f.write(text)
                        f.flush()
            print(f"{job[:4]} {method} l={l}: size {recs[0]['size']}, {len(lines)} solves", flush=True)
        return 0

    pool = ThreadPoolExecutor(workers)
    try:
        return sum(pool.map(one, todo))
    finally:
        pool.shutdown(cancel_futures=True)         # an interrupt leaves the queued jobs unstarted


def replay(out, cache_path, stride, models="models") -> tuple[int, int]:
    """Rerun every stride-th record of out from the cache alone. Returns (same, differ)."""
    solve = CloudPieceSolver(None, virtual_device(), cache=CountsCache(cache_path))
    same = differ = 0
    for r in _records(Path(out))[::stride]:
        job = (r["family"], r["params"], r["n"], r["seed"], r["method"], r["k"], r["s"], r["l"])
        try:
            (new,) = run_instance(*job[:4], [r["method"]], widths=[job[5:]], restarts=[1], time_limit_s=120.0, models_dir=models,
                                  solve=solve, piece_solver=r["piece_solver"])
        except KeyError as e:
            print(f"DIFFER {job}: no cached answer {e}", flush=True)
            differ += 1
            continue
        fields = ("size", "quantum_runs", "pasqal_runs", "atoms")
        if all(new[f] == r[f] for f in fields):
            same += 1
        else:
            print(f"DIFFER {job}: {[(f, r[f], new[f]) for f in fields if new[f] != r[f]]}", flush=True)
            differ += 1
    return same, differ


def _seeds(text: str) -> list[int]:
    return [x for part in text.split(",") for x in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=list(LABELS), required=True)
    ap.add_argument("--seeds", help="a-b, a single seed, or a comma list of both")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", help="comma list; default all of RUNS")
    ap.add_argument("--cache", help="cloud only: the counts cache")
    ap.add_argument("--calls", help="cloud only: one line per piece solve")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--models", default="models")
    ap.add_argument("--replay", type=int, metavar="STRIDE", help="rerun every STRIDE-th record from the cache, run nothing")
    ap.add_argument("--settings", type=int, help="keep the first N settings")
    ap.add_argument("--sizes", help="e.g. 100,150")
    a = ap.parse_args(argv)
    if a.solver == "cloud" and not (a.cache and a.calls) and not (a.replay and a.cache):
        ap.error("--solver cloud needs --cache and --calls")
    if a.solver != "cloud" and (a.cache or a.calls or a.replay):
        ap.error("--cache, --calls and --replay go with --solver cloud only")
    if a.replay:
        same, differ = replay(a.out, a.cache, a.replay, a.models)
        print(f"replay: {same} same, {differ} differ")
        return int(differ > 0)
    if not a.seeds:
        ap.error("--seeds is required")
    settings = (TRAIN_SETTINGS + HELD_OUT_SETTINGS)[:a.settings]
    sizes = tuple(map(int, a.sizes.split(","))) if a.sizes else TEST_SIZES
    done = done_keys(a.out)
    todo = [j for j in jobs(_seeds(a.seeds), a.methods and a.methods.split(","), settings, sizes) if _key(*j) not in done]
    print(f"{len(todo)} jobs to run", flush=True)
    if a.solver == "cloud":
        solver = CloudPieceSolver(client(), virtual_device(), cache=CountsCache(a.cache), tags=["plan5"])
    else:
        solver = uniform_piece_solver() if a.solver == "uniform" else exact_piece_solver
    failed = run_grid(todo, lambda job: solver, LABELS[a.solver], a.out, a.workers, a.models, a.calls)
    print(f"{failed} of {len(todo)} jobs failed")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
