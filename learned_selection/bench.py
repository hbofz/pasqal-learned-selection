"""Benchmark runner: every method on the same Erdos-Renyi instances, one JSON record per (instance, method).

    uv run python -m learned_selection.bench --n 100 --p 0.25 --seeds 0-9 --k 4 --s 2 --l 10 --out results/er.jsonl
"""
import argparse, json, statistics, time
from pathlib import Path

from .engine import best_of_restarts, run_code_faithful, run_paper_faithful
from .instances import erdos_renyi
from .solvers import exact_mis, is_independent, min_degree_greedy
from .vendored import weighted_greedy_independent_set


def gap_pct(size: int, best: int) -> float:
    return 0.0 if best == 0 else 100 * (best - size) / best


def run_instance(n, p, seed, k, s, l, restarts, time_limit_s) -> list[dict]:
    G = erdos_renyi(n, p, seed)
    rows = []

    def beam(method, run):
        r = best_of_restarts(run, restarts)
        rows.append(dict(method=method, k=k, s=s, l=l, restarts=restarts, piece_solver="exact-top-s",
                         solution=r.solution, quantum_runs=r.quantum_runs, pasqal_runs=r.pasqal_runs,
                         atoms=r.atoms, rounds=r.rounds, classical_s=r.classical_s, solver_s=r.solver_s))

    def classical(method, solve, restarts_used):
        t = time.perf_counter()
        S = solve()
        rows.append(dict(method=method, k=None, s=None, l=None, restarts=restarts_used, piece_solver=None,
                         solution=frozenset(S), quantum_runs=0, pasqal_runs=0, atoms=0, rounds=0,
                         classical_s=time.perf_counter() - t, solver_s=0.0))

    beam("gls-code", lambda i: run_code_faithful(G, k, s, l, seed=i))
    beam("gls-paper", lambda i: run_paper_faithful(G, k, s, l, seed=i))
    classical("greedy-pasqal", lambda: weighted_greedy_independent_set(G), 1)
    classical("greedy-mindeg", lambda: max((min_degree_greedy(G, i) for i in range(restarts * 20)), key=len),
              restarts * 20)
    proven = {}
    classical("exact", lambda: proven.setdefault("r", exact_mis(G, time_limit_s))[0], 1)

    best = max(len(r["solution"]) for r in rows)
    for r in rows:
        S = r.pop("solution")
        assert is_independent(G, S), f"{r['method']} returned a non-independent set"
        r.update(family="er", n=n, p=p, seed=seed, lattice="fresnel_61", includes_classical_time=True,
                 size=len(S), best_known=best, proven=proven["r"][1], gap_pct=gap_pct(len(S), best))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--seeds", default="0-9", help="inclusive range a-b")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--s", type=int, default=2)
    ap.add_argument("--l", type=int, default=10)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("a") as f:
        for seed in range(lo, hi + 1):
            for r in run_instance(a.n, a.p, seed, a.k, a.s, a.l, a.restarts, a.time_limit):
                f.write(json.dumps(r) + "\n")
                rows.append(r)
            print(f"seed {seed} done", flush=True)
    print(f"\nER n={a.n} p={a.p} seeds {a.seeds}, widths k={a.k} s={a.s} l={a.l}, restarts={a.restarts}, "
          f"all proven optimal: {all(r['proven'] for r in rows)}")
    print(f"{'method':15} {'mean gap %':>10} {'quantum runs':>13} {'pasqal runs':>12} {'classical s':>12}")
    for m in dict.fromkeys(r["method"] for r in rows):
        ms = [r for r in rows if r["method"] == m]
        mean = lambda key: statistics.mean(r[key] for r in ms)
        print(f"{m:15} {mean('gap_pct'):10.2f} {mean('quantum_runs'):13.1f} {mean('pasqal_runs'):12.1f} "
              f"{mean('classical_s'):12.2f}")


if __name__ == "__main__":
    main()
