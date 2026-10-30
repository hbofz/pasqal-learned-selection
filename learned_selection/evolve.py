"""GLS's own growth score with three tunable numbers (target degree, reward for placed neighbours, reward for raw
degree), tuned by CMA-ES against the mean final gap to the optimum. It tests whether GLS's fixed rule is simply
mistuned.

    uv run python -m learned_selection.evolve --workers 12 --generations 30 --popsize 8 --out models/cma-h2.json
"""
import argparse, json, os, time
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path

import cma

from .bench import gap_pct
from .engine import run_code_bounded
from .instances import TRAIN_SETTINGS, VAL_SEEDS, make
from .models import GLS_THETA, decode, theta_score
from .solvers import exact_mis

EVOLVE_SEEDS = range(5)


def _gap(job: tuple) -> float:
    """Gap of one run. The job is plain data, so it pickles; the score is built here."""
    theta, G, best, k, s, l = job
    return gap_pct(run_code_bounded(G, k, s, l, score=theta_score(theta)).size, best)


def _optimum(G) -> int:
    return len(exact_mis(G, 60, workers=1)[0])


def _pool(workers: int):
    return Pool(workers) if workers > 1 else nullcontext()


def _map(pool, f, jobs) -> list:
    return pool.map(f, jobs, chunksize=1) if pool else list(map(f, jobs))


def _objectives(pool, thetas: list, instances: list, optima: list, widths: tuple) -> list[float]:
    """The objective of every theta, with all (theta, instance) runs spread over the pool at once."""
    gaps = _map(pool, _gap, [(t, G, best, *widths) for t in thetas for G, best in zip(instances, optima)])
    n = len(instances)
    return [sum(gaps[i * n:(i + 1) * n]) / n for i in range(len(thetas))]


def objective(theta, instances, optima, k=4, s=2, l=10) -> float:
    """Mean gap (percent) of the code loop run with theta's growth score, against the given optima."""
    return _objectives(None, [theta], instances, optima, (k, s, l))[0]


def evolve(instances, optima, generations, popsize, workers, k=4, s=2, l=10) -> tuple[tuple, float, list[float]]:
    """CMA-ES over decoded thetas, started at GLS_THETA. Returns the best theta seen (GLS_THETA included), its
    objective and the best objective of each generation. Runs exactly `generations` rounds: pycma would stop
    after one flat generation, and these objectives are discrete."""
    es = cma.CMAEvolutionStrategy([0, 0, 0], 0.5, {"seed": 1, "popsize": popsize, "verbose": -9})
    with _pool(workers) as pool:
        def evaluate(thetas):
            return _objectives(pool, thetas, instances, optima, (k, s, l))

        best_theta, best = GLS_THETA, evaluate([GLS_THETA])[0]
        history = []
        for _ in range(generations):
            xs = es.ask()
            thetas = [decode(x) for x in xs]
            values = evaluate(thetas)
            es.tell(xs, values)
            i = min(range(len(values)), key=values.__getitem__)
            history.append(values[i])
            if values[i] < best:
                best_theta, best = thetas[i], values[i]
    return best_theta, best, history


def _cpu_s() -> float:
    """CPU seconds used by this process and the children it has waited for."""
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    train = [make(fam, 100, seed, **params) for fam, params in TRAIN_SETTINGS for seed in EVOLVE_SEEDS]
    val = [make(fam, 100, seed, **params) for fam, params in TRAIN_SETTINGS for seed in VAL_SEEDS]
    with _pool(a.workers) as pool:
        optima = _map(pool, _optimum, train + val)   # the found set's size, proven or not
    train_optima, val_optima = optima[:len(train)], optima[len(train):]

    start = _cpu_s()
    theta, best, history = evolve(train, train_optima, a.generations, a.popsize, a.workers)
    cpu_s = _cpu_s() - start
    for i, value in enumerate(history, 1):
        print(f"generation {i:3}: best mean gap {value:.3f}%")

    with _pool(a.workers) as pool:
        val_gap_gls, val_gap_theta = _objectives(pool, [GLS_THETA, theta], val, val_optima, (4, 2, 10))
    print(f"best theta {theta}, training mean gap {best:.3f}%")
    print(f"validation mean gap: GLS {val_gap_gls:.3f}%, tuned {val_gap_theta:.3f}%")
    print(f"evolution cpu seconds: {cpu_s:.0f}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(theta=list(theta), objective=best, history=history, cpu_s=cpu_s,
                                   val_gap_gls=val_gap_gls, val_gap_theta=val_gap_theta)))


if __name__ == "__main__":
    main()
