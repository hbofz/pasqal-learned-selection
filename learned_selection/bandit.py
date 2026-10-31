"""Per-instance selection of a method and its widths from graph structure alone.

`bandit-i` is full-information regression, not a bandit: every menu arm was run on every training instance once,
offline, and one ridge per arm predicts the reward `size - c * quantum_runs` from the graph's features. Deployed, it
reads the features, runs the arm of largest predicted reward for real, and that run's counts are the record's.

    uv run python -m learned_selection.bandit fit --records 'results/train*.jsonl' --out models/bandit-i.json

`bandit-r` picks (k, s, beam) again at the start of every round of one beam over the policy-gradient stack, with a
linear bandit per arm on eight numbers of the current level. An episode's reward is the final size minus c times the
quantum runs, less the stack's predicted value of the root graph, credited to every round's choice.

    uv run python -m learned_selection.bandit round --cost 0.01 --out cache/bandit-r --workers 18

This module never imports torch at the top: `bench` imports it lazily, and it reaches the models through
`bench.method_runner` and `bench.method_hooks`.
"""
import argparse, functools, glob, json, math, os, random, time
from pathlib import Path

import networkx as nx
import numpy as np

from . import dataset
from .bench import method_hooks, method_runner
from .curves import load
from .engine import exact_piece_solver, gls_generator, run_beam
from .features import graph_features
from .instances import TRAIN_SEEDS, TRAIN_SETTINGS, TRAIN_SIZES, make

MENU = [("bld-h3+pg-h4+dgin-h5", 1, 1, 1), ("bld-h3+pg-h4+dgin-h5", 2, 2, 5),
        ("bld-h3+pg-h4+dgin-h5", 4, 2, 10), ("bld-h3+pg-h4+dgin-h5", 8, 3, 30),
        ("bld-h3+sage-h4+dsage-h5+mcts-c0.5", 4, 2, 6), ("bld-h3+sage-h4+dsage-h5+mcts-c0.5", 4, 2, 25),
        ("bld-h3+sage-h4+dsage-h5+mcts-c0.5", 4, 2, 50), ("bld-h3+sage-h4+dsage-h5+mcts-c0.5", 4, 2, 100)]
COSTS = (0.001, 0.003, 0.01, 0.03)
LOGGED = (0, 1, 3, 5, 6)     # n, edges, caro_wei, deg_max, deg_mean among graph_features
ROUND_STACK = "bld-h3+pg-h4+dgin-h5"
ROUND_ARMS = ((1, 1, 1), (2, 2, 2), (4, 2, 4), (8, 3, 8))     # (k, s, beam)
ROUND_COSTS = (0.003, 0.01, 0.03)
ROUND_D = 8
FIXED_ARM_L = 9000     # l = FIXED_ARM_L + 1 + j runs arm j in every round, with no theta


def context(G) -> np.ndarray:
    """graph_features, then the log1p of its n, edges, caro_wei, deg_max and deg_mean. Structure only."""
    f = graph_features(G)
    return np.concatenate([f, np.log1p(f[list(LOGGED)])])


def reward_table(paths) -> tuple:
    """(instances, X, sizes, runs) from bench records of the menu arms: the instances (family, params, n, seed) in
    sorted order, their contexts, and sizes and runs of shape (instances, arms) in MENU order. Raises ValueError on
    any record whose seed is not a training seed or whose setting is held out, and on an instance with an arm missing
    or exploded."""
    arms = {arm: j for j, arm in enumerate(MENU)}
    table = {}
    for r in load(paths):
        if r["seed"] not in TRAIN_SEEDS or [r["family"], r["params"]] not in map(list, TRAIN_SETTINGS):
            raise ValueError(f"{r['family']} {r['params']} seed {r['seed']} is not a training instance")
        if (arm := arms.get((r["method"], r["k"], r["s"], r["l"]))) is not None:
            table.setdefault((r["family"], json.dumps(r["params"], sort_keys=True), r["n"], r["seed"]), {})[arm] = r
    instances, X, sizes, runs = [], [], [], []
    for key in sorted(table):
        got = table[key]
        if len(got) < len(MENU) or any(r["exploded"] for r in got.values()):
            raise ValueError(f"{key}: needs every menu arm finished, has {len(got)} of {len(MENU)}, "
                             f"{sum(r['exploded'] for r in got.values())} exploded")
        family, params, n, seed = key
        instances.append((family, json.loads(params), n, seed))
        X.append(context(make(family, n, seed, **json.loads(params))))
        sizes.append([got[j]["size"] for j in range(len(MENU))])
        runs.append([got[j]["quantum_runs"] for j in range(len(MENU))])
    return instances, np.array(X), np.array(sizes, float), np.array(runs, float)


def fit_ridge(X, sizes, runs, cost, lam=1.0) -> dict:
    """One ridge per arm on `size - cost * runs`, on standardised features and a constant that is not shrunk.
    Returns dict(mu, sd, theta), theta of shape (arms, features + 1)."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    penalty = lam * np.diag([1.0] * X.shape[1] + [0.0])
    theta = np.linalg.solve(Z.T @ Z + penalty, Z.T @ (sizes - cost * runs)).T
    return dict(mu=mu, sd=sd, theta=theta)


def pick_arm(model, x) -> int:
    """The arm of largest predicted reward; ties go to the first."""
    return int(np.argmax(model["theta"] @ np.append((np.asarray(x) - model["mu"]) / model["sd"], 1.0)))


def save_model(model_by_l, path) -> None:
    """JSON with the menu, the commit and one model per l (cost c = l / 1000), moved into place whole."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    models = {str(l): {k: np.asarray(v).tolist() for k, v in m.items()} for l, m in model_by_l.items()}
    tmp.write_text(json.dumps(dict(menu=MENU, commit=dataset._commit(), models=models)))
    os.replace(tmp, path)


def load_model(path) -> dict:
    """{l: model}. Raises ValueError if the file was fitted for another menu."""
    d = json.loads(Path(path).read_text())
    if d["menu"] != json.loads(json.dumps(MENU)):
        raise ValueError(f"{path} was fitted for another menu than this one")
    return {int(l): {k: np.array(v) for k, v in m.items()} for l, m in d["models"].items()}


def instance_runner(models_dir):
    """run(graph, k, s, l, seed) for bandit-i. The ridge models and every menu arm's runner are loaded here, so a
    missing file fails now. k and s are ignored, and c = l / 1000 picks the ridge."""
    d = Path(models_dir or "models")
    models = load_model(d / "bandit-i.json")
    runners = {m: method_runner(m, models_dir) for m in dict.fromkeys(m for m, *_ in MENU)}

    def run(G, k, s, l, seed):
        if l not in models:
            raise ValueError(f"bandit-i has models for l={sorted(models)}, not l={l}")
        method, ak, as_, al = MENU[pick_arm(models[l], context(G))]
        return runners[method](G, ak, as_, al, seed)
    return run


def round_context(level, n0, runs, rnd) -> np.ndarray:
    """Eight numbers of the current level of (graph, partial solution) branches, each scaled by the input size n0: a
    constant, mean size, mean density, log of the branch count, the round, solves so far, mean partial size, and size
    times density. Zeros for an empty level. Reads sizes and densities only, no independent set."""
    if not level:
        return np.zeros(ROUND_D)
    n0 = max(1, n0)
    size, density = np.mean([len(g) for g, _ in level]) / n0, np.mean([nx.density(g) for g, _ in level])
    partial = np.mean([len(s) for _, s in level]) / n0
    return np.array([1, size, density, math.log1p(len(level)) / 4, min(rnd, 60) / 60, min(runs, 2000) / n0 / 4,
                     partial, size * density])


def run_round_beam(G, choose, hooks, seed=0, small=5) -> tuple:
    """One beam whose arm of ROUND_ARMS comes from choose(context) at the start of each round, never on an empty
    level. Returns (RunResult, [(context, arm)] per round). `hooks` holds whichever of score, build, rank and value
    the stack has; the first three go to the generator, and value ranks the level by partial size plus its value.
    The running count of solves in the context is the solves made so far, with the small branches kept for the next
    round counted already, so the last context's count is the run's quantum_runs."""
    gen = gls_generator(seed=seed, **{h: hooks[h] for h in ("score", "build", "rank") if h in hooks})
    value, n0 = hooks.get("value"), len(G)
    state = dict(arm=0, runs=int(0 < n0 <= small), rnd=0, seen=[])

    def start(level):
        x = round_context(level, n0, state["runs"], state["rnd"])
        state["arm"] = choose(x)
        state["seen"].append((x, state["arm"]))

    def generate(g, _):
        maps = gen(g, ROUND_ARMS[state["arm"]][0])
        state["runs"] += len(maps)
        return maps

    def solve(piece, m, _):
        return exact_piece_solver(piece, m, ROUND_ARMS[state["arm"]][1])

    def prune(level, _):
        state["rnd"] += 1
        beam = ROUND_ARMS[state["arm"]][2]
        if len(level) > beam:
            keys = [len(s) for _, s in level]
            if value is not None:
                keys = [k + v for k, v in zip(keys, value([g for g, _ in level]), strict=True)]
            level = [level[i] for i in sorted(range(len(level)), key=keys.__getitem__, reverse=True)[:beam]]
        state["runs"] += sum(0 < len(g) <= small for g, _ in level)
        if level:
            start(level)
        return level
    start([(G, [])])
    return run_beam(G, generate, solve, 4, 2, prune, small), state["seen"]


@functools.lru_cache(maxsize=None)
def _round_hooks(stack, models_dir):
    return method_hooks(stack, models_dir)     # loaded once per worker


def round_episode(job: tuple) -> dict:
    """job = (theta, eps, family, params, n, seed, rseed, cost, models_dir). One epsilon-greedy beam on a training
    instance. Returns the reward size - cost * runs, size, runs, [(context, arm)] and v0, the stack's predicted value
    of the root graph. Numpy only, so it crosses a process boundary."""
    theta, eps, family, params, n, seed, rseed, cost, models_dir = job
    if seed not in TRAIN_SEEDS:
        raise ValueError(f"seed {seed} is not a training seed; episodes must not use it")
    hooks, rng, G = _round_hooks(ROUND_STACK, models_dir), np.random.default_rng(rseed), make(family, n, seed, **params)

    def choose(x):
        return int(rng.integers(len(ROUND_ARMS))) if rng.random() < eps else int((theta @ x).argmax())
    res, seen = run_round_beam(G, choose, hooks)
    return dict(reward=res.size - cost * res.quantum_runs, size=res.size, runs=res.quantum_runs, seen=seen,
                v0=float(hooks["value"]([G])[0]))


def update_round(A, b, episodes) -> None:
    """In place, per arm: A += x x^T and b += (reward - v0) x for every (x, arm) of every episode."""
    for e in episodes:
        for x, arm in e["seen"]:
            A[arm] += np.outer(x, x)
            b[arm] += (e["reward"] - e["v0"]) * x


def round_runner(models_dir):
    """run(graph, k, s, l, seed) for bandit-r. The stack's hooks are loaded here, so a missing file fails now. The arm
    each round is greedy on `{models_dir}/bandit-r/theta_{l}.json`, read when first needed (c = l / 1000). A larger l
    holds arm l - FIXED_ARM_L - 1 in every round and reads no theta. k and s are ignored."""
    d = Path(models_dir or "models")
    hooks, thetas = method_hooks(ROUND_STACK, d), {}

    def run(G, k, s, l, seed):
        if l > FIXED_ARM_L:
            arm = l - FIXED_ARM_L - 1
            choose = lambda x: arm
        else:
            if l not in thetas:
                thetas[l] = np.array(json.loads((d / "bandit-r" / f"theta_{l}.json").read_text())["theta"])
            choose = lambda x: int((thetas[l] @ x).argmax())
        return run_round_beam(G, choose, hooks, seed)[0]
    return run


def _fit(a) -> None:
    paths = sorted(p for pattern in a.records for p in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no records match {a.records}")
    instances, X, sizes, runs = reward_table(paths)
    print(f"{len(instances)} training instances, {len(MENU)} arms")
    models = {}
    for c in COSTS:
        models[round(1000 * c)] = model = fit_ridge(X, sizes, runs, c)
        reward = sizes - c * runs
        picked = np.array([pick_arm(model, x) for x in X])
        got = reward[np.arange(len(X)), picked].mean()
        best = reward.mean(0).argmax()
        shares = " ".join(f"{np.mean(picked == j):.2f}" for j in range(len(MENU)))
        print(f"c={c}: arm shares {shares}; training reward {got:.3f} against best fixed arm {best} "
              f"at {reward.mean(0)[best]:.3f}")
    save_model(models, a.out)
    print(f"wrote {a.out}")


def _round(a) -> None:
    from .az import _cpu_s
    from .builder import _close, _map, _open_pool
    out, arms = Path(a.out), range(len(ROUND_ARMS))
    out.mkdir(parents=True, exist_ok=True)
    instances = [(f, p, n, s) for f, p in TRAIN_SETTINGS for n in TRAIN_SIZES for s in TRAIN_SEEDS]
    A, b = np.array([np.eye(ROUND_D) for _ in arms]), np.zeros((len(ROUND_ARMS), ROUND_D))
    solve = lambda: np.array([np.linalg.solve(A[j], b[j]) for j in arms])
    cpu0, pool = _cpu_s(), _open_pool(a.workers)
    for t in range(1, a.iters + 1):
        step, eps, base = time.perf_counter(), max(0.1, 1 - t / (0.7 * a.iters)), a.seed * 100003 + t
        batch = random.Random(base).sample(instances, a.instances)
        theta = solve()
        results = _map(pool, round_episode, [(theta, eps, f, p, n, s, base * 1000 + i, a.cost, a.models)
                                             for i, (f, p, n, s) in enumerate(batch)])
        update_round(A, b, results)
        row = dict(t=t, eps=eps, size=float(np.mean([r["size"] for r in results])),
                   runs=float(np.mean([r["runs"] for r in results])),
                   reward=float(np.mean([r["reward"] for r in results])), seconds=time.perf_counter() - step)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"it {t}: eps {eps:.2f} size {row['size']:.2f} runs {row['runs']:.1f} reward {row['reward']:.3f} "
              f"{row['seconds']:.1f} s", flush=True)
    _close(pool)     # so the CPU total includes the workers
    l, cpu = round(1000 * a.cost), _cpu_s() - cpu0
    path = out / f"theta_{l}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(theta=solve().tolist(), cost=a.cost, commit=dataset._commit(), cpu_s=cpu)))
    os.replace(tmp, path)
    print(f"wrote {path}, cpu {cpu:.0f} s", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    commands = ap.add_subparsers(dest="command", required=True)
    fit, rnd = commands.add_parser("fit"), commands.add_parser("round")
    fit.add_argument("--records", nargs="+", required=True, help="bench record files or globs, training seeds only")
    fit.add_argument("--out", required=True)
    rnd.add_argument("--cost", type=float, required=True, help="cost per quantum run; the model is theta_{1000 * cost}")
    rnd.add_argument("--iters", type=int, default=100)
    rnd.add_argument("--instances", type=int, default=96)
    rnd.add_argument("--seed", type=int, default=0)
    rnd.add_argument("--models", default="models")
    rnd.add_argument("--out", required=True)
    rnd.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    (_fit if a.command == "fit" else _round)(a)


if __name__ == "__main__":
    main()
