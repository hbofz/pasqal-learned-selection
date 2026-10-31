"""Per-instance selection of a method and its widths from graph structure alone.

`bandit-i` is full-information regression, not a bandit: every menu arm was run on every training instance once,
offline, and one ridge per arm predicts the reward `size - c * quantum_runs` from the graph's features. Deployed, it
reads the features, runs the arm of largest predicted reward for real, and that run's counts are the record's.

    uv run python -m learned_selection.bandit fit --records 'results/train*.jsonl' --out models/bandit-i.json

This module never imports torch: `bench` imports it lazily, and it reaches the arms' models through
`bench.method_runner`.
"""
import argparse, glob, json, os
from pathlib import Path

import numpy as np

from . import dataset
from .bench import method_runner
from .curves import load
from .features import graph_features
from .instances import TRAIN_SEEDS, TRAIN_SETTINGS, make

MENU = [("bld-h3+pg-h4+dgin-h5", 1, 1, 1), ("bld-h3+pg-h4+dgin-h5", 2, 2, 5),
        ("bld-h3+pg-h4+dgin-h5", 4, 2, 10), ("bld-h3+pg-h4+dgin-h5", 8, 3, 30),
        ("bld-h3+sage-h4+dsage-h5+mcts-c0.5", 4, 2, 6), ("bld-h3+sage-h4+dsage-h5+mcts-c0.5", 4, 2, 25),
        ("bld-h3+sage-h4+dsage-h5+mcts-c0.5", 4, 2, 50), ("bld-h3+sage-h4+dsage-h5+mcts-c0.5", 4, 2, 100)]
COSTS = (0.001, 0.003, 0.01, 0.03)
LOGGED = (0, 1, 3, 5, 6)     # n, edges, caro_wei, deg_max, deg_mean among graph_features


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


def round_runner(models_dir):
    raise NotImplementedError("bandit-r is not written yet")


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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    fit = ap.add_subparsers(dest="command", required=True).add_parser("fit")
    fit.add_argument("--records", nargs="+", required=True, help="bench record files or globs, training seeds only")
    fit.add_argument("--out", required=True)
    _fit(ap.parse_args(argv))


if __name__ == "__main__":
    main()
