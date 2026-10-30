"""Learned rankers for the two places GLS makes a greedy choice: which vertex to place next while a piece grows
(growth hook) and which candidate pieces to solve (candidate-choice hook). A weighted ridge model and a
LightGBM lambdarank model, both trained on the labelled states from dataset.py. Also the three-number growth
score that evolve.py tunes with CMA-ES.

    uv run python -m learned_selection.models --train cache/train.npz --val cache/val.npz --out models/

Thread count for LightGBM is left to OMP_NUM_THREADS.
"""
import argparse, json, time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .engine import Rank
from .features import PIECE_FEATURES, VERTEX_FEATURES, piece_features, vertex_features
from .growth import VertexScore

MAX_RELEVANCE = 30   # LightGBM's default label gain table covers labels 0..30


@dataclass
class LinearModel:
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.w


def _groups(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(first row, row count, group index of every row) for group ids that are contiguous and non-decreasing."""
    if (np.diff(groups) < 0).any():
        raise ValueError("group ids must be non-decreasing")
    _, starts, inverse, counts = np.unique(groups, return_index=True, return_inverse=True, return_counts=True)
    return starts, counts, inverse


def fit_linear(X: np.ndarray, y: np.ndarray, groups: np.ndarray, weights: np.ndarray | None = None,
               ridge: float = 1e-3) -> LinearModel:
    """Ridge regression on within-group differences, so only the order inside a group matters. `weights` is one
    per row (a state's weight repeated on its rows); each group's rows share it, so a group's total weight is its
    state weight. With no weights every group counts equally."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    starts, counts, inverse = _groups(groups)

    def centred(A):   # A is 2-D: subtract each group's mean row
        return A - (np.add.reduceat(A, starts, axis=0) / counts[:, None])[inverse]

    Z, t = centred((X - mu) / sd), centred(y[:, None].astype(np.float64))[:, 0]
    d = (np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)) / counts[inverse]
    w = np.linalg.solve(Z.T @ (Z * d[:, None]) + ridge * np.eye(X.shape[1]), Z.T @ (d * t))
    return LinearModel(mu, sd, w)


def relevance(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Integer relevance labels for lambdarank: each value minus its group's minimum, capped at 30."""
    starts, _, inverse = _groups(groups)
    return np.minimum(y - np.minimum.reduceat(y, starts)[inverse], MAX_RELEVANCE).astype(np.int64)


def fit_lgbm(X: np.ndarray, rel: np.ndarray, groups: np.ndarray, X_val: np.ndarray, rel_val: np.ndarray,
             groups_val: np.ndarray, eval_at: int, weights: np.ndarray | None = None) -> lgb.Booster:
    """LambdaRank on relevance labels, stopped early on validation NDCG@eval_at."""
    params = dict(objective="lambdarank", metric="ndcg", eval_at=[eval_at], learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=20, deterministic=True, force_row_wise=True, seed=0, verbose=-1)
    return lgb.train(params, lgb.Dataset(X, rel, group=_groups(groups)[1], weight=weights), 1000,
                     valid_sets=[lgb.Dataset(X_val, rel_val, group=_groups(groups_val)[1])],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def topk_regret(pred: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int) -> float:
    """Mean over groups of (best y in the group) minus (best y among the k rows with the highest pred)."""
    starts, counts, _ = _groups(groups)
    total = 0.0
    for a, c in zip(starts, counts):
        yg = y[a:a + c]
        total += yg.max() - yg[np.argsort(-pred[a:a + c], kind="stable")[:k]].max()
    return total / len(counts)


def h2_score(model) -> VertexScore:
    """Growth hook: the model's prediction on each vertex's features in the state graph is its priority."""
    def for_state(g):
        nodes, X = vertex_features(g)
        priority = dict(zip(nodes, model.predict(X).tolist()))
        return lambda grow, cands: [priority[v] for v in cands]
    return for_state


def h4_rank(model) -> Rank:
    """Candidate-choice hook: the model's prediction on each candidate piece's features is its rank."""
    return lambda g, maps: model.predict(piece_features(g, maps))


def save(model, path) -> None:
    """A LinearModel to .json, a LightGBM booster to .txt."""
    path = Path(path)
    if path.suffix == ".json":
        path.write_text(json.dumps({k: getattr(model, k).tolist() for k in ("mu", "sd", "w")}))
    elif path.suffix == ".txt":
        model.save_model(str(path))
    else:
        raise ValueError(f"cannot save a model as {path.suffix!r}; use .json or .txt")


def load(path):
    path = Path(path)
    if path.suffix == ".json":
        return LinearModel(**{k: np.array(v) for k, v in json.loads(path.read_text()).items()})
    if path.suffix == ".txt":
        return lgb.Booster(model_file=str(path))
    raise ValueError(f"cannot load a model from {path.suffix!r}; use .json or .txt")


GLS_THETA = (5.0, 0.0, 0.0)   # target degree t, weight a on placed neighbours, weight c on raw degree


def decode(x) -> tuple:
    """CMA-ES coordinates to a theta; the origin is GLS_THETA."""
    return (float(5 + 5 * x[0]), float(x[1]), float(x[2]))


def theta_score(theta) -> VertexScore:
    """Growth hook: GLS's score with a tunable target degree, plus a reward for vertices that already have
    placed neighbours and one for high degree in the state graph. At GLS_THETA the extra terms add exactly 0.0,
    so it reproduces gls_score(5) bit for bit."""
    t, a, c = theta
    def for_state(g):
        def score(grow, cands):
            placed = lambda v: len(g._adj[v].keys() & grow.mapping.keys())
            return [1 - abs(t - grow.deg(v)) / grow.n + a * placed(v) / grow.n + c * g.degree(v) / len(g)
                    for v in cands]
        return score
    return for_state


def load_theta(path) -> tuple:
    """The theta stored in a result file written by evolve.py."""
    return tuple(json.loads(Path(path).read_text())["theta"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    train, val = np.load(a.train), np.load(a.val)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # (dataset suffix, feature names, baseline column, k for the regret, NDCG cut-off)
    hooks = {"h4": ("4", PIECE_FEATURES, "size", 4, 4), "h2": ("2", VERTEX_FEATURES, "gls_term", 1, 5)}
    regret, seconds = {}, {}
    for hook, (d, names, rule, k, eval_at) in hooks.items():
        X, y, g, w = (train[f"{c}{d}"] for c in "Xygw")
        Xv, yv, gv = (val[f"{c}{d}"] for c in "Xyg")
        start = time.perf_counter()
        lin = fit_linear(X, y, g, w)
        seconds[f"lin-{hook}"] = time.perf_counter() - start
        start = time.perf_counter()
        booster = fit_lgbm(X, relevance(y, g), g, Xv, relevance(yv, gv), gv, eval_at, weights=w)
        seconds[f"lgb-{hook}"] = time.perf_counter() - start
        save(lin, out / f"lin-{hook}.json")
        save(booster, out / f"lgb-{hook}.txt")
        regret[hook] = dict(k=k, rule=rule, rule_regret=topk_regret(Xv[:, names.index(rule)], yv, gv, k),
                            linear=topk_regret(lin.predict(Xv), yv, gv, k),
                            lgbm=topk_regret(booster.predict(Xv), yv, gv, k))

    print(f"validation top-k regret (lower is better)\n{'hook':6}{'k':>3}{'rule':>20}{'linear':>10}{'lgbm':>10}")
    for hook, r in regret.items():
        print(f"{hook:6}{r['k']:>3}{r['rule'] + ' ' + format(r['rule_regret'], '.4f'):>20}"
              f"{r['linear']:>10.4f}{r['lgbm']:>10.4f}")
    print("training seconds: " + ", ".join(f"{name} {s:.1f}" for name, s in seconds.items()))
    meta = dict(val_regret=regret, train_seconds=seconds,
                dataset=dict(train=json.loads(str(train["meta"])), val=json.loads(str(val["meta"]))))
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
