"""Training of the graph network scorers and the value models, from the labelled graphs of graphdata.py.

Rankers (jobs `conv-h2` and `conv-h4`) learn which vertices of a state are in a maximum independent set, and which
candidate piece has the best continuation. They start by imitating GLS's own choice (a warm start), then fit the
exact labels, keeping the epoch with the lowest validation top-k regret. `lin-h2` and `lin-h4` are the ridge
counterparts on the structural features. Value models (jobs `conv-h5` and
`lin-h5`) predict the independence number of a graph, and are kept at the lowest validation pruning regret.
A leading `d` on a job name (`dgin-h4`) marks a model trained on the states of a policy's own visits; it trains
the same way and only the file name differs.

    uv run python -m learned_selection.train --train cache/graphs_train_*.pkl --val cache/graphs_val_*.pkl \\
        --jobs gin-h2,gin-h4,gin-h5,lin-h5 --out models/ --threads 6 --workers 12
"""
import argparse, json, os, pickle, random, time
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np

# models comes before gnn: lightgbm is imported lazily inside models, so this process never holds both lightgbm
# and torch (on macOS their OpenMP runtimes clash).
from .dataset import _commit, _setting_name
from .features import PIECE_FEATURES, VERTEX_FEATURES, graph_features, piece_features, vertex_features
from .graphdata import graph_of
from .instances import HELD_OUT_SETTINGS, SEL_SEEDS, TEST_SEEDS
from .models import fit_linear, fit_value_linear, save, topk_regret

import torch
from .gnn import CONVS, GNN, collate, encode, save_gnn

WARM_AGREEMENT = 0.95
PRUNE_K = 4
GRAD_CLIP = 5.0
LARGE = 60   # children of at least this many nodes get their own error figure
_QUIET = lambda *_: None


def load_pickles(paths) -> tuple[list[dict], list[dict]]:
    """(the meta of each file, the concatenated states). Refuses test and selection seeds and held-out settings."""
    loaded = [pickle.load(open(p, "rb")) for p in paths]
    states = [r for d in loaded for r in d["states"]]
    held_out = {_setting_name(f, p) for f, p in HELD_OUT_SETTINGS}
    for r in states:
        if r["seed"] in TEST_SEEDS or r["seed"] in SEL_SEEDS:
            raise ValueError(f"seed {r['seed']} is a test or selection seed; it must not train or validate")
        if r["setting"] in held_out:
            raise ValueError(f"setting {r['setting']} is held out; it must not be used for training or validation")
    return [d["meta"] for d in loaded], states


def load_states(paths) -> list[dict]:
    return load_pickles(paths)[1]


def listwise_loss(scores: torch.Tensor, y: torch.Tensor, group: torch.Tensor, weight) -> torch.Tensor:
    """Within each group the target spreads evenly over the items whose y is the group maximum. The loss is the
    weighted mean over groups of minus the target-weighted log-softmax of the scores. Group ids are 0..G-1 and
    `weight` is one per group (or a scalar)."""
    g = int(group.max()) + 1
    y = y.to(scores.dtype)
    zeros = lambda: scores.new_zeros(g)
    top = scores.new_full((g,), -torch.inf).scatter_reduce(0, group, y, "amax", include_self=True)
    best = (y == top[group]).to(scores.dtype)
    target = best / zeros().index_add_(0, group, best)[group]
    peak = scores.detach().new_full((g,), -torch.inf).scatter_reduce(0, group, scores.detach(), "amax")
    log_z = zeros().index_add_(0, group, (scores - peak[group]).exp()).log() + peak
    per_group = -zeros().index_add_(0, group, target * (scores - log_z[group]))
    w = torch.as_tensor(weight, dtype=scores.dtype, device=scores.device).expand(g)
    return (w * per_group).sum() / w.sum()


def _pmap(fn, jobs: list, workers: int) -> list:
    if workers <= 1 or len(jobs) < 2:
        return list(map(fn, jobs))
    with Pool(workers) as pool:
        return pool.map(fn, jobs, chunksize=max(1, len(jobs) // (workers * 4)))


def _encode_ranker(job: tuple):
    """Runs in a worker and returns a numpy Batch. job = (n, edges, pieces or None)."""
    n, edges, pieces = job
    g = graph_of(dict(n=n, edges=edges))
    return encode(g, None if pieces is None else [set(p.tolist()) for p in pieces])


def _predict(net: GNN, batches: list, chunk: int = 128) -> np.ndarray:
    """The scores of every item, concatenated, from the model in eval mode on the CPU."""
    net.eval()
    with torch.inference_mode():
        return np.concatenate([net(collate(batches[i:i + chunk]).to("cpu")).double().numpy()
                               for i in range(0, len(batches), chunk)])


def _agree(pred: np.ndarray, rule: np.ndarray, sizes: np.ndarray) -> float:
    """The share of states whose top-scored item has the maximum rule value (a tied maximum counts)."""
    ends = np.cumsum(sizes)
    return float(np.mean([rule[a:b][np.argmax(pred[a:b])] == rule[a:b].max() for a, b in zip(ends - sizes, ends)]))


def train_ranker(conv: str, hook: str, train: list[dict], val: list[dict], device, epochs: int = 60,
                 patience: int = 8, warm_epochs: int = 10, batch: int = 64, lr: float = 1e-3, seed: int = 0,
                 workers: int = 1, log=_QUIET) -> tuple[GNN, dict]:
    """Train a vertex scorer (hook "h2", y = member) or a piece scorer (hook "h4", y = q). Returns the model with
    the best validation weights, on the CPU, and an info dict."""
    start = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    key, head, k = ("member", "vertex", 1) if hook == "h2" else ("q", "piece", 4)

    def prepare(states):
        used = [r for r in states if r[key] is not None and r["n"] > 0 and (hook == "h2" or len(r["pieces"]))]
        batches = _pmap(_encode_ranker, [(r["n"], r["edges"], r["pieces"] if hook == "h4" else None)
                                         for r in used], workers)
        col = VERTEX_FEATURES.index("gls_term") if hook == "h2" else PIECE_FEATURES.index("size")
        ys = [r[key].astype(np.float32) for r in used]
        rules = [(b.x if hook == "h2" else b.piece_x)[:, col] for b in batches]
        sizes = np.array([len(y) for y in ys])
        return dict(batches=batches, ys=ys, rules=rules, sizes=sizes, n=np.array([r["n"] for r in used]),
                    y=np.concatenate(ys) if ys else np.zeros(0), rule=np.concatenate(rules) if ys else np.zeros(0),
                    group=np.repeat(np.arange(len(ys)), sizes))

    tr, va = prepare(train), prepare(val)
    if not len(tr["ys"]) or not len(va["ys"]):
        raise ValueError(f"no states with a proven {key} label to train or validate on")
    encode_seconds = time.perf_counter() - start
    stack = lambda f: np.concatenate([f(b) for b in tr["batches"]]).astype(np.float64)
    mu, sd = (lambda x: (x.mean(axis=0), x.std(axis=0)))(stack(lambda b: b.x))
    pmu, psd = (lambda x: (x.mean(axis=0), x.std(axis=0)))(stack(lambda b: b.piece_x)) if hook == "h4" else (None, None)
    net = GNN(conv, head, mu, sd, pmu, psd).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    weight = tr["n"] / tr["n"].mean()
    to_t = lambda a, dtype=None: torch.as_tensor(a, dtype=dtype, device=device)

    def run_epoch(target: list):
        net.train()
        order = rng.permutation(len(tr["ys"]))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            group = np.repeat(np.arange(len(idx)), tr["sizes"][idx])
            scores = net(collate([tr["batches"][j] for j in idx]).to(device))
            loss = listwise_loss(scores, to_t(np.concatenate([target[j] for j in idx])), to_t(group),
                                 to_t(weight[idx], torch.float32))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()

    def evaluate():
        net.cpu()
        pred = _predict(net, va["batches"])
        net.to(device)
        return pred

    agreement, warm_run = None, 0
    for _ in range(warm_epochs):
        run_epoch(tr["rules"])
        warm_run += 1
        agreement = _agree(evaluate(), va["rule"], va["sizes"])
        log(f"warm {warm_run}: agreement {agreement:.3f}")
        if agreement >= WARM_AGREEMENT:
            break
    if agreement is None:
        agreement = _agree(evaluate(), va["rule"], va["sizes"])

    best, best_epoch, best_state, history = np.inf, 0, None, []
    for e in range(1, epochs + 1):
        run_epoch(tr["ys"])
        regret = topk_regret(evaluate(), va["y"], va["group"], k)
        history.append(float(regret))
        log(f"epoch {e}: val regret {regret:.4f}")
        if regret < best:
            best, best_epoch = regret, e
            best_state = {n: v.detach().clone() for n, v in net.state_dict().items()}
        elif e - best_epoch >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.cpu().eval()
    return net, dict(imitation_agreement=float(agreement), warm_epochs_run=warm_run,
                     val_regret=float(best if best_state is not None else topk_regret(
                         _predict(net, va["batches"]), va["y"], va["group"], k)),
                     rule_regret=float(topk_regret(va["rule"], va["y"], va["group"], k)),
                     best_epoch=best_epoch, history=history, epochs_run=len(history),
                     train_states=len(tr["ys"]), val_states=len(va["ys"]), encode_seconds=encode_seconds,
                     seconds=time.perf_counter() - start)


def _picks(states: list[dict], cap: int, seed: int):
    """Yield (state position, child index or None, alpha, s) for each state with a known alpha: the state itself,
    then at most `cap` of its children, sampled by random.Random(seed)."""
    rng = random.Random(seed)
    for i, rec in enumerate(states):
        if rec["alpha"] is None:
            continue
        yield i, None, rec["alpha"], 0
        ids = range(len(rec["children"]))
        for j in sorted(rng.sample(ids, cap)) if len(ids) > cap else ids:
            yield i, j, rec["children"][j][1], rec["children"][j][2]


def _iter_rows(states: list[dict], cap: int, seed: int):
    graph = None
    for i, j, alpha, s in _picks(states, cap, seed):
        if j is None:
            graph = graph_of(states[i])
            yield graph, alpha, i, s, False
        else:
            yield graph.subgraph(states[i]["children"][j][0].tolist()), alpha, i, s, True


def value_rows(states: list[dict], cap: int = 64, seed: int = 0) -> list[tuple]:
    """(graph, alpha, group, s, is_child) rows: each state with a known alpha (is_child False, s 0) and at most
    `cap` of its children. group is the state's position in `states`."""
    return list(_iter_rows(states, cap, seed))


def _encode_state_rows(job: tuple) -> list:
    """Runs in a worker and returns numpy Batches: the state graph, then each child. job = (n, edges, children)."""
    n, edges, children = job
    g = graph_of(dict(n=n, edges=edges))
    return [encode(g)] + [encode(g.subgraph(c.tolist()).copy()) for c in children]


def _row_metrics(pred, alpha, s, group, child, n) -> dict:
    err = np.abs(pred - alpha)
    mean = lambda mask: float(err[mask].mean()) if mask.any() else None
    return dict(val_mae=float(err.mean()), mae_of_zero=float(np.abs(alpha).mean()), child_mae=mean(child),
                state_mae=mean(~child), large_child_mae=mean(child & (n >= LARGE)),
                prune_regret=float(topk_regret((s + pred)[child], (s + alpha)[child], group[child], PRUNE_K)),
                rule_prune_regret=float(topk_regret(s[child], (s + alpha)[child], group[child], PRUNE_K)))


def _better(a: dict, b: dict) -> bool:
    """Whether epoch summary a beats b: lower prune regret, and on a tie lower child error."""
    return (a["prune_regret"], a["child_mae"]) < (b["prune_regret"], b["child_mae"])


def _row_arrays(states: list[dict], cap: int, seed: int) -> dict:
    picks = list(_picks(states, cap, seed))
    cols = np.array([(i, j is not None, a, s) for i, j, a, s in picks], dtype=np.float64).reshape(-1, 4)
    out = dict(group=cols[:, 0].astype(np.int64), child=cols[:, 1] > 0, alpha=cols[:, 2], s=cols[:, 3])
    if not out["child"].any():
        raise ValueError("no children with a proven value")
    return out


def train_value(conv: str, train: list[dict], val: list[dict], device, epochs: int = 60, patience: int = 8,
                batch: int = 16, lr: float = 1e-3, seed: int = 0, cap: int = 64, workers: int = 1,
                log=_QUIET) -> tuple[GNN, dict]:
    """Train a value network on the rows of value_rows. `batch` counts states. The loss is Huber on alpha plus a
    listwise loss that orders each state's children. Kept at the lowest validation pruning regret, ties broken by the child error."""
    start = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def prepare(states):
        arrays = _row_arrays(states, cap, seed)
        used = sorted(set(arrays["group"].tolist()))
        counts = np.bincount(arrays["group"])[used]
        picks = {}
        for i, j, _, _ in _picks(states, cap, seed):
            if j is not None:
                picks.setdefault(i, []).append(states[i]["children"][j][0])
        per_state = _pmap(_encode_state_rows, [(states[i]["n"], states[i]["edges"], picks.get(i, []))
                                               for i in used], workers)
        arrays.update(batches=[b for bs in per_state for b in bs], ends=np.cumsum(counts))
        arrays["starts"] = arrays["ends"] - counts
        arrays["n"] = np.array([len(b.x) for b in arrays["batches"]])
        return arrays

    tr, va = prepare(train), prepare(val)
    encode_seconds = time.perf_counter() - start
    x = np.concatenate([b.x for b in tr["batches"]]).astype(np.float64)
    net = GNN(conv, "value", x.mean(axis=0), x.std(axis=0)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    to_t = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=device)

    def run_epoch():
        net.train()
        order = rng.permutation(len(tr["starts"]))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            rows = np.concatenate([np.arange(tr["starts"][j], tr["ends"][j]) for j in idx])
            local = np.repeat(np.arange(len(idx)), tr["ends"][idx] - tr["starts"][idx])
            child = tr["child"][rows]
            alpha, s = to_t(tr["alpha"][rows]), to_t(tr["s"][rows])
            pred = net(collate([tr["batches"][r] for r in rows]).to(device))
            loss = torch.nn.functional.huber_loss(pred, alpha, delta=1.0)
            if child.any():
                dense = np.unique(local[child], return_inverse=True)[1]
                c = to_t(child, torch.bool)
                loss = loss + listwise_loss((s + pred)[c], (s + alpha)[c], to_t(dense, torch.long), 1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()

    def evaluate():
        net.cpu()
        pred = _predict(net, va["batches"])
        net.to(device)
        return _row_metrics(pred, va["alpha"], va["s"], va["group"], va["child"], va["n"])

    best, best_state, best_epoch, history, mae_history = None, None, 0, [], []
    for e in range(1, epochs + 1):
        run_epoch()
        m = evaluate()
        history.append(m["prune_regret"])
        mae_history.append(m["child_mae"])
        log(f"epoch {e}: val prune regret {m['prune_regret']:.4f}, mae {m['val_mae']:.3f}")
        if best is None or _better(m, best):
            best, best_epoch = m, e
            best_state = {n: v.detach().clone() for n, v in net.state_dict().items()}
        elif e - best_epoch >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.cpu().eval()
    return net, dict(**(best or evaluate()), best_epoch=best_epoch, history=history, mae_history=mae_history,
                     epochs_run=len(history), rows=len(tr["alpha"]), val_rows=len(va["alpha"]),
                     encode_seconds=encode_seconds, seconds=time.perf_counter() - start)


def fit_linear_value_job(train: list[dict], val: list[dict], cap: int = 64):
    """Ridge on the graph features of the rows of value_rows. Returns the model and an info dict."""
    start = time.perf_counter()

    def features(states):
        rows = [(graph_features(g), alpha, i, s, c) for g, alpha, i, s, c in _iter_rows(states, cap, 0)]
        return np.array([r[0] for r in rows]), dict(alpha=np.array([r[1] for r in rows], dtype=np.float64),
                                                    group=np.array([r[2] for r in rows]),
                                                    s=np.array([r[3] for r in rows], dtype=np.float64),
                                                    child=np.array([r[4] for r in rows]))
    X, y = features(train)
    model = fit_value_linear(X, y["alpha"])
    Xv, v = features(val)
    if not v["child"].any():
        raise ValueError("no children with a proven value")
    return model, dict(**_row_metrics(model.predict(Xv), v["alpha"], v["s"], v["group"], v["child"], Xv[:, 0]),
                       rows=len(y["alpha"]), val_rows=len(v["alpha"]), seconds=time.perf_counter() - start)


def _linear_rows(job: tuple):
    """Runs in a worker and returns numpy arrays (X, y) for one state. job = (hook, n, edges, pieces, y)."""
    hook, n, edges, pieces, y = job
    g = graph_of(dict(n=n, edges=edges))
    X = vertex_features(g)[1] if hook == "h2" else piece_features(g, [set(p.tolist()) for p in pieces])
    return X, y


def fit_linear_ranker_job(hook: str, train: list[dict], val: list[dict], workers: int = 1):
    """Weighted ridge on the within-state order of the vertex features (h2, y = member) or the piece features
    (h4, y = q), each state weighted by its node count as in models.main. Returns the model and an info dict."""
    start = time.perf_counter()
    key, rule, names, k = (("member", "gls_term", VERTEX_FEATURES, 1) if hook == "h2"
                           else ("q", "size", PIECE_FEATURES, 4))

    def rows(states):
        used = [r for r in states if r[key] is not None and r["n"] > 0 and (hook == "h2" or len(r["pieces"]))]
        if not used:
            raise ValueError(f"no states with a proven {key} label to train or validate on")
        out = _pmap(_linear_rows, [(hook, r["n"], r["edges"], r["pieces"], r[key]) for r in used], workers)
        sizes = np.array([len(y) for _, y in out])
        return (np.concatenate([X for X, _ in out]), np.concatenate([y for _, y in out]).astype(np.float64),
                np.repeat(np.arange(len(used)), sizes), np.repeat([r["n"] for r in used], sizes).astype(np.float64))
    X, y, g, w = rows(train)
    Xv, yv, gv, _ = rows(val)
    model = fit_linear(X, y, g, w)
    return model, dict(val_regret=float(topk_regret(model.predict(Xv), yv, gv, k)),
                       rule_regret=float(topk_regret(Xv[:, names.index(rule)], yv, gv, k)),
                       train_states=int(g[-1]) + 1, val_states=int(gv[-1]) + 1, seconds=time.perf_counter() - start)


def _jobs(spec: str) -> list[tuple[str, str, str]]:
    jobs = []
    for job in spec.split(","):
        conv, _, hook = job.partition("-")
        if conv[:1] in ("d", "b") and conv[1:] in (*CONVS, "lin"):   # d, b: trained on a policy's own states
            conv = conv[1:]
        if hook not in ("h2", "h4", "h5") or conv not in (*CONVS, "lin"):
            raise SystemExit(f"unknown job {job!r}; use [d|b]{{lin,{','.join(CONVS)}}}-h2|h4|h5")
        jobs.append((job, conv, hook))
    return jobs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--jobs", required=True, help="comma-separated: conv-h2|h4|h5 or lin-h2|h4|h5; a leading d or b marks a policy's or a builder's own states")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    jobs = _jobs(a.jobs)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (train_metas, train), (val_metas, val) = load_pickles(a.train), load_pickles(a.val)
    for job, conv, hook in jobs:
        log = lambda msg: print(f"  {job} {msg}", flush=True)
        if conv == "lin":
            model, info = (fit_linear_value_job(train, val) if hook == "h5"
                           else fit_linear_ranker_job(hook, train, val, workers=a.workers))
            save(model, out / f"{job}.json")
        elif hook == "h5":
            net, info = train_value(conv, train, val, a.device, epochs=a.epochs, workers=a.workers, log=log)
            save_gnn(net, out / f"{job}.pt")
        else:
            net, info = train_ranker(conv, hook, train, val, a.device, epochs=a.epochs, workers=a.workers, log=log)
            save_gnn(net, out / f"{job}.pt")
        meta = dict(job=job, info=info, device=a.device, epochs=a.epochs, commit=_commit(),
                    dataset=dict(train=train_metas, val=val_metas))
        (out / f"train_meta_{job}.json").write_text(json.dumps(meta, indent=2))
        if hook == "h5":
            line = (f"prune regret {info['prune_regret']:.4f} (rule {info['rule_prune_regret']:.4f}), "
                    f"val mae {info['val_mae']:.3f} (zero {info['mae_of_zero']:.3f})")
        else:
            line = (f"val regret {info['val_regret']:.4f} (rule {info['rule_regret']:.4f})"
                    + (f", agreement {info['imitation_agreement']:.3f}" if "imitation_agreement" in info else ""))
        timing = (f", encode {info['encode_seconds']:.0f} s, {info['epochs_run'] + info.get('warm_epochs_run', 0)} epochs"
                  if "encode_seconds" in info else "")
        print(f"{job}: {line}, {info['seconds']:.0f} s{timing}", flush=True)


if __name__ == "__main__":
    main()
