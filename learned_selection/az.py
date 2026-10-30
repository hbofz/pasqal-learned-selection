"""Self-play training of the tree search's prior, on top of search.run_mcts.

Each iteration plays the search against training instances with root noise, keeps the visit counts of every node
it spent at least 16 visits on, and fine-tunes the piece scorer (`az-h4`) towards those counts together with the
exact-labelled states of the base training set. Every step is its own process, so torch never crosses a fork: the
parent of the self-play pool imports no torch, and the workers load it through `method_hooks`.

    uv run python -m learned_selection.az selfplay --method lin-h2+gin-h4+sage-h5+mcts --models models \\
        --budget 100 --k 8 --workers 3 --out cache/az/selfplay_it1.pkl
    uv run python -m learned_selection.az tune --init models/gin-h4.pt --base-train cache/graphs_train_*.pkl \\
        --examples cache/az/selfplay_it1.pkl --out models/az/it1/az-h4.pt --threads 2 --workers 3
    uv run python -m learned_selection.az loop --method lin-h2+gin-h4+sage-h5+mcts --models models --iters 3 \\
        --budget 100 --k 8 --base-train cache/graphs_train_*.pkl --out models/az --cache cache/az --workers 3
"""
import argparse, json, os, pickle, random, resource, shutil, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .bench import GNN_KINDS, KIND_EXT, method_hooks, parse_method, search_of, shard
from .dataset import _commit, _setting_name
from .instances import TRAIN_SEEDS, TRAIN_SETTINGS, TRAIN_SIZES, make
from .search import run_mcts

NOISE = (0.25, 0.3)   # root noise: epsilon and the Dirichlet concentration
PIECE_SOLVER_WIDTH = 2
GRAD_CLIP = 5.0
_QUIET = lambda *_: None


def self_play_job(job: tuple) -> list[dict]:
    """job = (family, params, n, seed, method, models_dir, budget, k). One search with root noise and a trace;
    the examples of nodes whose children were visited, tagged with the instance. Numpy only."""
    family, params, n, seed, method, models_dir, budget, k = job
    if seed not in TRAIN_SEEDS:
        raise ValueError(f"seed {seed} is not a training seed; self-play must not use it")
    trace = []
    run_mcts(make(family, n, seed, **params), k, PIECE_SOLVER_WIDTH, budget, c=search_of(method), seed=seed,
             trace=trace, noise=(*NOISE, np.random.default_rng(seed)), **method_hooks(method, models_dir))
    tag = dict(setting=_setting_name(family, params), instance_n=n, seed=seed)
    return [dict(e, **tag) for e in trace if e["visits"].sum() > 0]


def soft_listwise_loss(scores, target, group, weight):
    """As train.listwise_loss, with the target given per item and already normalised within its group: the
    weighted mean over groups of minus the target-weighted log-softmax of the scores. Group ids are 0..G-1 and
    `weight` is one per group (or a scalar)."""
    import torch
    g = int(group.max()) + 1
    zeros = lambda: scores.new_zeros(g)
    peak = scores.detach().new_full((g,), -torch.inf).scatter_reduce(0, group, scores.detach(), "amax")
    log_z = zeros().index_add_(0, group, (scores - peak[group]).exp()).log() + peak
    per_group = -zeros().index_add_(0, group, target.to(scores.dtype) * (scores - log_z[group]))
    w = torch.as_tensor(weight, dtype=scores.dtype, device=scores.device).expand(g)
    return (w * per_group).sum() / w.sum()


def fine_tune_policy(net, qstates: list[dict], examples: list[dict], epochs: int = 10, lr: float = 3e-4,
                     batch: int = 64, seed: int = 0, workers: int = 1, log=_QUIET) -> dict:
    """Fine-tune the piece scorer `net` in place. An epoch shuffles the union of the exact states (target uniform
    over the candidates of the best q) and the self-play examples (target the share of visits per expanded
    candidate). Every piece of a state is encoded, since the size rank depends on all of them, and the loss sees
    the target's items only. Each state is weighted by its node count; Adam, gradient clip 5, no early stopping."""
    import torch
    from .gnn import collate
    from .train import _encode_ranker, _pmap
    start = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    exact = [r for r in qstates if r["q"] is not None and r["n"] > 0 and len(r["pieces"])]
    units = []   # (n, the items the loss sees, their target)
    for r in exact:
        best = (r["q"] == r["q"].max()).astype(np.float64)
        units.append((r["n"], np.arange(len(best)), best / best.sum()))
    for e in examples:
        units.append((e["n"], e["expanded"], e["visits"] / e["visits"].sum()))
    batches = _pmap(_encode_ranker, [(r["n"], r["edges"], r["pieces"]) for r in exact]
                    + [(e["n"], e["edges"], e["pieces"]) for e in examples], workers)
    if not units:
        raise ValueError("no states or examples to fine-tune on")
    n = np.array([u[0] for u in units], dtype=np.float64)
    weight = torch.as_tensor(n / n.mean(), dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    net.train()
    losses = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(units))
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            sizes = [len(b.piece_x) for b in (batches[j] for j in idx)]
            offset = np.cumsum([0] + sizes[:-1])
            keep = np.concatenate([units[j][1] + o for j, o in zip(idx, offset)])
            group = np.repeat(np.arange(len(idx)), [len(units[j][1]) for j in idx])
            scores = net(collate([batches[j] for j in idx]).to("cpu"))[torch.as_tensor(keep)]
            loss = soft_listwise_loss(scores, torch.as_tensor(np.concatenate([units[j][2] for j in idx])),
                                      torch.as_tensor(group), weight[torch.as_tensor(idx)])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total / len(units))
        log(f"epoch {epoch}: loss {losses[-1]:.4f}")
    net.eval()
    return dict(loss=losses, states=len(exact), examples=len(examples), seconds=time.perf_counter() - start)


def az_name(method: str) -> str:
    """The method with its H4 part replaced by az-h4, the search suffix kept."""
    parts = ["az-h4" if hook == "h4" else f"{kind}-{hook}" for kind, hook in parse_method(method)]
    return "+".join(parts + ([method.split("+")[-1]] if search_of(method) is not None else []))


def _write_pickle(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _cpu_s() -> float:
    """CPU seconds of this process and of every child it has waited for."""
    own, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime


def selfplay(a) -> None:
    start, cpu = time.perf_counter(), _cpu_s()
    commit = _commit()
    jobs = shard([(fam, params, n, seed, a.method, a.models, a.budget, a.k) for fam, params in TRAIN_SETTINGS
                  for n in TRAIN_SIZES for seed in TRAIN_SEEDS], a.shard)
    examples = []
    pool = Pool(a.workers) if a.workers > 1 else None
    for i, found in enumerate(pool.imap(self_play_job, jobs) if pool else map(self_play_job, jobs), 1):
        examples += found
        if i % 20 == 0 or i == len(jobs):
            print(f"self-play: {i}/{len(jobs)} instances, {len(examples)} examples", flush=True)
    if pool:
        pool.close()
        pool.join()
    meta = dict(method=a.method, models=str(a.models), budget=a.budget, k=a.k, shard=a.shard, instances=len(jobs),
                examples=len(examples), cpu_s=_cpu_s() - cpu, wall_s=time.perf_counter() - start, commit=commit)
    _write_pickle(dict(meta=meta, examples=examples), a.out)
    print(f"{len(examples)} examples from {len(jobs)} instances, cpu {meta['cpu_s']:.0f} s, "
          f"wall {meta['wall_s']:.0f} s", flush=True)


def tune(a) -> None:
    import torch
    from .gnn import load_gnn, save_gnn
    from .train import load_pickles
    torch.set_num_threads(a.threads)
    start, cpu = time.perf_counter(), _cpu_s()
    _, qstates = load_pickles(a.base_train)
    if a.max_states and len(qstates) > a.max_states:
        qstates = random.Random(0).sample(qstates, a.max_states)
    loaded = [pickle.load(open(p, "rb")) for p in a.examples]
    examples = [e for d in loaded for e in d["examples"]]
    if any(e["seed"] not in TRAIN_SEEDS for e in examples):
        raise ValueError("an example comes from a seed outside the training seeds")
    net = load_gnn(a.init)
    if net.head != "piece":
        raise ValueError(f"{a.init} has a {net.head!r} head, but the policy needs a 'piece' head")
    info = fine_tune_policy(net, qstates, examples, workers=a.workers, log=lambda m: print(f"  {m}", flush=True))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    save_gnn(net, tmp)
    os.replace(tmp, out)
    meta = dict(info=info, init=str(a.init), base_train=[str(p) for p in a.base_train],
                examples=[str(p) for p in a.examples], max_states=a.max_states, commit=_commit(),
                self_play=[d["meta"] for d in loaded])
    (out.parent / f"train_meta_{out.stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"tuned on {info['states']} states and {info['examples']} examples, loss {info['loss'][0]:.4f} -> "
          f"{info['loss'][-1]:.4f}, cpu {_cpu_s() - cpu:.0f} s, wall {time.perf_counter() - start:.0f} s", flush=True)


def _run(*args) -> None:
    subprocess.run([sys.executable, "-m", "learned_selection.az", *map(str, args)], check=True)


def loop(a) -> None:
    out, cache, base = Path(a.out), Path(a.cache), parse_method(a.method)
    h4_kind = next(kind for kind, hook in base if hook == "h4")
    warm = Path(a.models) / (f"{h4_kind}-h4.pt" if h4_kind in GNN_KINDS else "gin-h4.pt")
    suffix = "" if a.shard is None else "_shard" + a.shard.replace("/", "of")
    start, cpu = time.perf_counter(), _cpu_s()
    for t in range(1, a.iters + 1):
        step = time.perf_counter()
        method, models = (a.method, a.models) if t == 1 else (az_name(a.method), out / f"it{t - 1}")
        played = [cache / f"selfplay_it{i}{suffix}.pkl" for i in range(1, t + 1)]   # every iteration so far is kept
        examples, net = played[-1], out / f"it{t}" / "az-h4.pt"
        if not examples.exists():
            _run("selfplay", "--method", method, "--models", models, "--budget", a.budget, "--k", a.k,
                 "--workers", a.workers, "--out", examples, *(["--shard", a.shard] if a.shard else []))
        if not net.exists():
            _run("tune", "--init", warm if t == 1 else out / f"it{t - 1}" / "az-h4.pt", "--base-train", *a.base_train,
                 "--examples", *played, "--out", net, "--threads", a.threads, "--workers", a.workers,
                 *(["--max-states", a.max_states] if a.max_states else []))
        for kind, hook in base:
            if hook != "h4" and kind != "exact":
                shutil.copy2(Path(a.models) / f"{kind}-{hook}{KIND_EXT[kind]}", net.parent)
        print(f"iteration {t}: wall {time.perf_counter() - step:.0f} s, cpu so far {_cpu_s() - cpu:.0f} s", flush=True)
    print(f"loop: wall {time.perf_counter() - start:.0f} s, cpu {_cpu_s() - cpu:.0f} s", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("selfplay")
    s.add_argument("--method", required=True)
    s.add_argument("--models", default="models")
    s.add_argument("--budget", type=int, default=100)
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--out", required=True)
    s.add_argument("--shard", help="I/N: run every N-th instance starting at I")
    t = sub.add_parser("tune")
    t.add_argument("--init", required=True)
    t.add_argument("--base-train", nargs="+", required=True)
    t.add_argument("--examples", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--workers", type=int, default=1)
    t.add_argument("--max-states", type=int, help="use a random sample of this many of the exact states")
    lp = sub.add_parser("loop")
    lp.add_argument("--method", required=True)
    lp.add_argument("--models", default="models")
    lp.add_argument("--iters", type=int, default=3)
    lp.add_argument("--budget", type=int, default=100)
    lp.add_argument("--k", type=int, default=8)
    lp.add_argument("--base-train", nargs="+", required=True)
    lp.add_argument("--out", default="models/az")
    lp.add_argument("--cache", default="cache/az")
    lp.add_argument("--workers", type=int, default=1)
    lp.add_argument("--threads", type=int, default=2)
    lp.add_argument("--shard")
    lp.add_argument("--max-states", type=int)
    a = ap.parse_args(argv)
    if "method" in a and (search_of(a.method) is None or {h for _, h in parse_method(a.method)} != {"h2", "h4", "h5"}):
        ap.error("--method must be a tree-search name with h2, h4 and h5 hooks")
    {"selfplay": selfplay, "tune": tune, "loop": loop}[a.command](a)


if __name__ == "__main__":
    main()
