"""A GFlowNet on the piece builder: the same network, trained to sample a piece with probability proportional to
exp(beta * (one-step reward - the value model's estimate of the state)).

The partition function differs enormously between states and starts, so the usual single learned log Z cannot fit.
Instead, for one start of one state, every sampled path should have the same residual
`sum log P_F - log R - sum log P_B`, and the loss is the variance of that residual over the start's samples. The
backward policy is uniform over the placed vertices (not the start) whose removal leaves the rest connected on the
lattice around the start trap. The deployed hook samples one piece per vertex and computes no independent set.

    uv run python -m learned_selection.gfn --init models/bld-h3.pt --value dgin-h5 --rank dgin-h4 --models models \\
        --train 'cache/graphs_train_*.pkl' --beta 2 --out cache/gfn/b2 --threads 6 --workers 18
"""
import argparse, io, json, math, random, time, zlib
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
import torch

from .builder import (GRAD_CLIP, PieceBuilder, _atomic_save, _close, _copy_companions, _from_saved, _lattice, _map,
                      _open_pool, _saved, _training_states, _write_meta, blob_of, build_piece, load_builder,
                      one_step_reward, pair_weights)
from .gnn import encode


def parent_count(place: dict, adj: list[list[int]], start_trap: int) -> int:
    """How many placed vertices, not the start, can be removed leaving the rest connected on the lattice through the
    start trap; at least 1."""
    occ = set(place.values())
    count = 0
    for t in occ - {start_trap}:
        rest = occ - {t}
        seen, stack = {start_trap}, [start_trap]
        while stack:
            for x in adj[stack.pop()]:
                if x in rest and x not in seen:
                    seen.add(x)
                    stack.append(x)
        count += len(seen) == len(rest)
    return max(count, 1)


def state_seed(g: nx.Graph) -> list[int]:
    """A seed from the state alone: its size, its edge count and a hash of its sorted labels."""
    return [len(g), g.number_of_edges(), zlib.crc32(",".join(map(str, sorted(g))).encode())]


def gfn_hook(model: PieceBuilder) -> Callable:
    """Build hook: one sampled piece from every vertex of g, in g order, from one embedding pass. The generator is
    seeded by the state, so a state's pieces do not depend on what was built before it."""
    adj, start_trap, trap_x = _lattice()
    weights = pair_weights(model)

    def build(g: nx.Graph) -> list[dict]:
        if not len(g):
            return []
        rng = np.random.default_rng(state_seed(g))
        with torch.inference_mode():
            H = model.embed(encode(g).to("cpu")).numpy()
        return [build_piece(g, H, weights, v, adj, start_trap, trap_x, rng)[0] for v in g]
    return build


def tb_rollout(job: tuple) -> tuple:
    """job = (blob, n, edges, starts, samples, seed, beta, value name, models dir). Samples `samples` pieces from each
    of `starts` random vertices. Returns (n, edges, rows), a row being (start, log_r, log_pb, decisions, reward, piece
    size); a piece with no decision is left out. Numpy only."""
    from .bench import method_hooks
    from .graphdata import graph_of
    blob, n, edges, starts, samples, seed, beta, value_name, models_dir = job
    g = graph_of(dict(n=n, edges=edges))
    if not len(g):
        return n, edges, []
    value = method_hooks(value_name, models_dir)["value"]
    model = _from_saved(torch.load(io.BytesIO(blob), weights_only=True))
    adj, start_trap, trap_x = _lattice()
    weights = pair_weights(model)     # after the load: the weights are the blob's
    with torch.inference_mode():
        H = model.embed(encode(g).to("cpu")).numpy()
    base = float(value([g])[0])
    rng, rows = np.random.default_rng(seed), []
    for v in random.Random(seed).sample(list(g), min(starts, len(g))):
        for _ in range(samples):
            piece, decisions = build_piece(g, H, weights, v, adj, start_trap, trap_x, rng)
            if decisions:
                order = list(piece.items())
                log_pb = -sum(math.log(parent_count(dict(order[:i]), adj, start_trap)) for i in range(2, len(order) + 1))
                reward = one_step_reward(g, piece, value)
                rows.append((v, beta * (reward - base), log_pb, decisions, reward, len(piece)))
    return n, edges, rows


def _groups(rows: list) -> list[list]:
    """The rows of each start that has at least 2 of them."""
    groups = {}
    for row in rows:
        groups.setdefault(row[0], []).append(row)
    return [group for group in groups.values() if len(group) >= 2]


def path_logp(model: PieceBuilder, n: int, edges: np.ndarray, rows: list) -> torch.Tensor:
    """The log probability of each row's path, from one embedding pass: the sum over its decisions of the chosen
    pair's log softmax over the options."""
    from .graphdata import graph_of
    ds = [d for row in rows for d in row[3]]
    if not ds:
        return torch.zeros(len(rows))
    h = model.embed(encode(graph_of(dict(n=n, edges=edges))).to("cpu"))
    sizes = [len(d[0]) for d in ds]
    logits = model.pair_logits(h, np.concatenate([d[0] for d in ds]), np.concatenate([d[1] for d in ds]))
    group = torch.repeat_interleave(torch.arange(len(ds)), torch.as_tensor(sizes))
    peak = logits.detach().new_full((len(ds),), -torch.inf).scatter_reduce(0, group, logits.detach(), "amax")
    log_z = logits.new_zeros(len(ds)).index_add_(0, group, (logits - peak[group]).exp()).log() + peak
    chosen = torch.as_tensor(np.cumsum([0] + sizes[:-1]) + [d[2] for d in ds])
    path = torch.repeat_interleave(torch.arange(len(rows)), torch.as_tensor([len(row[3]) for row in rows]))
    return logits.new_zeros(len(rows)).index_add_(0, path, logits[chosen] - log_z)


def tb_loss(model: PieceBuilder, n: int, edges: np.ndarray, rows: list) -> tuple[torch.Tensor, int]:
    """(for each start with at least 2 samples, the sum over them of the squared difference between a sample's residual
    and the start's mean residual, the number of samples used), from one embedding pass."""
    groups = _groups(rows)
    used = [r for group in groups for r in group]
    if not used:
        return torch.zeros(()), 0
    residual = path_logp(model, n, edges, used) - torch.as_tensor([r[1] + r[2] for r in used], dtype=torch.float32)
    loss, at = torch.zeros(()), 0
    for group in groups:
        x, at = residual[at:at + len(group)], at + len(group)
        loss = loss + ((x - x.mean()) ** 2).sum()
    return loss, len(used)


def _step(model, opt, pool, a, t: int, states: list[dict]) -> dict:
    began, seed = time.perf_counter(), a.seed * 100003 + t
    batch = random.Random(seed).sample(states, a.states)
    blob = blob_of(model)
    found = _map(pool, tb_rollout, [(blob, r["n"], r["edges"], a.starts, a.samples, seed * 1000 + i, a.beta, a.value,
                                     a.models) for i, r in enumerate(batch)])
    torch.set_num_threads(a.threads)      # a rollout without workers leaves torch at one thread
    rows = [row for _, _, rs in found for row in rs]
    used = sum(len(g) for _, _, rs in found for g in _groups(rs))
    opt.zero_grad()
    total = 0.0
    for n, edges, rs in found:
        loss, count = tb_loss(model, n, edges, rs)
        if count:
            (loss / used).backward()
            total += loss.item() / used
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
    opt.step()
    mean = lambda xs: float(np.mean(xs)) if len(xs) else 0.0
    return dict(iter=t, reward=mean([r[4] for r in rows]), size=mean([r[5] for r in rows]),
                decisions=sum(len(r[3]) for r in rows), samples=used, loss=total, grad_norm=norm,
                seconds=time.perf_counter() - began)


def train(a) -> None:
    from .az import _cpu_s
    from .dataset import _commit
    states = [r for r in _training_states(a) if r["n"] >= a.min_n]
    if len(states) < a.states:
        raise ValueError(f"{len(states)} states of at least {a.min_n} nodes, fewer than the {a.states} an iteration needs")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    model = load_builder(a.init)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "log.jsonl"
    log.write_text("")
    t, spent, cpu = 0, 0.0, _cpu_s()
    while t < a.iters:
        stop = min((t // a.every + 1) * a.every, a.iters)
        pool = _open_pool(a.workers)       # one pool per stretch between checkpoints, so its CPU is counted at each
        for step in range(t + 1, stop + 1):
            row = _step(model, opt, pool, a, step, states)
            with open(log, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"gfn {step}: reward {row['reward']:.3f} size {row['size']:.1f} loss {row['loss']:.4f} "
                  f"{row['seconds']:.0f} s", flush=True)
        _close(pool)
        t, spent = stop, _cpu_s() - cpu
        _copy_companions(a, out / f"it{t}")
        _atomic_save(_saved(model), out / f"it{t}" / "gfn-h3.pt")
        meta = dict(iter=t, iters=a.iters, init=str(a.init), rank=a.rank, value=a.value, beta=a.beta, seed=a.seed, lr=a.lr,
                    states=a.states, starts=a.starts, samples=a.samples, min_n=a.min_n, train=[str(p) for p in a.train],
                    max_states=a.max_states, cpu_s=spent, commit=_commit())
        _write_meta(meta, out / f"it{t}" / "train_meta_gfn-h3.json")
    print(f"gfn: {t} iterations, cpu {spent:.0f} s", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--rank", default="dgin-h4")
    ap.add_argument("--value", default="dgin-h5")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-states", type=int, help="use a seeded sample of this many states")
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
