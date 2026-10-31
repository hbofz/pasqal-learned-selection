"""A learned piece builder: it adds one (vertex, trap) pair at a time, in place of the breadth-first growth.

A pair (u, t) is feasible when t is free and next to a filled trap and u's placed neighbours are exactly the vertices
on the traps next to t, so every piece it builds is an induced subgraph of the lattice. A GNN body embeds the
vertices once per state; a small MLP scores each feasible pair from the vertex's embedding and ten features of the
pair. The deployed hook takes the highest score at each step and computes no independent set.

Training clones the growth of a vertex scorer, then improves the builder by REINFORCE on a one-step reward: the best of
the piece's engine sets by size plus the value model's estimate of what is left. Rollouts run in worker processes
on numpy copies of the weights and return plain arrays.

    uv run python -m learned_selection.builder clone --expert dgin-h2 --models models --train 'cache/graphs_train_*.pkl' \\
        --epochs 10 --seed 0 --out models/bld/clone0 --threads 2 --workers 8
    uv run python -m learned_selection.builder rl --init models/bld/clone0/bld-h3.pt --value dgin-h5 --rank dgin-h4 \\
        --models models --train 'cache/graphs_train_*.pkl' --out cache/bld/rl --threads 2 --workers 8
"""
import argparse, glob, io, json, multiprocessing, os, random, shutil, time
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch import nn

from .engine import Builder, _remove_closed
from .gnn import GNN, encode, load_gnn
from .growth import code_start_trap, grow_piece, lattice_adjacency
from .instances import TRAIN_SEEDS
from .lattice import load_coords
from .solvers import first_maximum_set, top_maximal_sets

GRAD_CLIP = 5.0

PAIR_FEATURES = ("trap_x", "trap_y", "trap_centre", "trap_degree", "filled_nbrs", "free_nbrs", "size", "deg",
                 "unplaced_deg", "deg_share")


def trap_features(coords: dict, adj: list[list[int]]) -> np.ndarray:
    """(traps, 4): x and y centred and scaled by the largest absolute coordinate, their norm, and degree over 6."""
    xy = np.array([coords[t] for t in range(len(adj))], float)
    xy = (xy - xy.mean(0)) / np.abs(xy - xy.mean(0)).max()
    return np.c_[xy, np.hypot(*xy.T), [len(a) / 6 for a in adj]]


@cache
def _lattice() -> tuple:
    """The lattice's neighbour lists, the start trap and the trap features."""
    coords = load_coords()
    adj, _ = lattice_adjacency(coords)
    return adj, code_start_trap(len(adj)), trap_features(coords, adj)


def pair_options(g: nx.Graph, place: dict, occ: dict, adj: list[list[int]], trap_x: np.ndarray) -> list[tuple]:
    """Every feasible (vertex, trap, features), ordered by trap then vertex. `place` maps vertex to trap, `occ` trap
    to vertex."""
    nbrs = g._adj
    out = []
    n, size = len(g), len(place)
    for t in sorted({x for s in occ for x in adj[s] if x not in occ}):
        near = {occ[x] for x in adj[t] if x in occ}
        cands = None
        for v in near:
            cands = set(nbrs[v]) if cands is None else cands & set(nbrs[v])
        free = len(adj[t]) - len(near)
        for u in sorted(cands or ()):
            if u not in place and nbrs[u].keys() & place.keys() == near:
                d = len(nbrs[u])
                out.append((u, t, (*trap_x[t], len(near) / 6, free / 6, size / 61, d / n, (d - len(near)) / n,
                                   d / max(1, n - size))))
    return out


class PieceBuilder(nn.Module):
    def __init__(self, body: GNN):
        super().__init__()
        self.body = body
        hidden = body.config["hidden"]
        self.pair = nn.Sequential(nn.Linear(hidden + len(PAIR_FEATURES), hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def embed(self, batch) -> torch.Tensor:
        """The body's vertex embeddings: its layers without the output head."""
        b = self.body
        h = b.embed((batch.x - b.mu) / b.sd)
        for conv, norm in zip(b.convs, b.norms):
            h = norm(h + torch.relu(conv(h, batch.edge_index)))
        return h

    def pair_logits(self, h: torch.Tensor, idx, phi) -> torch.Tensor:
        """One logit per pair: idx are the vertices' rows of h, phi their (pairs, 10) features."""
        return self.pair(torch.cat([h[torch.as_tensor(idx)], torch.as_tensor(phi)], 1)).squeeze(1)


def pair_weights(model: PieceBuilder) -> tuple:
    """The four numpy arrays of the pair scorer: first weight and bias, second weight and bias."""
    a, b = model.pair[0], model.pair[2]
    return tuple(x.detach().numpy().astype(np.float32) for x in (a.weight, a.bias, b.weight, b.bias))


def build_piece(g: nx.Graph, H: np.ndarray, weights: tuple, start, adj: list[list[int]], start_trap: int,
                trap_x: np.ndarray, rng: np.random.Generator | None = None) -> tuple[dict, list]:
    """One piece from `start` on `start_trap`. H holds the vertex embeddings in g order. Without rng it takes the
    highest logit, the first on a tie, and returns no decisions; with rng it samples (Gumbel noise on the logits)
    and returns the decisions as (rows of H, features, index chosen). Logits are numpy: nothing here is autograd."""
    at = {v: i for i, v in enumerate(g)}
    place, occ, decisions = {start: start_trap}, {start_trap: start}, []
    w1, b1, w2, b2 = weights
    while options := pair_options(g, place, occ, adj, trap_x):
        idx = np.fromiter((at[o[0]] for o in options), int, len(options))
        phi = np.array([o[2] for o in options], np.float32)
        logit = (np.maximum(np.c_[H[idx], phi] @ w1.T + b1, 0) @ w2.T + b2).ravel()
        if rng is None:
            chosen = int(np.argmax(logit))
        else:
            chosen = int(np.argmax(logit + rng.gumbel(size=len(logit))))
            decisions.append((idx, phi, chosen))
        u, t = options[chosen][:2]
        place[u], occ[t] = t, u
    return place, decisions


def builder_hook(model: PieceBuilder) -> Builder:
    """Build hook: one greedy piece from every vertex of g, in g order, from one embedding pass."""
    adj, start_trap, trap_x = _lattice()
    weights = pair_weights(model)

    def build(g: nx.Graph) -> list[dict]:
        if not len(g):
            return []
        with torch.inference_mode():
            H = model.embed(encode(g).to("cpu")).numpy()
        return [build_piece(g, H, weights, v, adj, start_trap, trap_x)[0] for v in g]
    return build


def _saved(model: PieceBuilder) -> dict:
    return {"config": model.body.config, "state": model.state_dict()}


def _atomic_save(obj, path) -> None:
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_builder(model: PieceBuilder, path) -> None:
    _atomic_save(_saved(model), path)


def _from_saved(saved: dict) -> PieceBuilder:
    model = PieceBuilder(GNN(**saved["config"]))
    model.load_state_dict(saved["state"])
    return model.eval()


def load_builder(path) -> PieceBuilder:
    return _from_saved(torch.load(path, weights_only=True))


def blob_of(model: PieceBuilder) -> bytes:
    """The model as bytes, for a worker to rebuild it."""
    b = io.BytesIO()
    torch.save(_saved(model), b)
    return b.getvalue()


def clone_examples(rec: dict, scorer_factory, adj: list[list[int]], start_trap: int, trap_x: np.ndarray,
                   starts: int = 8) -> tuple:
    """(n, edges, decisions) of one state: for up to `starts` vertices, the growth of `scorer_factory(g)` replayed
    placement by placement, each after the first as (rows of the options, their features, index of the expert's pair)."""
    from .graphdata import graph_of
    g = graph_of(rec)    # its vertices are 0..n-1 in order, so a vertex is its own row of the embedding
    scorer, decisions = scorer_factory(g), []
    for s in random.Random(rec["n"]).sample(list(g), min(starts, len(g))):
        place, occ = {}, {}
        for u, t in grow_piece(g, s, adj, start_trap, random.Random(0), scorer).items():
            if place:
                options = pair_options(g, place, occ, adj, trap_x)
                chosen = next(i for i, o in enumerate(options) if o[:2] == (u, t))
                decisions.append((np.array([o[0] for o in options]), np.array([o[2] for o in options], np.float32), chosen))
            place[u], occ[t] = t, u
    return rec["n"], rec["edges"], decisions


def decision_loss(model: PieceBuilder, n: int, edges: np.ndarray, decisions: list, advantages) -> tuple:
    """(minus the advantage-weighted log softmax of each chosen pair over its options, averaged over the decisions,
    the options' logits), from one embedding pass. The logits are detached."""
    from .graphdata import graph_of
    h = model.embed(encode(graph_of(dict(n=n, edges=edges))).to("cpu"))
    sizes = [len(d[0]) for d in decisions]
    logits = model.pair_logits(h, np.concatenate([d[0] for d in decisions]), np.concatenate([d[1] for d in decisions]))
    group = torch.repeat_interleave(torch.arange(len(sizes)), torch.as_tensor(sizes))
    peak = logits.detach().new_full((len(sizes),), -torch.inf).scatter_reduce(0, group, logits.detach(), "amax")
    log_z = logits.new_zeros(len(sizes)).index_add_(0, group, (logits - peak[group]).exp()).log() + peak
    chosen = torch.as_tensor(np.cumsum([0] + sizes[:-1]) + [d[2] for d in decisions])
    adv = torch.as_tensor(np.asarray(advantages), dtype=logits.dtype)
    return -(adv * (logits[chosen] - log_z)).sum() / len(sizes), logits.detach()


def clone_fit(model: PieceBuilder, examples: list, epochs: int, seed: int, lr: float = 1e-3, batch: int = 16,
              log=None) -> list[tuple]:
    """Adam over the examples in batches of `batch` states, gradient clip 5. One (loss, agreement) per epoch: the
    mean state loss, and the share of decisions where the highest logit is the expert's pair."""
    examples = [e for e in examples if e[2]]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        order = random.Random(seed * 1000 + epoch).sample(examples, len(examples))
        total = hits = count = 0
        for i in range(0, len(order), batch):
            opt.zero_grad()
            for n, edges, ds in order[i:i + batch]:
                loss, logits = decision_loss(model, n, edges, ds, np.ones(len(ds)))
                (loss / len(order[i:i + batch])).backward()
                at = np.cumsum([0] + [len(d[0]) for d in ds])
                hits += sum(int(logits[a:b].argmax()) == d[2] for a, b, d in zip(at, at[1:], ds))
                total, count = total + loss.item(), count + len(ds)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        history.append((total / len(order), hits / count))
        if log:
            log(f"epoch {epoch + 1}: loss {history[-1][0]:.4f} agreement {history[-1][1]:.3f}")
    return history


def one_step_reward(g: nx.Graph, piece: dict, value, s: int = 2, small: int = 5) -> float:
    """Over the engine's own sets of the piece (the first maximum set up to `small` vertices, else the top `s`
    maximal sets), the best of the set's size plus the value of what is left of g."""
    sub = g.subgraph(piece)
    sets = [first_maximum_set(sub)] if len(piece) <= small else top_maximal_sets(sub, s)
    return max(len(S) + float(v) for S, v in zip(sets, value([_remove_closed(g, S) for S in sets])))


def start_advantages(rewards, starts) -> np.ndarray:
    """Each reward minus the mean of the rewards with its start key, over the whole batch, divided by the standard
    deviation of those differences (at least 1e-3)."""
    r = np.asarray(rewards, float)
    ids = {k: i for i, k in enumerate(dict.fromkeys(starts))}
    group = np.array([ids[k] for k in starts], int)
    centred = r - (np.bincount(group, r) / np.bincount(group))[group]
    return centred / max(centred.std(), 1e-3) if len(r) else centred


def rollout_job(job: tuple) -> tuple:
    """job = (blob, n, edges, starts, samples, seed, value name, rank name, models dir). Builds a greedy piece from
    every vertex, keeps the `starts` vertices the rank model scores highest, and samples `samples` pieces from each.
    Returns (n, edges, rows), a row being (start, reward, piece size, decisions). Numpy only."""
    from .bench import method_hooks
    from .graphdata import graph_of
    blob, n, edges, starts, samples, seed, value_name, rank_name, models_dir = job
    g = graph_of(dict(n=n, edges=edges))
    if not len(g):
        return n, edges, []
    hooks = method_hooks(f"{rank_name}+{value_name}", models_dir)
    model = _from_saved(torch.load(io.BytesIO(blob), weights_only=True))
    adj, start_trap, trap_x = _lattice()
    weights = pair_weights(model)     # after the load: the weights are the blob's
    with torch.inference_mode():
        H = model.embed(encode(g).to("cpu")).numpy()
    verts = list(g)
    ranks = hooks["rank"](g, [build_piece(g, H, weights, v, adj, start_trap, trap_x)[0] for v in verts])
    chosen = [verts[i] for i in sorted(range(len(verts)), key=ranks.__getitem__, reverse=True)[:starts]]
    rng, rows = np.random.default_rng(seed), []
    for v in chosen:
        for _ in range(samples):
            piece, decisions = build_piece(g, H, weights, v, adj, start_trap, trap_x, rng)
            rows.append((v, one_step_reward(g, piece, hooks["value"]), len(piece), decisions))
    return n, edges, rows


def _expert_job(job: tuple) -> tuple:
    from .bench import method_hooks
    rec, expert, models_dir = job
    return clone_examples(rec, method_hooks(expert, models_dir)["score"], *_lattice())


def _open_pool(workers: int):
    """A spawn pool of one-thread workers, or None for a single worker, which runs in this process. A worker that dies
    raises BrokenProcessPool from the map instead of leaving it waiting."""
    return (ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                initializer=torch.set_num_threads, initargs=(1,)) if workers > 1 else None)


def _map(pool, fn, jobs: list) -> list:
    return list(pool.map(fn, jobs, chunksize=2)) if pool else list(map(fn, jobs))


def _close(pool) -> None:
    if pool:
        pool.shutdown(wait=True)     # children's CPU is counted only once they are waited for


def _refuse_changed(saved: dict, now: dict) -> None:
    """A resume continues the run it saved, so a setting that differs from the saved one is an error."""
    for name, old in saved.items():
        if now[name] != old:
            raise ValueError(f"resume with changed {name}: saved {old!r}, now {now[name]!r}")


def _training_states(a) -> list[dict]:
    from .train import load_pickles
    paths = sorted(p for pattern in a.train for p in glob.glob(pattern))
    if not paths:
        raise ValueError(f"no file matches {a.train}")
    states = load_pickles(paths)[1]
    if outside := {r["seed"] for r in states} - set(TRAIN_SEEDS):
        raise ValueError(f"seeds {sorted(outside)} are not training seeds")
    return random.Random(0).sample(states, a.max_states) if a.max_states and len(states) > a.max_states else states


def _copy_companions(a, dest: Path) -> None:
    """The rank and value model files beside the builder, so `--models dest` resolves bld-h3+<rank>+<value>."""
    from .bench import KIND_EXT
    dest.mkdir(parents=True, exist_ok=True)
    for name in (a.rank, a.value):
        shutil.copy2(Path(a.models) / f"{name}{KIND_EXT[name.split('-')[0]]}", dest)


def _write_meta(meta: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, path)


def clone(a) -> None:
    from .az import _cpu_s
    from .dataset import _commit
    start, cpu = time.perf_counter(), _cpu_s()
    states = _training_states(a)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    model = PieceBuilder(load_gnn(Path(a.models) / f"{a.expert}.pt"))
    pool = _open_pool(a.workers)
    examples = _map(pool, _expert_job, [(dict(n=r["n"], edges=r["edges"]), a.expert, a.models) for r in states])
    _close(pool)
    torch.set_num_threads(a.threads)      # a run without workers loaded the expert here, at one thread
    decisions = sum(len(e[2]) for e in examples)
    print(f"{decisions} expert decisions from {len(states)} states", flush=True)
    history = clone_fit(model, examples, a.epochs, a.seed, log=lambda m: print(f"  {m}", flush=True))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_builder(model.eval(), out / "bld-h3.pt")
    _copy_companions(a, out)
    meta = dict(expert=a.expert, rank=a.rank, value=a.value, seed=a.seed, epochs=a.epochs, states=len(states),
                decisions=decisions, loss=[h[0] for h in history], agreement=[h[1] for h in history],
                train=[str(p) for p in a.train], max_states=a.max_states, cpu_s=_cpu_s() - cpu,
                wall_s=time.perf_counter() - start, commit=_commit())
    _write_meta(meta, out / "train_meta_bld-h3.json")
    print(f"cloned {a.expert}: agreement {history[0][1]:.3f} -> {history[-1][1]:.3f}, cpu {meta['cpu_s']:.0f} s, "
          f"wall {meta['wall_s']:.0f} s", flush=True)


def _rl_step(model, opt, pool, a, t: int, states: list[dict]) -> dict:
    began, seed = time.perf_counter(), a.seed * 100003 + t
    batch = random.Random(seed).sample(states, a.states)
    blob = blob_of(model)
    found = _map(pool, rollout_job, [(blob, r["n"], r["edges"], a.starts, a.samples, seed * 1000 + i, a.value, a.rank,
                                      a.models) for i, r in enumerate(batch)])
    torch.set_num_threads(a.threads)      # a rollout without workers leaves torch at one thread
    rows = [(i, row) for i, (_, _, rs) in enumerate(found) for row in rs]
    adv = start_advantages([row[1] for _, row in rows], [(i, row[0]) for i, row in rows])
    opt.zero_grad()
    total, at = 0.0, 0
    for n, edges, rs in found:
        mine, at = adv[at:at + len(rs)], at + len(rs)
        decisions = [d for row in rs for d in row[3]]
        if decisions:
            loss, _ = decision_loss(model, n, edges, decisions, np.repeat(mine, [len(row[3]) for row in rs]))
            (loss / len(found)).backward()
            total += loss.item() / len(found)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
    opt.step()
    mean = lambda xs: float(np.mean(xs)) if len(xs) else 0.0
    return dict(iter=t, reward=mean([r[1] for _, r in rows]), size=mean([r[2] for _, r in rows]),
                decisions=sum(len(r[3]) for _, r in rows), loss=total, grad_norm=norm,
                seconds=time.perf_counter() - began)


def rl(a) -> None:
    from .az import _cpu_s
    from .dataset import _commit
    states = [r for r in _training_states(a) if r["n"] >= a.min_n]
    if len(states) < a.states:
        raise ValueError(f"{len(states)} states of at least {a.min_n} nodes, fewer than the {a.states} an iteration needs")
    torch.set_num_threads(a.threads)
    model = load_builder(a.init)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, before = 0, 0.0
    settings = dict(seed=a.seed, lr=a.lr, init=str(a.init), rank=a.rank, value=a.value, states=a.states, starts=a.starts,
                    samples=a.samples, min_n=a.min_n, train=[str(p) for p in a.train], max_states=a.max_states)
    if (out / "state.pt").exists():
        saved = torch.load(out / "state.pt", weights_only=True)
        _refuse_changed(saved.get("settings", {}), settings)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        t, before = saved["t"], saved["cpu_s"]
    log = out / "log.jsonl"      # lines past the last saved state are redone
    kept = [l for l in log.read_text().splitlines() if json.loads(l)["iter"] <= t] if log.exists() else []
    log.write_text("".join(l + "\n" for l in kept))
    cpu = _cpu_s()
    while t < a.iters:
        stop = min((t // a.every + 1) * a.every, a.iters)
        pool = _open_pool(a.workers)       # one pool per stretch between checkpoints, so its CPU is counted at each
        for step in range(t + 1, stop + 1):
            row = _rl_step(model, opt, pool, a, step, states)
            with open(log, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"rl {step}: reward {row['reward']:.3f} size {row['size']:.1f} loss {row['loss']:.4f} "
                  f"{row['seconds']:.0f} s", flush=True)
        _close(pool)
        t, spent = stop, before + _cpu_s() - cpu
        meta = dict(iter=t, iters=a.iters, init=str(a.init), rank=a.rank, value=a.value, seed=a.seed, lr=a.lr,
                    states=a.states, starts=a.starts, samples=a.samples, min_n=a.min_n, train=[str(p) for p in a.train],
                    max_states=a.max_states, cpu_s=spent, commit=_commit())
        if t % a.every == 0:
            _copy_companions(a, out / f"it{t}")
            save_builder(model, out / f"it{t}" / "bld-h3.pt")
            _write_meta(meta, out / f"it{t}" / "train_meta_bld-h3.json")
        _atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), t=t, cpu_s=spent, settings=settings),
                     out / "state.pt")
        _write_meta(meta, out / "train_meta_bld-h3.json")
    print(f"rl: {t} iterations, cpu {before + _cpu_s() - cpu:.0f} s", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("clone", "rl"):
        p = sub.add_parser(name)
        p.add_argument("--models", default="models")
        p.add_argument("--rank", default="dgin-h4")
        p.add_argument("--value", default="dgin-h5")
        p.add_argument("--train", nargs="+", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True)
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--max-states", type=int, help="use a seeded sample of this many states")
        if name == "clone":
            p.add_argument("--expert", default="dgin-h2")
            p.add_argument("--epochs", type=int, default=10)
        else:
            p.add_argument("--init", required=True)
            p.add_argument("--iters", type=int, default=400)
            p.add_argument("--states", type=int, default=128)
            p.add_argument("--starts", type=int, default=8)
            p.add_argument("--samples", type=int, default=2)
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--min-n", type=int, default=20)
            p.add_argument("--every", type=int, default=100)
    a = ap.parse_args(argv)
    {"clone": clone, "rl": rl}[a.command](a)


if __name__ == "__main__":
    main()
