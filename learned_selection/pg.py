"""Policy gradient on the piece ranker (`pg-h4`), reward = final set size - c * quantum runs.

A rollout runs the bounded code loop once, ranking candidate pieces by the network's score plus Gumbel noise, which
draws the ordered top k with the probability `top_k_logp` gives. Each iteration rolls out several samples of a batch
of training instances, centres every reward on its instance's mean, and takes one Adam step on the log probability of
the choices made. The other hooks (`--stack`) stay fixed. Rollout and training compute are a one-time classical
cost: it is printed and saved, never added to a benchmark's per-instance runs.

    uv run python -m learned_selection.pg --init models/dgin-h4.pt --stack dgin-h2+dgin-h5 --models models \\
        --c 0.01 --out cache/pg
"""
import argparse, functools, io, json, multiprocessing, os, random, shutil, time
from pathlib import Path

import numpy as np
import torch

from .az import _cpu_s
from .bench import KIND_EXT, method_hooks, parse_method
from .dataset import _commit
from .engine import run_code_bounded
from .gnn import encode, load_gnn, save_gnn
from .instances import TRAIN_SEEDS, TRAIN_SETTINGS, make

PG_SIZES = (60, 100)
PG_K, PG_S, PG_L = 4, 2, 10
MAX_DECISIONS = 16
GRAD_CLIP = 5.0


def training_instances() -> list[tuple]:
    """(family, params, n, seed) over the training settings, sizes and seeds."""
    return [(f, p, n, s) for f, p in TRAIN_SETTINGS for n in PG_SIZES for s in TRAIN_SEEDS]


def top_k_logp(scores: torch.Tensor, order) -> torch.Tensor:
    """Log probability of drawing `order` as the top of the list: each chosen score minus the logsumexp of the scores
    not yet chosen. Adding Gumbel noise to the scores and sorting draws exactly this."""
    left = torch.ones(len(scores), dtype=torch.bool)
    logp = scores.new_zeros(())
    for i in map(int, order):
        logp = logp + scores[i] - torch.logsumexp(scores[left], 0)
        left = left.clone()   # the mask is saved for backward, so it is not changed in place
        left[i] = False
    return logp


def _blob(net) -> bytes:
    buf = io.BytesIO()
    save_gnn(net, buf)
    return buf.getvalue()


@functools.lru_cache(maxsize=None)
def _hooks(stack, models_dir):
    return method_hooks(stack, models_dir)   # loaded once per worker


def rollout_job(job: tuple) -> dict:
    """job = (blob, stack, models_dir, family, params, n, seed, rseed, c). One bounded run with a noisy ranker; the
    network is in `blob`, the other hooks are named by `stack`. Numpy only."""
    blob, stack, models_dir, family, params, n, seed, rseed, c = job
    if seed not in TRAIN_SEEDS:
        raise ValueError(f"seed {seed} is not a training seed; rollouts must not use it")
    net, noise, kept = load_gnn(io.BytesIO(blob)), np.random.default_rng(rseed), []

    def rank(g, maps):
        b = encode(g, maps)
        with torch.inference_mode():
            r = net(b.to("cpu")).double().numpy() + noise.gumbel(size=len(maps))
        kept.append((b, np.argsort(-r, kind="stable")[:PG_K]))
        return r
    res = run_code_bounded(make(family, n, seed, **params), PG_K, PG_S, PG_L, rank=rank, seed=0,
                           **_hooks(stack, models_dir))
    keep = sorted(random.Random(rseed).sample(range(len(kept)), min(MAX_DECISIONS, len(kept))))
    return dict(key=(family, json.dumps(params), n, seed), reward=res.size - c * res.quantum_runs, size=res.size,
                runs=res.quantum_runs, decisions=[kept[i] for i in keep], calls=len(kept))


def advantages(results: list[dict]) -> list[float]:
    """Each reward minus the mean of its instance's rollouts, over the std of those differences across the batch."""
    by = {}
    for r in results:
        by.setdefault(r["key"], []).append(r["reward"])
    diff = np.array([r["reward"] - np.mean(by[r["key"]]) for r in results])
    return (diff / max(diff.std(), 1e-3)).tolist()


def pg_update(net, opt, results: list[dict], grad_clip: float = GRAD_CLIP) -> dict:
    """One Adam step on minus the advantage-weighted log probability of the kept decisions, each rollout's decisions
    weighted by calls / kept so a sample stands for all its generate calls. Rollouts with no decisions or no
    advantage add nothing, and a batch with none leaves the network alone."""
    opt.zero_grad()
    used = 0
    for r, adv in zip(results, advantages(results)):
        if not r["decisions"] or adv == 0:
            continue
        weight = r["calls"] / len(r["decisions"])
        for b, order in r["decisions"]:
            (-weight * adv * top_k_logp(net(b.to("cpu")), order) / len(results)).backward()
            used += 1
    if used:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()
    return dict(decisions=used)


def _atomic(write, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="models/dgin-h4.pt")
    ap.add_argument("--stack", default="dgin-h2+dgin-h5", help="the model names supplying the h2 and h5 hooks")
    ap.add_argument("--models", default="models")
    ap.add_argument("--c", type=float, default=0.01, help="cost per quantum run")
    ap.add_argument("--iters", type=int, default=160)
    ap.add_argument("--every", type=int, default=40, help="save a model and the training state every this many")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--instances", type=int, default=48)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args(argv)
    if {h for _, h in parse_method(a.stack)} != {"h2", "h5"}:
        ap.error("--stack must name an h2 and an h5 model")
    net = load_gnn(a.init)
    if net.head != "piece":
        ap.error(f"{a.init} has a {net.head!r} head, but the policy needs a 'piece' head")
    torch.set_num_threads(a.threads)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    out, models = Path(a.out), Path(a.models)
    out.mkdir(parents=True, exist_ok=True)
    t, cpu0, wall0 = 0, 0.0, 0.0
    if (out / "state.pt").exists():
        saved = torch.load(out / "state.pt", weights_only=True)
        net.load_state_dict(saved["net"])
        opt.load_state_dict(saved["opt"])
        t, cpu0, wall0 = saved["t"], saved["cpu"], saved["wall"]
    log = out / "log.jsonl"   # drop the rows of iterations a lost run did not get to save
    rows = [r for r in map(json.loads, log.read_text().splitlines()) if r["t"] <= t] if log.exists() else []
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    start, cpu_start = time.perf_counter(), _cpu_s()

    def save_model():
        dest = out / f"it{t}"
        dest.mkdir(exist_ok=True)
        _atomic(lambda p: save_gnn(net, p), dest / "pg-h4.pt")
        for kind, hook in parse_method(a.stack):
            if kind != "exact":
                shutil.copy2(models / f"{kind}-{hook}{KIND_EXT[kind]}", dest)

    def save_state():
        state = dict(net=net.state_dict(), opt=opt.state_dict(), t=t, cpu=cpu0 + _cpu_s() - cpu_start,
                     wall=wall0 + time.perf_counter() - start)
        _atomic(lambda p: torch.save(state, p), out / "state.pt")
        return state

    def new_pool():
        return multiprocessing.get_context("spawn").Pool(a.workers, torch.set_num_threads, (1,)) if a.workers > 1 else None

    def stop(pool):
        if pool:
            pool.close()
            pool.join()   # a child's CPU is counted only once it has been waited for

    pool = new_pool() if t < a.iters else None
    while t < a.iters:
        t += 1
        step = time.perf_counter()
        base = a.seed * 100003 + t
        batch = random.Random(base).sample(training_instances(), a.instances)
        blob = _blob(net)
        jobs = [(blob, a.stack, models, f, p, n, s, base * 1000 + i * 10 + j, a.c)
                for i, (f, p, n, s) in enumerate(batch) for j in range(a.rollouts)]
        results = (pool.map(rollout_job, jobs, chunksize=2) if pool else list(map(rollout_job, jobs)))
        row = dict(t=t, size=float(np.mean([r["size"] for r in results])), runs=float(np.mean([r["runs"] for r in results])),
                   reward=float(np.mean([r["reward"] for r in results])), **pg_update(net, opt, results),
                   seconds=time.perf_counter() - step)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"it {t}: size {row['size']:.2f} runs {row['runs']:.1f} reward {row['reward']:.3f} "
              f"decisions {row['decisions']} {row['seconds']:.1f} s", flush=True)
        if t % a.every == 0:
            stop(pool)   # so the saved CPU total includes the workers; they hold no state, a fresh pool follows
            save_model()
            save_state()
            pool = new_pool() if t < a.iters else None
    stop(pool)
    if t % a.every:
        save_model()
    state = save_state()
    (out / "train_meta_pg-h4.json").write_text(json.dumps(
        dict(c=a.c, seed=a.seed, cpu_s=state["cpu"], wall_s=state["wall"], commit=_commit()), indent=2))
    print(f"{t} iterations, cpu {state['cpu']:.0f} s, wall {state['wall']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
