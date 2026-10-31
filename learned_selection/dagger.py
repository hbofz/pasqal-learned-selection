"""Retrain a stack of models on the states it visits itself, iteration by iteration."""
import argparse, pickle, subprocess, sys, time
from pathlib import Path

STACKS = {"lin": dict(start="lin-h2h4+sage-h5", policy="dlin-h2h4+dsage-h5", jobs="dlin-h2,dlin-h4,dsage-h5"),
          "gin": dict(start="gin-h2h4h5", policy="dgin-h2h4h5", jobs="dgin-h2,dgin-h4,dgin-h5"),
          "bld": dict(start="bld-h3+dgin-h4+dgin-h5", policy="bld-h3+bgin-h4+bgin-h5", jobs="bgin-h4,bgin-h5")}


def commands(stack, t, base_train, base_val, out, cache, workers, threads, max_states=8, shard=None):
    """The argv of each step of iteration t: label train, label val, train."""
    s = STACKS[stack]
    policy, models = (s["start"], "models") if t == 1 else (s["policy"], f"{out}/it{t - 1}")
    py = [sys.executable, "-m"]

    def label(split):
        return [*py, "learned_selection.graphdata", "--split", split, "--workers", str(workers),
                "--out", f"{cache}/it{t}_{split}.pkl", "--policy", policy, "--models", models,
                "--max-states", str(max_states), *(["--shard", shard] if shard else [])]

    def every(base, split):
        return [*base, *(f"{cache}/it{i}_{split}.pkl" for i in range(1, t + 1))]

    train = [*py, "learned_selection.train", "--train", *every(base_train, "train"), "--val", *every(base_val, "val"),
             "--jobs", s["jobs"], "--out", f"{out}/it{t}", "--threads", str(threads), "--workers", str(workers)]
    return [label("train"), label("val"), train]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=list(STACKS), required=True)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--max-states", type=int, default=8)
    ap.add_argument("--label-shard", metavar="I/N", help="label every N-th instance starting at I (smoke runs)")
    ap.add_argument("--base-train", nargs="+", required=True)
    ap.add_argument("--base-val", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--cache")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args(argv)
    out, cache = a.out or f"models/dagger/{a.stack}", a.cache or f"cache/dagger/{a.stack}"
    last_job = STACKS[a.stack]["jobs"].split(",")[-1]
    for t in range(1, a.iters + 1):
        cmds = commands(a.stack, t, a.base_train, a.base_val, out, cache, a.workers, a.threads, a.max_states, a.label_shard)
        outputs = [f"{cache}/it{t}_train.pkl", f"{cache}/it{t}_val.pkl", f"{out}/it{t}/train_meta_{last_job}.json"]
        for cmd, path in zip(cmds, outputs):      # one process per step, so torch and the pools stay apart
            if Path(path).exists():
                print(f"it{t}: {path} exists, skipped", flush=True)
                continue
            start = time.time()
            subprocess.run(cmd, check=True)
            cpu = ""
            if path.endswith(".pkl") and Path(path).exists():
                cpu = f", label cpu {pickle.load(open(path, 'rb'))['meta']['label_cpu_s']:.0f}s"
            print(f"it{t}: {Path(path).name} took {time.time() - start:.0f}s{cpu}", flush=True)


if __name__ == "__main__":
    main()
