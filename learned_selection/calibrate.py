"""Calibrate the exact classical stand-in against a Pasqal cloud emulator.

    uv run python -m learned_selection.calibrate [--set small|medium]

Pieces: the first-round GLS candidates (k=4) of the graphs in SETS (small: 11-20 atoms, medium: 25-26). Each piece is
solved exactly and on the emulator, and also fed uniformly random bitstrings as a control, because repair alone
can turn noise into an optimal set on small pieces. Decisions therefore rest on raw (pre-repair) statistics.
"""
import argparse, json, random, statistics, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cloud import EMULATION_DEVICE, CloudPieceSolver, client, summarize_counts
from .engine import gls_generator
from .instances import erdos_renyi
from .solvers import top_maximal_sets

RUNS, S = 500, 2
MAX_ATOMS = 30   # AnalogDevice: max layout filling 0.5 of 61 traps
SETS = {"small": [(60, p, seed) for p in (0.25, 0.5) for seed in range(5)],
        "medium": [(300, 0.25, seed) for seed in range(10)]}
OUT = {"small": Path("results/calibration.jsonl"), "medium": Path("results/calibration_medium.jsonl")}
_local = threading.local()


def pieces(graphs):
    for n, p, seed in graphs:
        G = erdos_renyi(n, p, seed)
        for i, m in enumerate(gls_generator()(G, 4)):
            yield dict(n=n, p=p, seed=seed, candidate=i), G.subgraph(m).copy(), m


def jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def raw_opt_rate(stats, opt):
    return stats["valid_sizes"].get(opt, 0) / stats["shots"]


def calibrate_one(item):
    meta, piece, m = item
    if not hasattr(_local, "solver"):
        _local.solver = CloudPieceSolver(client(), EMULATION_DEVICE, runs=RUNS)
    exact = top_maximal_sets(piece, S)
    opt = len(exact[0])
    emu_top = _local.solver(piece, m, S)
    emu = _local.solver.log[-1]
    rng = random.Random(hash((meta["p"], meta["seed"], meta["candidate"])) & 0xFFFF)
    noise = Counter("".join(rng.choice("01") for _ in piece) for _ in range(RUNS))
    rnd_top, rnd = summarize_counts(piece, noise, [str(v) for v in piece], S)
    return dict(**meta, n_atoms=len(piece), opt=opt, exact_top_sizes=[len(x) for x in exact],
                top1_jaccard=jaccard(exact[0], emu_top[0]),
                emu_top_sizes=[len(x) for x in emu_top], emu_invalid_rate=emu["invalid_rate"],
                emu_raw_opt_rate=raw_opt_rate(emu, opt), rnd_top_sizes=[len(x) for x in rnd_top],
                rnd_invalid_rate=rnd["invalid_rate"], rnd_raw_opt_rate=raw_opt_rate(rnd, opt),
                emulator="EMU_MPS", device=EMULATION_DEVICE.name, runs=RUNS, batch_id=emu["batch_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="small")
    which = ap.parse_args().set
    items = list(pieces(SETS[which]))
    todo = [it for it in items if len(it[1]) <= MAX_ATOMS]
    print(f"{len(items)} pieces, {len(items) - len(todo)} skipped for exceeding {MAX_ATOMS} atoms", flush=True)
    OUT[which].parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool, OUT[which].open("a") as f:
        for row in pool.map(calibrate_one, todo):
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            print(f"{len(rows)}/{len(todo)} n_atoms={row['n_atoms']} emu_raw_opt={row['emu_raw_opt_rate']:.2f} "
                  f"rnd_raw_opt={row['rnd_raw_opt_rate']:.2f}", flush=True)

    share = lambda pred: sum(map(pred, rows)) / len(rows)
    emu_top1 = share(lambda r: r["emu_top_sizes"][0] == r["opt"])
    rnd_top1 = share(lambda r: r["rnd_top_sizes"][0] == r["opt"])
    beats = share(lambda r: r["emu_raw_opt_rate"] > r["rnd_raw_opt_rate"])
    print(f"\npieces {len(rows)}, atoms {min(r['n_atoms'] for r in rows)}-{max(r['n_atoms'] for r in rows)}")
    print(f"top-1 repaired set optimal: emulator {emu_top1:.0%}, random control {rnd_top1:.0%}")
    print(f"raw optimal-shot rate, mean: emulator {statistics.mean(r['emu_raw_opt_rate'] for r in rows):.3f}, "
          f"random {statistics.mean(r['rnd_raw_opt_rate'] for r in rows):.3f}; emulator beats random on {beats:.0%}")
    print(f"raw invalid-shot rate, mean: emulator {statistics.mean(r['emu_invalid_rate'] for r in rows):.3f}")
    print(f"top-1 set overlap (Jaccard) exact vs emulator, mean: {statistics.mean(r['top1_jaccard'] for r in rows):.3f}")
    print("decision rule: stand-in OK if emulator top-1 optimal >= 90% AND emulator beats random >= 90%")
    lo, hi = min(r["n_atoms"] for r in rows), max(r["n_atoms"] for r in rows)
    print("verdict:", f"stand-in OK for pieces of {lo}-{hi} atoms only" if emu_top1 >= 0.9 and beats >= 0.9
          else f"stand-in NOT validated for {lo}-{hi} atoms")
    if rnd_top1 >= 0.9:
        print("note: random bitstrings plus repair also reach the optimum, so repair, not the quantum solve, "
              "does most of the work at these piece sizes")


if __name__ == "__main__":
    main()
