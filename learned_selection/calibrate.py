"""Calibrate the exact classical stand-in against a Pasqal cloud emulator (spec section 5).

    uv run python -m learned_selection.calibrate

Pieces: the first-round GLS candidates (k=4) of Erdos-Renyi n=60, p in {0.25, 0.5}, seeds 0-4. Each piece is
solved exactly and on the emulator, and also fed uniformly random bitstrings as a control, because repair alone
can turn noise into an optimal set on small pieces. Decisions therefore rest on raw (pre-repair) statistics.
"""
import json, random, statistics, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cloud import EMULATION_DEVICE, CloudPieceSolver, client, summarize_counts
from .engine import gls_generator
from .instances import erdos_renyi
from .solvers import top_maximal_sets

RUNS, S = 500, 2
MAX_ATOMS = 30   # AnalogDevice: max layout filling 0.5 of 61 traps
OUT = Path("results/calibration.jsonl")
_local = threading.local()


def pieces():
    for p in (0.25, 0.5):
        for seed in range(5):
            G = erdos_renyi(60, p, seed)
            for i, m in enumerate(gls_generator()(G, 4)):
                yield dict(p=p, seed=seed, candidate=i), G.subgraph(m).copy(), m


def raw_opt_rate(stats, opt):
    return stats["valid_sizes"].get(opt, 0) / stats["shots"]


def calibrate_one(item):
    meta, piece, m = item
    if not hasattr(_local, "solver"):
        _local.solver = CloudPieceSolver(client(), EMULATION_DEVICE, runs=RUNS)
    opt = len(top_maximal_sets(piece, 1)[0])
    emu_top = _local.solver(piece, m, S)
    emu = _local.solver.log[-1]
    rng = random.Random(hash((meta["p"], meta["seed"], meta["candidate"])) & 0xFFFF)
    noise = Counter("".join(rng.choice("01") for _ in piece) for _ in range(RUNS))
    rnd_top, rnd = summarize_counts(piece, noise, [str(v) for v in piece], S)
    return dict(**meta, n_atoms=len(piece), opt=opt, exact_top_sizes=[len(x) for x in top_maximal_sets(piece, S)],
                emu_top_sizes=[len(x) for x in emu_top], emu_invalid_rate=emu["invalid_rate"],
                emu_raw_opt_rate=raw_opt_rate(emu, opt), rnd_top_sizes=[len(x) for x in rnd_top],
                rnd_invalid_rate=rnd["invalid_rate"], rnd_raw_opt_rate=raw_opt_rate(rnd, opt),
                emulator="EMU_MPS", device=EMULATION_DEVICE.name, runs=RUNS, batch_id=emu["batch_id"])


def main():
    items = list(pieces())
    todo = [it for it in items if len(it[1]) <= MAX_ATOMS]
    print(f"{len(items)} pieces, {len(items) - len(todo)} skipped for exceeding {MAX_ATOMS} atoms", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool, OUT.open("a") as f:
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
    print("decision rule (my call): stand-in OK if emulator top-1 optimal >= 90% AND emulator beats random >= 90%")
    print("verdict:", "stand-in OK" if emu_top1 >= 0.9 and beats >= 0.9 else "REPORT TO HAMZAH: stand-in not validated")
    if rnd_top1 >= 0.9:
        print("note: random bitstrings plus repair also reach the optimum, so repair, not the quantum solve, "
              "does most of the work at these piece sizes")


if __name__ == "__main__":
    main()
