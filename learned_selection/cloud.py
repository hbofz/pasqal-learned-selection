"""Pasqal cloud emulators as a piece solver.

Emulation runs on Pulser's AnalogDevice: its one calibrated layout, TriangularLatticeLayout(61, 5.0 um), has
exactly the fresnel_61 trap ids and coordinates, and Pasqal's schedule (OMEGA, DELTA) sits inside its limits
(max_amp 12.566, max_abs_detuning 125.66 rad/us). Its max layout filling of 0.5 caps a piece at 30 atoms.
virtual_device() lifts that cap for pieces of 31 to 61 atoms: AnalogDevice's channels and limits with a max layout
filling of 1.0. It has no pre-calibrated layouts, so the register is placed on AnalogDevice's layout.

The real FRESNEL QPU (spec fetched 2026-10-29, data/fresnel_device.json) cannot run most pieces as they are:
its one calibrated layout is TriangularLatticeLayout(120, 5.0 um), which contains all 61 fresnel_61 traps at
identical coordinates, but min_layout_filling 0.35 means a register on it needs 42 to 60 atoms. Its channel
limits are also lower (max_amp 11.31, max_abs_detuning 62.83 rad/us), so piece_sequence clips the schedule.

Job results are bitstring -> count dicts, bit i belonging to the register's i-th qubit in insertion order, never by sorted qubit ids.
"""
import dataclasses, hashlib, json, random, re, subprocess, sys, threading, time, zlib
from collections import Counter
from functools import cache
from importlib import resources
from pathlib import Path

import networkx as nx
from pasqal_cloud.device import EmulatorType
from pasqal_cloud.device.device_types import DeviceTypeName
from pasqal_cloud.utils.polling import NON_TERMINAL_STATUSES
from pulser import Pulse, Sequence
from pulser.devices import AnalogDevice, Device, VirtualDevice
from pulser.waveforms import InterpolatedWaveform

from .engine import Mapping, PieceSolver
from .lattice import load_coords
from .solvers import is_independent, repair

PROJECT_ID = "6c44ee19-cbf0-4355-934f-4825f40cca40"
EMULATION_DEVICE = AnalogDevice
DURATION_NS = 4000
OMEGA = [1e-9, 4.850055051963065, 12.566370614359172, 11.882582923954192, 1e-9]           # Pasqal's schedule
DELTA = [-125.66370614359172, -23.735663070454464, 2.429518510655143, 13.330733571729787, 31.06390344878028]


def virtual_device(max_atoms: int = 61) -> VirtualDevice:
    """AnalogDevice with its layout filling cap lifted, so a piece can use up to max_atoms of the 61 traps."""
    return dataclasses.replace(AnalogDevice.to_virtual(), max_atom_num=max_atoms, max_layout_filling=1.0)


def client():
    from pasqal_cloud.pasqal_cloud_client import PasqalCloudClient

    def kc(*a):
        return subprocess.run(["security", "find-generic-password", "-s", "pasqal-cloud", *a],
                              capture_output=True, text=True, check=True).stdout
    account = re.search(r'"acct"<blob>="([^"]+)"', kc()).group(1)
    return PasqalCloudClient(username=account, password=kc("-w").strip(), project_id=PROJECT_ID)


@cache
def load_device() -> Device:
    """The FRESNEL QPU spec as fetched from Pasqal cloud."""
    return Device.from_abstract_repr(resources.files("learned_selection.data").joinpath("fresnel_device.json").read_text())


def _match(layout, coords) -> dict | None:
    traps = layout.traps_dict
    tm = {}
    for t, (x, y) in coords.items():
        hit = [i for i, p in traps.items() if abs(p[0] - x) < 1e-3 and abs(p[1] - y) < 1e-3]
        if len(hit) != 1:
            return None
        tm[t] = hit[0]
    return tm


def calibrated_layout(device: Device):
    """The device's pre-calibrated layout that holds every fresnel_61 trap at the same coordinates. A device without
    any (a VirtualDevice) uses AnalogDevice's."""
    coords = load_coords()
    for layout in getattr(device, "pre_calibrated_layouts", ()) or AnalogDevice.pre_calibrated_layouts:
        if _match(layout, coords):
            return layout
    raise ValueError("no pre-calibrated layout on this device contains the fresnel_61 lattice")


def trap_map(device: Device) -> dict[int, int]:
    """fresnel_61 trap id -> trap id on the device's calibrated layout."""
    return _match(calibrated_layout(device), load_coords())


def piece_sequence(piece: nx.Graph, mapping: Mapping, device: Device) -> Sequence:
    tm = trap_map(device)
    reg = calibrated_layout(device).define_register(*[tm[mapping[v]] for v in piece],
                                                    qubit_ids=[str(v) for v in piece])
    ch = device.channels["rydberg_global"]
    omega = [min(w, ch.max_amp) for w in OMEGA]
    delta = [max(min(d, ch.max_abs_detuning), -ch.max_abs_detuning) for d in DELTA]
    seq = Sequence(reg, device)
    seq.declare_channel("rydberg_global", "rydberg_global")
    seq.add(Pulse(InterpolatedWaveform(DURATION_NS, omega), InterpolatedWaveform(DURATION_NS, delta), 0),
            "rydberg_global")
    return seq


def decode(bitstring: str, qubit_ids: list[str]) -> list[str]:
    return [q for q, b in zip(qubit_ids, bitstring) if b == "1"]


def summarize_counts(piece: nx.Graph, counts: dict, qubit_ids: list[str], s: int):
    """(top-s repaired sets by frequency, raw stats). Raw stats are taken before repair, because repair alone
    can turn noise into an optimal set on small pieces."""
    label = {str(v): v for v in piece}
    if any(len(b) != len(qubit_ids) for b in counts):
        raise ValueError(f"bitstring length differs from the {len(qubit_ids)} qubits")
    merged, valid_sizes, shots, invalid = Counter(), Counter(), 0, 0
    for bits, n in counts.items():
        nodes = [label[q] for q in decode(bits, qubit_ids)]
        shots += n
        if is_independent(piece, nodes):
            valid_sizes[len(nodes)] += n
        else:
            invalid += n
        merged[repair(piece, nodes)] += n
    top = [S for S, _ in sorted(merged.items(), key=lambda kv: (-kv[1], sorted(map(repr, kv[0]))))[:s]]
    return top, {"shots": shots, "invalid_rate": invalid / shots if shots else 0.0, "valid_sizes": dict(valid_sizes)}


class CountsCache:
    """Append-only JSONL of answers by content key. One fn at a time per key; a failing fn writes nothing."""

    def __init__(self, path):
        self.path, self.rows, self._lock, self._keys, self._nl = Path(path), {}, threading.Lock(), {}, False
        if self.path.exists():
            text = self.path.read_text()
            self._nl = bool(text) and not text.endswith("\n")       # a torn last line: start the next on its own line
            for line in filter(str.strip, text.splitlines()):
                try:
                    row = json.loads(line)
                    self.rows[row.pop("key")] = row
                except (ValueError, KeyError, AttributeError):
                    print(f"{self.path}: skipping torn line {line[:60]!r}", file=sys.stderr)

    def __len__(self):
        return len(self.rows)

    def get_or_run(self, key: str, fn) -> dict:
        with self._lock:
            if key in self.rows:
                return self.rows[key]
            klock = self._keys.setdefault(key, threading.Lock())
        with klock:
            with self._lock:
                if key in self.rows:
                    return self.rows[key]
            row = json.loads(json.dumps(fn(), default=str))           # what a reload would give back
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(("\n" if self._nl else "") + json.dumps({"key": key, **row}) + "\n")
                    f.flush()
                self._nl, self.rows[key] = False, row
            return row


class CloudPieceSolver:
    """PieceSolver that runs each piece on a Pasqal cloud emulator. With a cache every answer is saved by content key
    (cloud_client=None then replays, a miss raises KeyError); without one, one log entry per call."""

    def __init__(self, cloud_client, device: Device = EMULATION_DEVICE, emulator: EmulatorType = EmulatorType.EMU_MPS, runs: int = 500,
                 cache: CountsCache | None = None, retries: int = 3, tags: list[str] | None = None, poll_s: float = 15.0,
                 deadline_s: float = 14400.0):
        self.client, self.device, self.emulator, self.runs = cloud_client, device, emulator, runs
        self.cache, self.retries, self.tags, self.poll_s, self.deadline_s = cache, retries, tags, poll_s, deadline_s
        self.log: list[dict] = []

    def __call__(self, piece: nx.Graph, mapping: Mapping, s: int) -> list[frozenset]:
        return self.answer(piece, mapping, s)[0]

    def answer(self, piece: nx.Graph, mapping: Mapping, s: int):
        """(top-s sets, content key)."""
        seq = piece_sequence(piece, mapping, self.device)
        key = hashlib.sha256(f"{self.emulator.value}|{self.runs}|{seq.to_abstract_repr()}".encode()).hexdigest()
        ids = list(seq.register.qubit_ids)
        run = lambda: self._row(piece, seq, ids, key)
        row = run() if self.cache is None else self.cache.get_or_run(key, run)
        top, stats = summarize_counts(piece, row["counts"], ids, s)
        if self.cache is None:
            self.log.append({"n_atoms": len(piece), "batch_id": row["batch_id"], "key": key, **stats, "top_sizes": [len(S) for S in top]})
        return top, key

    def _row(self, piece, seq, ids, key) -> dict:
        if self.client is None:
            raise KeyError(key)
        for attempt in range(1, self.retries + 2):
            try:
                batch = self._attempt(seq)
                counts = batch.ordered_jobs[0].result
                if counts is None or batch.status != "DONE":
                    raise RuntimeError(f"batch {batch.id} ended {batch.status} without counts")
                if sum(counts.values()) != self.runs:
                    raise ValueError(f"batch {batch.id} has {sum(counts.values())} shots, wanted {self.runs}")
                summarize_counts(piece, counts, ids, 1)                # raises on a wrong-length bitstring
                return {"atoms": len(piece), "ids": ids, "edges": [[u, v] for u, v in piece.edges()], "counts": dict(counts),
                        "batch_id": batch.id, "runs": self.runs, "emulator": self.emulator.value, "created_at": batch.created_at,
                        "start_datetime": batch.start_datetime, "end_datetime": batch.end_datetime}
            except Exception as e:
                if attempt > self.retries:
                    raise
                print(f"attempt {attempt} failed ({type(e).__name__}: {e}), retrying", file=sys.stderr)
                time.sleep(60 * attempt)

    def _attempt(self, seq):
        batch = self.client.create_batch(seq.to_abstract_repr(), jobs=[{"runs": self.runs}], device_type=DeviceTypeName(str(self.emulator)),
                                         wait=False, tags=self.tags)
        try:
            t0 = time.monotonic()
            while batch.status in NON_TERMINAL_STATUSES:
                if time.monotonic() - t0 > self.deadline_s:
                    try:
                        self.client.cancel_batch(batch.id)
                    except Exception:
                        pass
                    raise TimeoutError(f"batch {batch.id} still {batch.status} after {self.deadline_s:g} s")
                time.sleep(self.poll_s)
                try:
                    batch.refresh()
                except Exception:
                    pass
            return batch
        finally:
            self.client.batches.pop(batch.id, None)


def uniform_piece_solver(runs: int = 500) -> PieceSolver:
    """Control: runs uniform random bitstrings through the same decode and repair. Deterministic, needs no client."""
    def solve(piece: nx.Graph, mapping: Mapping, s: int) -> list[frozenset]:
        ids = [str(v) for v in piece]
        seed = zlib.crc32(json.dumps([ids, [[u, v] for u, v in piece.edges()], [mapping[v] for v in piece]]).encode())
        rng = random.Random(seed)
        counts = Counter("".join(rng.choice("01") for _ in piece) for _ in range(runs))
        return summarize_counts(piece, counts, ids, s)[0]
    return solve
