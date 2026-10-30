"""Pasqal cloud emulators as a piece solver.

Emulation runs on Pulser's AnalogDevice: its one calibrated layout, TriangularLatticeLayout(61, 5.0 um), has
exactly the fresnel_61 trap ids and coordinates, and Pasqal's schedule (OMEGA, DELTA) sits inside its limits
(max_amp 12.566, max_abs_detuning 125.66 rad/us). Its max layout filling of 0.5 caps a piece at 30 atoms.

The real FRESNEL QPU (spec fetched 2026-10-29, data/fresnel_device.json) cannot run most pieces as they are:
its one calibrated layout is TriangularLatticeLayout(120, 5.0 um), which contains all 61 fresnel_61 traps at
identical coordinates, but min_layout_filling 0.35 means a register on it needs 42 to 60 atoms. Its channel
limits are also lower (max_amp 11.31, max_abs_detuning 62.83 rad/us), so piece_sequence clips the schedule.

Job results are bitstring -> count dicts, bit i belonging to the register's i-th qubit in insertion order, never by sorted qubit ids.
"""
import re, subprocess
from collections import Counter
from functools import cache
from importlib import resources

import networkx as nx
from pasqal_cloud.device import EmulatorType
from pulser import Pulse, Sequence
from pulser.devices import AnalogDevice, Device
from pulser.waveforms import InterpolatedWaveform

from .engine import Mapping
from .lattice import load_coords
from .solvers import is_independent, repair

PROJECT_ID = "6c44ee19-cbf0-4355-934f-4825f40cca40"
EMULATION_DEVICE = AnalogDevice
DURATION_NS = 4000
OMEGA = [1e-9, 4.850055051963065, 12.566370614359172, 11.882582923954192, 1e-9]           # Pasqal's schedule
DELTA = [-125.66370614359172, -23.735663070454464, 2.429518510655143, 13.330733571729787, 31.06390344878028]


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
    """The device's pre-calibrated layout that holds every fresnel_61 trap at the same coordinates."""
    coords = load_coords()
    for layout in device.pre_calibrated_layouts:
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


class CloudPieceSolver:
    """PieceSolver that runs each piece on a Pasqal cloud emulator. One log entry per call."""

    def __init__(self, cloud_client, device: Device = EMULATION_DEVICE, emulator: EmulatorType = EmulatorType.EMU_MPS, runs: int = 500):
        self.client, self.device, self.emulator, self.runs = cloud_client, device, emulator, runs
        self.log: list[dict] = []

    def __call__(self, piece: nx.Graph, mapping: Mapping, s: int) -> list[frozenset]:
        seq = piece_sequence(piece, mapping, self.device)
        batch = self.client.create_batch(seq.to_abstract_repr(), jobs=[{"runs": self.runs}],
                                         emulator=self.emulator, wait=True)
        counts = batch.ordered_jobs[0].result
        top, stats = summarize_counts(piece, counts, list(seq.register.qubit_ids), s)
        self.log.append({"n_atoms": len(piece), "batch_id": batch.id, **stats,
                         "top_sizes": [len(S) for S in top]})
        return top
