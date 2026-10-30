import networkx as nx, pytest
from pulser.exceptions.sequence import MinQubitNumberError
from learned_selection.cloud import EMULATION_DEVICE, decode, load_device, calibrated_layout, trap_map, piece_sequence, summarize_counts
from learned_selection.engine import gls_generator
from learned_selection.instances import erdos_renyi
from learned_selection.lattice import load_coords

def test_decode_uses_insertion_order():
    ids = [str(i) for i in range(12)]            # sorted(ids) would put "10", "11" before "2"
    assert decode("001000000001", ids) == ["2", "11"]

@pytest.mark.parametrize("dev", [EMULATION_DEVICE, load_device()], ids=["AnalogDevice", "FRESNEL"])
def test_trap_map_places_every_trap_on_the_calibrated_layout(dev):
    c = load_coords()
    layout, tm = calibrated_layout(dev), trap_map(dev)
    assert sorted(tm) == list(range(61)) and len(set(tm.values())) == 61
    traps = layout.traps_dict
    assert all(abs(traps[tm[t]][0] - c[t][0]) < 1e-3 and abs(traps[tm[t]][1] - c[t][1]) < 1e-3 for t in c)

def test_fresnel_rejects_pieces_under_42_atoms():
    G = erdos_renyi(40, 0.25, 0)
    m = gls_generator()(G, 1)[0]
    assert len(m) < 42
    with pytest.raises(MinQubitNumberError):
        piece_sequence(G.subgraph(m), m, load_device())

def test_piece_sequence_matches_mapping_geometry():
    G = erdos_renyi(40, 0.25, 0)
    m = gls_generator()(G, 1)[0]
    piece, dev, c = G.subgraph(m), EMULATION_DEVICE, load_coords()
    seq = piece_sequence(piece, m, dev)          # building it runs Pulser's device validation
    pos = seq.register.qubits
    assert list(pos) == [str(v) for v in piece]
    for v in piece:
        assert abs(float(pos[str(v)][0]) - c[m[v]][0]) < 1e-3 and abs(float(pos[str(v)][1]) - c[m[v]][1]) < 1e-3

def test_summarize_counts_separates_raw_quality_from_repair():
    piece = nx.path_graph(6)
    ids = [str(v) for v in piece]
    top_g, st_g = summarize_counts(piece, {"101010": 500}, ids, 2)
    top_n, st_n = summarize_counts(piece, {"111111": 250, "110011": 250}, ids, 2)
    assert top_g == [frozenset({0, 2, 4})] and st_g["invalid_rate"] == 0 and st_g["valid_sizes"] == {3: 500}
    assert st_n["invalid_rate"] == 1 and st_n["valid_sizes"] == {}
    assert all(len(S) == 3 for S in top_n)       # repair alone reaches the optimum: decisions must use raw stats

def test_virtual_device_takes_large_pieces():
    from learned_selection.cloud import virtual_device
    from learned_selection.instances import power_law
    G = power_law(150, 8, 0)
    m = next(m for m in gls_generator()(G, len(G)) if len(m) > 30)
    seq = piece_sequence(G.subgraph(m), m, virtual_device())
    assert len(seq.register.qubits) == len(m) and seq.to_abstract_repr()

def test_spread_keeps_evenly_spaced_items():
    from learned_selection.calibrate import spread
    xs = list(range(40))
    picked = spread(xs, 12)
    assert len(picked) == len(set(picked)) == 12 and picked[0] == 0 and picked[-1] == 39 and picked == sorted(picked)
    assert spread(xs, len(xs)) == xs
    assert spread(xs, 1) == [xs[0]]
