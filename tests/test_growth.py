import random, networkx as nx, pytest
from learned_selection.engine import gls_generator, vendored_gls_generator, _remove_closed
from learned_selection.growth import lattice_adjacency, code_start_trap
from learned_selection.instances import erdos_renyi
from learned_selection.lattice import load_coords
from learned_selection.vendored import Lattice

def _graphs():
    gs = [erdos_renyi(n, p, s) for n in (20, 60, 150) for p in (0.05, 0.25, 0.75) for s in (0, 1)]
    gs.append(_remove_closed(erdos_renyi(120, 0.2, 4), [0, 7, 19]))   # a mid-run state with label gaps
    for H in (nx.barabasi_albert_graph(100, 3, seed=0), nx.random_geometric_graph(100, 0.15, seed=0)):
        nx.set_node_attributes(H, 1, "weight")
        gs.append(H)
    return gs

@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("G", _graphs())
def test_port_matches_vendored_from_every_start(G, seed):
    # each mapping holds its start vertex on the seed trap, so equal sorted lists mean equal mappings per start;
    # insertion order is compared too, because it sets the piece's node order and so the solvers' tie-breaks
    items = lambda maps: [list(m.items()) for m in maps]
    assert items(gls_generator(seed=seed)(G, len(G))) == items(vendored_gls_generator(seed=seed)(G, len(G)))

def test_lattice_adjacency_matches_vendored():
    c = load_coords()
    L = nx.convert_node_labels_to_integers(Lattice(c, 6.6).lattice)
    adj, avg = lattice_adjacency(c)
    assert adj == [list(L.neighbors(t)) for t in L] and avg == 5 and code_start_trap(61) == 32

def test_gls_generator_leaves_global_rng_alone():
    G = erdos_renyi(60, 0.25, 0)
    random.seed(5); expected = random.random()
    random.seed(5); gls_generator(seed=3)(G, 4)
    assert random.random() == expected

def test_score_hook_changes_growth():
    G = erdos_renyi(80, 0.25, 0)
    by_degree = lambda g: (lambda grow, nodes: [grow.deg(v) for v in nodes])
    assert gls_generator(score=by_degree)(G, 4) != gls_generator()(G, 4)

def test_rank_hook_orders_candidates():
    G = erdos_renyi(80, 0.25, 0)
    smallest = gls_generator(rank=lambda g, maps: [-len(m) for m in maps])(G, 3)
    assert [len(m) for m in smallest] == sorted(len(m) for m in gls_generator()(G, len(G)))[:3]
