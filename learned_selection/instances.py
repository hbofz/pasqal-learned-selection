"""Problem instances. Every node carries weight=1, which the vendored code requires."""
import networkx as nx
import numpy as np


def _weighted(G: nx.Graph) -> nx.Graph:
    nx.set_node_attributes(G, 1, "weight")
    return G


def erdos_renyi(n: int, p: float, seed: int) -> nx.Graph:
    return _weighted(nx.gnp_random_graph(n, p, seed=seed))


def factor_market(n: int, theta: float, seed: int, market: float = 0.5, sector: float = 0.4,
                  sectors: int = 11, days: int = 250) -> nx.Graph:
    """Correlation graph of simulated returns: one market factor, one sector factor, and noise.

    Vertices i < j are joined when their correlation exceeds theta (positive correlation only).
    A larger `market` loading raises every correlation, which models a crisis.
    """
    rng = np.random.default_rng(seed)
    sector_of = rng.integers(sectors, size=n)
    b_market = market * rng.uniform(0.5, 1.5, n)
    b_sector = sector * rng.uniform(0.5, 1.5, n)
    R = (rng.standard_normal((days, 1)) * b_market
         + rng.standard_normal((days, sectors))[:, sector_of] * b_sector
         + rng.standard_normal((days, n)))
    C = np.corrcoef(R, rowvar=False)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(np.argwhere(np.triu(C > theta, k=1)).tolist())
    return _weighted(G)


def power_law(n: int, m: int, seed: int) -> nx.Graph:
    return _weighted(nx.barabasi_albert_graph(n, m, seed=seed))


def sbm(n: int, blocks: int, p_in: float, p_out: float, seed: int) -> nx.Graph:
    """Stochastic block model with near-equal blocks; each node keeps its `block` attribute."""
    sizes = [n // blocks + (b < n % blocks) for b in range(blocks)]
    probs = [[p_in if a == b else p_out for b in range(blocks)] for a in range(blocks)]
    B = nx.stochastic_block_model(sizes, probs, seed=seed)
    G = nx.Graph()  # networkx adds nodes in edge order, so rebuild with nodes 0..n-1 in order
    G.add_nodes_from((v, B.nodes[v]) for v in range(n))
    G.add_edges_from(B.edges())
    return _weighted(G)


def unit_disk(n: int, radius: float, seed: int) -> nx.Graph:
    """Random points in the unit square joined within `radius`; each node keeps its `pos` attribute."""
    return _weighted(nx.random_geometric_graph(n, radius, seed=seed))


FAMILIES = {"er": erdos_renyi, "factor": factor_market, "power_law": power_law, "sbm": sbm, "unit_disk": unit_disk}


def make(family: str, n: int, seed: int, **params) -> nx.Graph:
    return FAMILIES[family](n=n, seed=seed, **params)


TRAIN_SETTINGS = [("er", {"p": 0.1}), ("er", {"p": 0.25}), ("er", {"p": 0.75}),
                  ("power_law", {"m": 2}), ("power_law", {"m": 8}),
                  ("sbm", {"blocks": 4, "p_in": 0.3, "p_out": 0.03}),
                  ("factor", {"theta": 0.2}), ("factor", {"theta": 0.3})]
HELD_OUT_SETTINGS = [("unit_disk", {"radius": 0.1}), ("unit_disk", {"radius": 0.15}),
                     ("factor", {"theta": 0.3, "market": 1.0}), ("factor", {"theta": 0.4, "market": 1.0}),
                     ("factor", {"theta": 0.5, "market": 1.0})]
BIG_SETTINGS = [("er", {"p": 0.25}), ("power_law", {"m": 2})]

TRAIN_SIZES = (60, 100, 150)
TEST_SIZES = (100, 150)
SEL_SIZES = (100, 150)
BIG_N = 300
TRAIN_SEEDS = range(0, 20)
VAL_SEEDS = range(100, 105)
SEL_SEEDS = range(105, 125)
TEST_SEEDS = range(1000, 1020)
