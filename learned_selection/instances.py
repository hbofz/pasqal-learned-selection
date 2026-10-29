"""Problem instances. Every node carries weight=1, which the vendored code requires."""
import networkx as nx


def erdos_renyi(n: int, p: float, seed: int) -> nx.Graph:
    G = nx.gnp_random_graph(n, p, seed=seed)
    nx.set_node_attributes(G, 1, "weight")
    return G
