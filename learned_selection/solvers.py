"""Classical independent-set solvers: exact references, the exact piece stand-in, greedy baselines."""
import random

import networkx as nx
from ortools.sat.python import cp_model


def canonical(graph: nx.Graph) -> nx.Graph:
    """Copy with nodes and edges in repr-sorted order. Views of str-labelled graphs iterate Python sets, whose
    order changes with the per-process hash seed; anything order-dependent runs on this copy instead."""
    H = nx.Graph()
    H.add_nodes_from(sorted(graph.nodes(data=True), key=lambda nd: repr(nd[0])))
    H.add_edges_from(sorted((tuple(sorted(e, key=repr)) for e in graph.edges()), key=lambda e: (repr(e[0]), repr(e[1]))))
    return H


def _deterministic(graph: nx.Graph) -> nx.Graph:
    return graph if all(type(v) is int for v in graph) else canonical(graph)


def is_independent(graph: nx.Graph, nodes) -> bool:
    S = set(nodes)
    return not any(u in S for v in S for u in graph[v])


def _mis_model(graph: nx.Graph, maximal: bool):
    graph = _deterministic(graph)
    nodes = list(graph)
    idx = {v: i for i, v in enumerate(nodes)}
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(len(nodes))]
    for u, v in graph.edges():
        m.Add(x[idx[u]] + x[idx[v]] <= 1)
    if maximal:
        for v in nodes:
            m.Add(x[idx[v]] + sum(x[idx[u]] for u in graph[v]) >= 1)
    m.Maximize(sum(x))
    return m, x, nodes


def exact_mis(graph: nx.Graph, time_limit_s: float = 60.0) -> tuple[frozenset, bool]:
    """CP-SAT maximum independent set and whether it was proven optimal."""
    if len(graph) == 0:
        return frozenset(), True
    m, x, nodes = _mis_model(graph, maximal=False)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 8
    status = solver.Solve(m)
    S = frozenset(nodes[i] for i in range(len(nodes)) if solver.Value(x[i]))
    return S, status == cp_model.OPTIMAL


def top_maximal_sets(graph: nx.Graph, s: int) -> list[frozenset]:
    """Up to s distinct maximal independent sets, largest first. The exact stand-in for a quantum piece solve."""
    if len(graph) == 0:
        return [frozenset()]
    m, x, nodes = _mis_model(graph, maximal=True)
    out = []
    for _ in range(s):
        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 1   # single worker so ties resolve the same way every run
        solver.parameters.random_seed = 0
        status = solver.Solve(m)
        if status == cp_model.INFEASIBLE:
            break
        assert status == cp_model.OPTIMAL, f"piece solve not optimal: {solver.StatusName(status)}"
        chosen = [i for i in range(len(nodes)) if solver.Value(x[i])]
        out.append(frozenset(nodes[i] for i in chosen))
        others = [x[i] for i in range(len(nodes)) if i not in set(chosen)]
        m.Add(sum(others) - sum(x[i] for i in chosen) >= 1 - len(chosen))   # exclude exactly this set
    return out


def first_maximum_set(graph: nx.Graph) -> list:
    """Pasqal's small-piece rule: the first maximum independent set in include-first DFS order over graph.nodes()."""
    graph = _deterministic(graph)
    nodes = list(graph.nodes())
    best: list = []

    def rec(subset, index):
        nonlocal best
        if is_independent(graph, subset) and len(subset) > len(best):
            best = subset[:]
        for i in range(index, len(nodes)):
            subset.append(nodes[i])
            rec(subset, i + 1)
            subset.pop()

    rec([], 0)
    return best


def min_degree_greedy(graph: nx.Graph, seed: int) -> frozenset:
    rng = random.Random(seed)
    H = graph.copy()
    S = set()
    while len(H):
        d = min(dg for _, dg in H.degree())
        v = rng.choice(sorted((u for u, du in H.degree() if du == d), key=repr))
        S.add(v)
        H.remove_nodes_from([v, *H[v]])
    return frozenset(S)


def repair(graph: nx.Graph, nodes) -> frozenset:
    """Turn any vertex set into a maximal independent set: drop conflicts greedily, then fill greedily."""
    S = set(nodes)
    while True:
        conflicts = {v: sum(u in S for u in graph[v]) for v in S}
        worst = max(conflicts.values(), default=0)
        if worst == 0:
            break
        S.remove(min((v for v in S if conflicts[v] == worst), key=repr))
    for v in sorted(graph, key=lambda v: (graph.degree(v), repr(v))):
        if v not in S and not any(u in S for u in graph[v]):
            S.add(v)
    return frozenset(S)
