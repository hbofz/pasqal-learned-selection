"""A learned piece builder: it adds one (vertex, trap) pair at a time, in place of the breadth-first growth.

A pair (u, t) is feasible when t is free and next to a filled trap and u's placed neighbours are exactly the vertices
on the traps next to t, so every piece it builds is an induced subgraph of the lattice. A GNN body embeds the
vertices once per state; a small MLP scores each feasible pair from the vertex's embedding and ten features of the
pair. The deployed hook takes the highest score at each step and computes no independent set."""
import os

import networkx as nx
import numpy as np
import torch
from torch import nn

from .engine import Builder
from .gnn import GNN, encode
from .growth import code_start_trap, lattice_adjacency
from .lattice import load_coords

PAIR_FEATURES = ("trap_x", "trap_y", "trap_centre", "trap_degree", "filled_nbrs", "free_nbrs", "size", "deg",
                 "unplaced_deg", "deg_share")


def trap_features(coords: dict, adj: list[list[int]]) -> np.ndarray:
    """(traps, 4): x and y centred and scaled by the largest absolute coordinate, their norm, and degree over 6."""
    xy = np.array([coords[t] for t in range(len(adj))], float)
    xy = (xy - xy.mean(0)) / np.abs(xy - xy.mean(0)).max()
    return np.c_[xy, np.hypot(*xy.T), [len(a) / 6 for a in adj]]


def pair_options(g: nx.Graph, place: dict, occ: dict, adj: list[list[int]], trap_x: np.ndarray) -> list[tuple]:
    """Every feasible (vertex, trap, features), ordered by trap then vertex. `place` maps vertex to trap, `occ` trap
    to vertex."""
    nbrs = g._adj
    out = []
    n, size = len(g), len(place)
    for t in sorted({x for s in occ for x in adj[s] if x not in occ}):
        near = {occ[x] for x in adj[t] if x in occ}
        cands = None
        for v in near:
            cands = set(nbrs[v]) if cands is None else cands & set(nbrs[v])
        free = len(adj[t]) - len(near)
        for u in sorted(cands or ()):
            if u not in place and nbrs[u].keys() & place.keys() == near:
                d = len(nbrs[u])
                out.append((u, t, (*trap_x[t], len(near) / 6, free / 6, size / 61, d / n, (d - len(near)) / n,
                                   d / max(1, n - size))))
    return out


class PieceBuilder(nn.Module):
    def __init__(self, body: GNN):
        super().__init__()
        self.body = body
        hidden = body.config["hidden"]
        self.pair = nn.Sequential(nn.Linear(hidden + len(PAIR_FEATURES), hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def embed(self, batch) -> torch.Tensor:
        """The body's vertex embeddings: its layers without the output head."""
        b = self.body
        h = b.embed((batch.x - b.mu) / b.sd)
        for conv, norm in zip(b.convs, b.norms):
            h = norm(h + torch.relu(conv(h, batch.edge_index)))
        return h

    def pair_logits(self, h: torch.Tensor, idx, phi) -> torch.Tensor:
        """One logit per pair: idx are the vertices' rows of h, phi their (pairs, 10) features."""
        return self.pair(torch.cat([h[torch.as_tensor(idx)], torch.as_tensor(phi)], 1)).squeeze(1)


def pair_weights(model: PieceBuilder) -> tuple:
    """The four numpy arrays of the pair scorer: first weight and bias, second weight and bias."""
    a, b = model.pair[0], model.pair[2]
    return tuple(x.detach().numpy().astype(np.float32) for x in (a.weight, a.bias, b.weight, b.bias))


def build_piece(g: nx.Graph, H: np.ndarray, weights: tuple, start, adj: list[list[int]], start_trap: int,
                trap_x: np.ndarray, rng: np.random.Generator | None = None) -> tuple[dict, list]:
    """One piece from `start` on `start_trap`. H holds the vertex embeddings in g order. Without rng it takes the
    highest logit, the first on a tie, and returns no decisions; with rng it samples (Gumbel noise on the logits)
    and returns the decisions as (rows of H, features, index chosen). Logits are numpy: nothing here is autograd."""
    at = {v: i for i, v in enumerate(g)}
    place, occ, decisions = {start: start_trap}, {start_trap: start}, []
    w1, b1, w2, b2 = weights
    while options := pair_options(g, place, occ, adj, trap_x):
        idx = np.fromiter((at[o[0]] for o in options), int, len(options))
        phi = np.array([o[2] for o in options], np.float32)
        logit = (np.maximum(np.c_[H[idx], phi] @ w1.T + b1, 0) @ w2.T + b2).ravel()
        if rng is None:
            chosen = int(np.argmax(logit))
        else:
            chosen = int(np.argmax(logit + rng.gumbel(size=len(logit))))
            decisions.append((idx, phi, chosen))
        u, t = options[chosen][:2]
        place[u], occ[t] = t, u
    return place, decisions


def builder_hook(model: PieceBuilder) -> Builder:
    """Build hook: one greedy piece from every vertex of g, in g order, from one embedding pass."""
    coords = load_coords()
    adj, _ = lattice_adjacency(coords)
    start_trap, trap_x, weights = code_start_trap(len(adj)), trap_features(coords, adj), pair_weights(model)

    def build(g: nx.Graph) -> list[dict]:
        if not len(g):
            return []
        with torch.inference_mode():
            H = model.embed(encode(g).to("cpu")).numpy()
        return [build_piece(g, H, weights, v, adj, start_trap, trap_x)[0] for v in g]
    return build


def save_builder(model: PieceBuilder, path) -> None:
    tmp = f"{path}.tmp"
    torch.save({"config": model.body.config, "state": model.state_dict()}, tmp)
    os.replace(tmp, path)


def load_builder(path) -> PieceBuilder:
    saved = torch.load(path, weights_only=True)
    model = PieceBuilder(GNN(**saved["config"]))
    model.load_state_dict(saved["state"])
    return model.eval()
