"""Graph neural networks over the structural features of a state graph, in plain PyTorch (no torch_geometric).

Three message-passing layers (GIN, GraphSAGE, GAT) feed one of three heads: a score per vertex, a score per
candidate piece, or a value per graph. The hooks at the bottom plug a trained network into the beam engine.
Graph structure only: nothing here computes an independent set, so a model never solves anything itself.

A Batch holds numpy arrays, so it pickles cheaply between processes; Batch.to turns it into torch tensors."""
import dataclasses
from dataclasses import dataclass
from itertools import chain

import networkx as nx
import numpy as np
import torch
from torch import nn

from .engine import Rank, Value
from .features import PIECE_FEATURES, VERTEX_FEATURES, piece_features, vertex_features
from .growth import VertexScore

CONVS = ("gin", "sage", "gat")
HEADS = ("vertex", "piece", "value")
_I64 = np.int64
_FIELDS = ("x", "edge_index", "graph", "piece_nodes", "piece_of", "closed_nodes", "closed_of", "piece_graph",
           "piece_x")


@dataclass
class Batch:
    x: np.ndarray             # float32 (N, 9): raw vertex features
    edge_index: np.ndarray    # int64 (2, 2E): both directions
    graph: np.ndarray         # int64 (N,): the graph id of each node
    n_graphs: int
    piece_nodes: np.ndarray   # int64 (sum P,): the members of each candidate
    piece_of: np.ndarray      # int64 (sum P,): the candidate each member belongs to
    closed_nodes: np.ndarray  # int64: the members of each closed neighbourhood N[P]
    closed_of: np.ndarray     # int64: the candidate each of those belongs to
    piece_graph: np.ndarray   # int64 (n_pieces,): the graph id of each candidate
    piece_x: np.ndarray       # float32 (n_pieces, 13): raw candidate features

    def to(self, device) -> "Batch":
        """A copy whose arrays are torch tensors on the device."""
        return dataclasses.replace(self, **{f: torch.from_numpy(getattr(self, f)).to(device) for f in _FIELDS})


def encode(g: nx.Graph, maps: list | None = None) -> Batch:
    """One graph as a Batch. Node i is list(g)[i]. Each map is a dict or set of g's nodes."""
    maps = maps or []
    nodes = list(g)
    idx = {v: i for i, v in enumerate(nodes)}
    adj = [[idx[u] for u in g._adj[v]] for v in nodes]
    n = len(nodes)
    x = np.zeros((n, len(VERTEX_FEATURES)), dtype=np.float32)
    if n:
        x[:] = vertex_features(g)[1]
    edge_index = np.empty((2, sum(map(len, adj))), dtype=_I64)
    edge_index[0] = np.repeat(np.arange(n, dtype=_I64), [len(a) for a in adj])
    edge_index[1] = np.fromiter(chain.from_iterable(adj), dtype=_I64, count=edge_index.shape[1])
    pn, po, cn, co = [], [], [], []
    if maps:
        sets = [set(a) for a in adj]
        for j, m in enumerate(maps):
            P = sorted(idx[v] for v in m)
            closed = set(P).union(*(sets[i] for i in P))
            pn.append(P), po.append([j] * len(P))
            cn.append(sorted(closed)), co.append([j] * len(closed))
    flat = lambda xs: np.fromiter(chain.from_iterable(xs), dtype=_I64)
    px = piece_features(g, maps).astype(np.float32) if maps else np.zeros((0, len(PIECE_FEATURES)), np.float32)
    return Batch(x, edge_index, np.zeros(n, dtype=_I64), 1, flat(pn), flat(po), flat(cn), flat(co),
                 np.zeros(len(maps), dtype=_I64), px)


def collate(batches: list[Batch]) -> Batch:
    """Concatenate batches, offsetting the node, graph and piece ids."""
    node_off = np.cumsum([0] + [len(b.x) for b in batches[:-1]])
    graph_off = np.cumsum([0] + [b.n_graphs for b in batches[:-1]])
    piece_off = np.cumsum([0] + [len(b.piece_x) for b in batches[:-1]])
    cat = lambda f, offs: np.concatenate([getattr(b, f) + o for b, o in zip(batches, offs)])
    return Batch(np.concatenate([b.x for b in batches]),
                 np.concatenate([b.edge_index + o for b, o in zip(batches, node_off)], axis=1),
                 cat("graph", graph_off), sum(b.n_graphs for b in batches),
                 cat("piece_nodes", node_off), cat("piece_of", piece_off),
                 cat("closed_nodes", node_off), cat("closed_of", piece_off),
                 cat("piece_graph", graph_off), np.concatenate([b.piece_x for b in batches]))


def _mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


def _sum_into(values, index, n):
    """Row sums of values grouped by index: an (n, d) tensor."""
    return values.new_zeros((n,) + values.shape[1:]).index_add_(0, index, values)


class _GIN(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.mlp = _mlp(d, d, d)

    def forward(self, h, edge_index):
        return self.mlp(h + _sum_into(h[edge_index[0]], edge_index[1], len(h)))


class _Sage(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.self_lin, self.nbr_lin = nn.Linear(d, d), nn.Linear(d, d, bias=False)

    def forward(self, h, edge_index):
        total = _sum_into(h[edge_index[0]], edge_index[1], len(h))
        count = torch.bincount(edge_index[1], minlength=len(h)).clamp(min=1).unsqueeze(1)
        return self.self_lin(h) + self.nbr_lin(total / count)


class _GAT(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0, "hidden must be a multiple of heads"
        self.heads, self.width = heads, d // heads
        self.lin = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.width) / self.width ** 0.5)
        self.a_dst = nn.Parameter(torch.randn(heads, self.width) / self.width ** 0.5)

    def forward(self, h, edge_index):
        n = len(h)
        loops = torch.arange(n, device=h.device)
        src, dst = torch.cat([edge_index[0], loops]), torch.cat([edge_index[1], loops])
        wh = self.lin(h).view(n, self.heads, self.width)
        logit = nn.functional.leaky_relu((wh * self.a_src).sum(-1)[src] + (wh * self.a_dst).sum(-1)[dst], 0.2)
        top = logit.new_full((n, self.heads), -torch.inf).scatter_reduce(
            0, dst.unsqueeze(1).expand_as(logit), logit, "amax", include_self=True)
        w = (logit - top[dst]).exp()
        alpha = w / _sum_into(w, dst, n)[dst]
        return _sum_into(alpha.unsqueeze(-1) * wh[src], dst, n).reshape(n, self.heads * self.width)


_CONV = {"gin": _GIN, "sage": _Sage, "gat": _GAT}


class GNN(nn.Module):
    def __init__(self, conv, head, mu, sd, piece_mu=None, piece_sd=None, hidden=64, layers=3, heads=4):
        super().__init__()
        if conv not in CONVS or head not in HEADS:
            raise ValueError(f"conv must be one of {CONVS} and head one of {HEADS}, got {conv!r} and {head!r}")
        d_piece = len(PIECE_FEATURES)
        piece_mu = np.zeros(d_piece) if piece_mu is None else piece_mu
        piece_sd = np.ones(d_piece) if piece_sd is None else piece_sd
        self.config = dict(conv=conv, head=head, mu=np.asarray(mu).tolist(), sd=np.asarray(sd).tolist(),
                           piece_mu=np.asarray(piece_mu).tolist(), piece_sd=np.asarray(piece_sd).tolist(),
                           hidden=hidden, layers=layers, heads=heads)
        self.head = head
        for name, a in (("mu", mu), ("sd", sd), ("piece_mu", piece_mu), ("piece_sd", piece_sd)):
            a = torch.as_tensor(np.asarray(a), dtype=torch.float32)
            self.register_buffer(name, torch.where(a == 0, torch.ones_like(a), a) if name.endswith("sd") else a)
        self.embed = nn.Linear(len(VERTEX_FEATURES), hidden)
        self.convs = nn.ModuleList(_CONV[conv](hidden, heads) for _ in range(layers))
        self.out = _mlp({"piece": 4 * hidden + d_piece}.get(head, hidden), hidden, 1)

    def forward(self, b: Batch) -> torch.Tensor:
        h = self.embed((b.x - self.mu) / self.sd)
        for conv in self.convs:
            h = h + torch.relu(conv(h, b.edge_index))
        if self.head == "vertex":
            return self.out(h).squeeze(1)
        if self.head == "value":
            return _sum_into(self.out(h).squeeze(1), b.graph, b.n_graphs)
        n_pieces = len(b.piece_x)
        pv, po, cv, co = h[b.piece_nodes], b.piece_of, h[b.closed_nodes], b.closed_of
        p_count = torch.bincount(po, minlength=n_pieces).clamp(min=1).unsqueeze(1)
        c_count = torch.bincount(co, minlength=n_pieces)
        p_max = h.new_zeros((n_pieces, h.shape[1])).scatter_reduce(
            0, po.unsqueeze(1).expand_as(pv), pv, "amax", include_self=False)
        c_sum = _sum_into(cv, co, n_pieces)
        g_sum = _sum_into(h, b.graph, b.n_graphs)[b.piece_graph]
        g_count = torch.bincount(b.graph, minlength=b.n_graphs)[b.piece_graph]
        rest = (g_sum - c_sum) / (g_count - c_count).clamp(min=1).unsqueeze(1)
        feats = torch.cat([_sum_into(pv, po, n_pieces) / p_count, p_max, c_sum / c_count.clamp(min=1).unsqueeze(1),
                           rest, (b.piece_x - self.piece_mu) / self.piece_sd], dim=1)
        return self.out(feats).squeeze(1)


def _predict(model: GNN, batch: Batch) -> np.ndarray:
    with torch.inference_mode():
        return model(batch.to("cpu")).double().numpy()


def vertex_scores(model: GNN, g: nx.Graph) -> np.ndarray:
    """One score per vertex, in g order."""
    return _predict(model, encode(g))


def piece_scores(model: GNN, g: nx.Graph, maps: list) -> np.ndarray:
    """One score per candidate piece."""
    return _predict(model, encode(g, maps)) if maps else np.zeros(0)


def graph_values(model: GNN, graphs: list) -> np.ndarray:
    """One value per graph, from a single forward pass."""
    return _predict(model, collate([encode(g) for g in graphs])) if graphs else np.zeros(0)


def gnn_score(model: GNN) -> VertexScore:
    """Growth hook: the model's score of each vertex in the state graph is its priority."""
    def for_state(g):
        priority = dict(zip(g, vertex_scores(model, g).tolist()))
        return lambda grow, cands: [priority[v] for v in cands]
    return for_state


def gnn_rank(model: GNN) -> Rank:
    """Candidate-choice hook: the model's score of each candidate piece is its rank."""
    return lambda g, maps: piece_scores(model, g, maps)


def gnn_value(model: GNN) -> Value:
    """Pruning hook: the model's value of each remaining graph."""
    return lambda graphs: graph_values(model, graphs)


def save_gnn(model: GNN, path) -> None:
    torch.save({"config": model.config, "state": model.state_dict()}, path)


def load_gnn(path) -> GNN:
    saved = torch.load(path, weights_only=True)
    model = GNN(**saved["config"])
    model.load_state_dict(saved["state"])
    return model.eval()
