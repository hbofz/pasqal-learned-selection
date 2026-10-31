import json, resource

import networkx as nx, numpy as np, pytest, torch
from learned_selection import pg
from learned_selection.dataset import _setting_name
from learned_selection.engine import exact_piece_solver
from learned_selection.features import PIECE_FEATURES, VERTEX_FEATURES
from learned_selection.gnn import GNN, encode, load_gnn, save_gnn
from learned_selection.instances import HELD_OUT_SETTINGS, TRAIN_SEEDS, erdos_renyi
from learned_selection.pg import MAX_DECISIONS, advantages, pg_update, rollout_job, top_k_logp, training_instances


def _toy(head, seed):
    torch.manual_seed(seed)
    d = len(VERTEX_FEATURES)
    return GNN("gin", head, np.zeros(d), np.ones(d), np.zeros(len(PIECE_FEATURES)), np.ones(len(PIECE_FEATURES)),
               hidden=8).eval()


def _toy_models(path):
    path.mkdir(exist_ok=True)
    for name, head in (("dgin-h2", "vertex"), ("dgin-h4", "piece"), ("dgin-h5", "value")):
        save_gnn(_toy(head, 0), path / f"{name}.pt")
    return path


def _job(models, blob, c=0.01, rseed=3, seed=2, n=40, family="er", params={"p": 0.1}):
    return (blob, "dgin-h2+dgin-h5", models, family, params, n, seed, rseed, c)


def test_training_instances_stay_on_training_seeds_and_settings():
    inst = training_instances()
    assert len(inst) == 8 * 2 * 20 and {i[3] for i in inst} == set(TRAIN_SEEDS) and {i[2] for i in inst} == {60, 100}
    held = {_setting_name(f, p) for f, p in HELD_OUT_SETTINGS}
    assert not {_setting_name(i[0], i[1]) for i in inst} & held


def test_top_k_logp_matches_gumbel_sampling():
    s = torch.tensor([0.5, 1.5, 0.0, -1.0])
    rng = np.random.default_rng(0)
    draws = np.argsort(-(s.numpy() + rng.gumbel(size=(200_000, 4))), axis=1)[:, :2]
    freq = np.mean((draws == [1, 0]).all(axis=1))
    assert float(top_k_logp(s, np.array([1, 0])).exp()) == pytest.approx(freq, abs=0.005)


def test_rollout_top_k_equals_the_order_it_records(tmp_path, monkeypatch):
    models = _toy_models(tmp_path)
    solved = []

    def run(G, k, s, l, **kw):
        def solve(piece, m, width):
            solved.append(frozenset(m))
            return exact_piece_solver(piece, m, width)
        return real(G, k, s, l, solve=solve, **kw)
    real = pg.run_code_bounded
    monkeypatch.setattr(pg, "run_code_bounded", run)
    out = rollout_job(_job(models, pg._blob(_toy("piece", 1))))
    assert out["calls"] == len(out["decisions"]) <= MAX_DECISIONS    # a small graph: every decision is kept
    b, order = out["decisions"][0]
    nodes = list(erdos_renyi(40, 0.1, 2))
    piece = lambda j: frozenset(nodes[i] for i in b.piece_nodes[b.piece_of == j])
    assert len(order) == pg.PG_K and set(solved[:pg.PG_K]) == {piece(j) for j in order}


def test_rollout_keeps_at_most_max_decisions_and_is_repeatable(tmp_path):
    models = _toy_models(tmp_path)
    blob = pg._blob(_toy("piece", 1))
    a, b = (rollout_job(_job(models, blob, n=100, family="power_law", params={"m": 2})) for _ in range(2))
    assert a["calls"] > MAX_DECISIONS and len(a["decisions"]) == MAX_DECISIONS
    assert a["reward"] == b["reward"] and all((x[1] == y[1]).all() for x, y in zip(a["decisions"], b["decisions"]))
    assert a["key"] == ("power_law", '{"m": 2}', 100, 2) and a["reward"] == pytest.approx(a["size"] - 0.01 * a["runs"])


def test_advantages_are_centred_per_instance():
    res = [dict(key="a", reward=1.0), dict(key="a", reward=3.0), dict(key="b", reward=10.0), dict(key="b", reward=10.0)]
    adv = advantages(res)
    assert adv[0] < 0 < adv[1] and adv[2] == adv[3] == 0


def _state():
    g = erdos_renyi(24, 0.2, 0)
    return encode(g, [{0, 1, 2}, {3, 4, 5}, {6, 7, 8}])


def test_pg_update_raises_the_score_of_a_rewarded_piece():
    net, b = _toy("piece", 0), _state()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    scores = lambda: net(b.to("cpu")).detach()
    gap = lambda s: float(s[0] - s[1])
    before = gap(scores())
    res = [dict(key="a", reward=3.0, decisions=[(b, np.array([0]))], calls=1),
           dict(key="a", reward=1.0, decisions=[(b, np.array([1]))], calls=1)]
    assert pg_update(net, opt, res)["decisions"] == 2
    assert gap(scores()) > before


def test_pg_update_survives_no_decisions_and_equal_rewards():
    net, b = _toy("piece", 0), _state()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    for res in ([dict(key="a", reward=3.0, decisions=[], calls=0), dict(key="a", reward=1.0, decisions=[], calls=0)],
                [dict(key="a", reward=2.0, decisions=[(b, np.array([0]))], calls=1)] * 2):
        assert pg_update(net, opt, res)["decisions"] == 0
        assert all(torch.equal(v, before[k]) and v.isfinite().all() for k, v in net.state_dict().items())


def test_pg_resume_matches_an_unbroken_run(tmp_path):
    models = _toy_models(tmp_path / "models")
    run = lambda out, iters: pg.main(["--init", str(models / "dgin-h4.pt"), "--stack", "dgin-h2+dgin-h5", "--models",
                                      str(models), "--iters", str(iters), "--every", "2", "--instances", "2",
                                      "--rollouts", "2", "--workers", "1", "--threads", "1", "--lr", "1e-2", "--out", str(out)])
    run(tmp_path / "split", 2)
    assert (tmp_path / "split" / "it2" / "pg-h4.pt").exists() and (tmp_path / "split" / "it2" / "dgin-h5.pt").exists()
    run(tmp_path / "split", 4)
    run(tmp_path / "whole", 4)
    a, b = (torch.load(tmp_path / d / "state.pt", weights_only=True) for d in ("split", "whole"))
    assert a["t"] == b["t"] == 4 and a["net"].keys() == b["net"].keys()
    assert all(torch.allclose(a["net"][k], b["net"][k], atol=1e-6) for k in a["net"])
    moved = load_gnn(tmp_path / "whole" / "it4" / "pg-h4.pt").state_dict()
    start = load_gnn(models / "dgin-h4.pt").state_dict()
    assert any(not torch.equal(moved[k], start[k]) for k in moved)
    rows = [json.loads(line) for line in (tmp_path / "split" / "log.jsonl").read_text().splitlines()]
    assert [r["t"] for r in rows] == [1, 2, 3, 4]
    meta = json.loads((tmp_path / "split" / "train_meta_pg-h4.json").read_text())
    assert meta["c"] == 0.01 and meta["cpu_s"] > 0 and meta["wall_s"] > 0


def test_a_checkpoint_counts_the_cpu_of_the_workers(tmp_path, monkeypatch):
    models = _toy_models(tmp_path / "models")
    children = lambda: (lambda r: r.ru_utime + r.ru_stime)(resource.getrusage(resource.RUSAGE_CHILDREN))
    before, seen, atomic = children(), [], pg._atomic

    def spy(write, path):
        atomic(write, path)
        if path.name == "state.pt":
            seen.append((torch.load(path, weights_only=True)["cpu"], children() - before))
    monkeypatch.setattr(pg, "_atomic", spy)
    pg.main(["--init", str(models / "dgin-h4.pt"), "--models", str(models), "--iters", "2", "--every", "1",
             "--instances", "2", "--rollouts", "1", "--workers", "2", "--threads", "1", "--out", str(tmp_path / "o")])
    cpu, kids = seen[0]   # the checkpoint after iteration 1, taken mid-run
    assert kids > 0 and cpu >= kids
    assert seen[1][0] >= cpu


def test_higher_run_cost_gives_a_lower_reward(tmp_path):
    models = _toy_models(tmp_path)
    blob = pg._blob(_toy("piece", 1))
    free, costly = rollout_job(_job(models, blob, c=0.0)), rollout_job(_job(models, blob, c=1.0))
    assert (free["size"], free["runs"]) == (costly["size"], costly["runs"]) and costly["runs"] > 0
    assert free["reward"] - costly["reward"] == pytest.approx(costly["runs"])


def test_rollouts_refuse_seeds_outside_training():
    for seed in (100, 105, 200, 1000):
        with pytest.raises(ValueError):
            rollout_job((b"", "dgin-h2+dgin-h5", "models", "er", {"p": 0.1}, 40, seed, 0, 0.01))
