import json, numpy as np, pytest
from learned_selection.bench import BEAM_METHODS, gap_pct, run_instance
from learned_selection.curves import bootstrap_ci, curve_gain, load, paired
from learned_selection.features import PIECE_FEATURES, VERTEX_FEATURES
from learned_selection.models import fit_lgbm, fit_linear, save

KEYS = {"family", "params", "n", "p", "seed", "method", "track", "k", "s", "l", "restarts", "lattice",
        "piece_solver", "includes_classical_time", "small", "blockade", "max_branches", "time_limit_s", "commit",
        "size", "best_known", "proven", "gap_pct", "quantum_runs", "pasqal_runs", "atoms", "rounds",
        "classical_s", "solver_s", "exploded", "max_atoms", "over_30", "atoms_all"}

def test_records_complete_and_consistent():
    recs = run_instance("er", {"p": 0.25}, 25, 0, ["gls-code", "gls", "gls-screen", "greedy-pasqal", "greedy-mindeg",
                        "exact"], widths=[(4, 2, 10)], restarts=[1, 2], time_limit_s=10)
    for r in recs:
        assert KEYS <= r.keys() and 0 <= r["gap_pct"] <= 100 and r["size"] <= r["best_known"]
    assert {r["track"] for r in recs if r["method"] == "gls-screen"} == {"hybrid"}
    assert sorted(r["restarts"] for r in recs if r["method"] == "gls") == [1, 2]
    assert gap_pct(9, 10) == 10.0 and gap_pct(0, 0) == 0.0

def test_exploded_baseline_is_recorded():
    recs = run_instance("er", {"p": 0.0}, 40, 0, ["gls-code", "gls"], widths=[(4, 2, 10)], restarts=[1],
                        time_limit_s=10, max_branches=50)
    code, bounded = (next(r for r in recs if r["method"] == m) for m in ("gls-code", "gls"))
    assert code["exploded"] and code["size"] is None and code["gap_pct"] is None
    assert not bounded["exploded"] and bounded["size"] == 40

def test_bootstrap_and_paired_diff():
    assert bootstrap_ci([2.0] * 10) == (2.0, 2.0)
    rows = [dict(family="er", params={"p": 0.25}, n=40, seed=i, method=m, k=4, s=2, l=10, restarts=1, gap_pct=g,
                 quantum_runs=10, exploded=False) for i in range(5) for m, g in (("gls", 10.0), ("lin-h4", 7.0))]
    (row,) = paired(rows, "lin-h4", "gls", seeds=range(5))
    assert row["diff"] == row["lo"] == row["hi"] == -3.0

def test_curve_gain_by_hand():
    # GLS envelope: (10 runs, 10%) to (100 runs, 5%). At 10**1.5 runs it reads 7.5%, and it reaches 6% at 10**1.8 runs.
    base = dict(family="er", params={"p": 0.25}, n=40, restarts=1, exploded=False, pasqal_runs=1)
    pts = [("gls", (1, 1, 1), 10, 10.0), ("gls", (8, 3, 30), 100, 5.0), ("lin-h4", (4, 2, 10), 10 ** 1.5, 6.0)]
    rows = [dict(base, seed=i, method=m, k=k, s=s, l=l, quantum_runs=r, gap_pct=g)
            for i in range(5) for m, (k, s, l), r, g in pts]
    (row,) = curve_gain(rows, "lin-h4")
    assert row["d_gap"] == pytest.approx(-1.5) and row["run_ratio"] == pytest.approx(10 ** -0.3)
    assert row["d_gap_lo"] == pytest.approx(row["d_gap_hi"])

def test_load_combines_reference_and_method_files(tmp_path):
    inst = dict(family="er", params={"p": 0.25}, n=40, seed=0, exploded=False)
    (tmp_path / "a.jsonl").write_text(json.dumps(dict(inst, method="gls", size=9, proven=None)) + "\n")
    (tmp_path / "b.jsonl").write_text(json.dumps(dict(inst, method="exact", size=10, proven=True)) + "\n")
    gls = next(r for r in load([tmp_path / "a.jsonl", tmp_path / "b.jsonl"]) if r["method"] == "gls")
    assert gls["best_known"] == 10 and gls["gap_pct"] == 10.0 and gls["proven"] is True

def test_every_beam_method_runs_with_trained_models(tmp_path):
    rng = np.random.default_rng(0)
    for hook, d in (("h2", len(VERTEX_FEATURES)), ("h4", len(PIECE_FEATURES))):
        X, g, y = rng.standard_normal((60, d)), np.repeat(np.arange(6), 10), rng.integers(0, 3, 60)
        save(fit_linear(X, y, g), tmp_path / f"lin-{hook}.json")
        save(fit_lgbm(X, y, g, X, y, g, eval_at=2), tmp_path / f"lgb-{hook}.txt")
    (tmp_path / "cma-h2.json").write_text(json.dumps({"theta": [6.0, 0.5, 0.0]}))
    recs = run_instance("er", {"p": 0.25}, 25, 0, BEAM_METHODS, widths=[(2, 2, 5)], restarts=[1],
                        time_limit_s=10, models_dir=tmp_path)
    assert {r["method"] for r in recs} == set(BEAM_METHODS) and all(r["size"] for r in recs)
