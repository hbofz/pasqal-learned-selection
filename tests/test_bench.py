from learned_selection.bench import run_instance, gap_pct

KEYS = {"family", "n", "p", "seed", "method", "k", "s", "l", "restarts", "lattice", "piece_solver",
        "includes_classical_time", "size", "best_known", "proven", "gap_pct", "quantum_runs",
        "pasqal_runs", "atoms", "rounds", "classical_s", "solver_s"}

def test_records_complete_and_consistent():
    recs = run_instance(n=25, p=0.25, seed=0, k=4, s=2, l=10, restarts=1, time_limit_s=10)
    assert {r["method"] for r in recs} == {"gls-code", "gls-paper", "greedy-pasqal", "greedy-mindeg", "exact"}
    for r in recs:
        assert KEYS <= r.keys() and 0 <= r["gap_pct"] <= 100 and r["size"] <= r["best_known"]
    assert gap_pct(9, 10) == 10.0 and gap_pct(0, 0) == 0.0
