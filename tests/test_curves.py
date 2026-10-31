import json, pytest
from learned_selection.curves import (COLORS, PAIRED_MARGIN, _print_settings, _stats, _style, _tables, curve_gain,
                                      envelope, gap_at, headline, main, paired_gain, plot, runs_at, summarize)
from learned_selection.instances import (HELD_OUT_SETTINGS, SEL_SEEDS, TEST_SEEDS, TEST_SIZES, TRAIN_SETTINGS,
                                         VAL_SEEDS)

# The GLS envelope in these rows runs from (10 runs, 10%) to (100 runs, 5%). At 10**1.5 runs it reads 7.5%, and it
# reaches 6% at 10**1.8 runs.
GLS = [((1, 1, 1), 10, 10.0), ((8, 3, 30), 100, 5.0)]
(FA, PA), (FB, PB) = TRAIN_SETTINGS[:2]


def _rows(family, params, n, seeds, runs, gap, method="lin-h4"):
    """GLS at two widths and `method` at (4, 2, 10) on each seed; best known 100, so size is 100 - gap."""
    pts = [("gls", *p) for p in GLS] + [(method, (4, 2, 10), runs, gap)]
    return [dict(family=family, params=params, n=n, seed=i, method=m, k=k, s=s, l=l, restarts=1, exploded=False,
                 quantum_runs=r, pasqal_runs=r, over_30=0, gap_pct=g, size=100 - g)
            for i in seeds for m, (k, s, l), r, g in pts]


def test_envelope_drops_dominated_points():
    pts = [(50, 8.0), (10, 10.0), (20, 12.0), (100, 5.0), (200, 5.0), (30, 10.0)]
    assert envelope(pts) == [(10, 10.0), (50, 8.0), (100, 5.0)]


def test_gap_and_runs_at_interpolate_in_log_runs_across_segments():
    env = [(10, 10.0), (100, 6.0), (1000, 4.0)]
    assert gap_at(env, 10 ** 1.5) == pytest.approx(8.0) and gap_at(env, 10 ** 2.5) == pytest.approx(5.0)
    assert gap_at(env, 10) == pytest.approx(10.0) and gap_at(env, 1000) == pytest.approx(4.0)
    assert runs_at(env, 8.0) == pytest.approx(10 ** 1.5) and runs_at(env, 5.0) == pytest.approx(10 ** 2.5)
    assert runs_at(env, 10.0) == pytest.approx(10) and runs_at(env, 4.0) == pytest.approx(1000)
    assert gap_at(env, 9) is None and gap_at(env, 1001) is None
    assert runs_at(env, 10.5) is None and runs_at(env, 3.9) is None


def test_val_split_pools_training_settings_at_n100_on_val_seeds():
    rows = (_rows(FA, PA, 100, VAL_SEEDS, 10 ** 1.5, 6.0) + _rows(FB, PB, 100, VAL_SEEDS, 10 ** 1.5, 6.0)
            + _rows(FA, PA, 100, TEST_SEEDS[:5], 10 ** 1.5, 9.0)          # other seeds in a val cell
            + _rows(FA, PA, 150, VAL_SEEDS, 10 ** 1.5, 9.0)               # outside the val grid
            + _rows(*HELD_OUT_SETTINGS[0], 100, VAL_SEEDS, 10 ** 1.5, 9.0))
    (h,) = headline(rows, "quantum_runs", reps=50, seeds=VAL_SEEDS)
    assert (h["pool"], h["method"], h["cells"], h["instances"]) == ("validation", "lin-h4", 2, 10)
    assert h["d_gap"] == pytest.approx(-1.5) and h["run_ratio"] == pytest.approx(10 ** -0.3)


def test_run_ratio_is_undefined_not_zero_when_the_method_makes_no_runs():
    (row,) = curve_gain(_rows("er", {"p": 0.75}, 100, range(3), 0, 6.0), "lin-h4", reps=50)
    assert row["run_ratio"] is None and row["d_gap"] is None
    assert row["run_ratio_beats"] == row["d_gap_beats"] == 1 and row["run_ratio_none"] == 50


def test_headline_counts_undefined_cells_by_side_and_averages_the_defined_ones():
    cells = [(FA, PA, TEST_SIZES[0]), (FA, PA, TEST_SIZES[1]), (FB, PB, TEST_SIZES[0]), (FB, PB, TEST_SIZES[1])]
    method = [(10 ** 1.5, 6.0),   # inside: d_gap -1.5, run_ratio 10**-0.3
              (10 ** 1.5, 4.0),   # below the best gap: d_gap -3.5, run_ratio beats the envelope
              (1000, 6.0),        # past the most runs: d_gap expensive, run_ratio 10**1.2
              (5, 12.0)]          # under the fewest runs and above the worst gap: d_gap beats, run_ratio expensive
    rows = [r for (f, p, n), (runs, gap) in zip(cells, method) for r in _rows(f, p, n, TEST_SEEDS[:3], runs, gap)]
    (h,) = headline(rows, "quantum_runs", reps=50)
    assert h["pool"] == "training" and h["cells"] == 4
    assert h["d_gap"] == pytest.approx(-2.5) and h["d_gap_lo"] == pytest.approx(-2.5)
    assert h["run_ratio"] == pytest.approx(10 ** 0.45) and h["run_ratio_hi"] == pytest.approx(10 ** 0.45)
    assert (h["d_gap_cells"], h["d_gap_beats"], h["d_gap_expensive"]) == (2, 1, 1)
    assert (h["run_ratio_cells"], h["run_ratio_beats"], h["run_ratio_expensive"]) == (2, 1, 1)
    assert h["d_gap_none"] == h["run_ratio_none"] == 0


def test_a_resample_averages_over_the_cells_defined_in_it():
    # The second cell's mean runs (52.5) are inside the envelope, but a resample drawing 5 runs twice is not.
    rows = (_rows(FA, PA, 100, TEST_SEEDS[:2], 10 ** 1.5, 6.0)
            + _rows(FB, PB, 100, TEST_SEEDS[:1], 5, 6.0) + _rows(FB, PB, 100, TEST_SEEDS[1:2], 100, 6.0))
    (h,) = headline(rows, "quantum_runs", reps=200)
    assert h["d_gap_cells"] == 2 and h["d_gap_none"] == 0


def test_cli_val_split_reports_val_seeds_only(tmp_path, capsys):
    rows = _rows(FA, PA, 100, VAL_SEEDS, 10 ** 1.5, 6.0) + _rows(FA, PA, 100, TEST_SEEDS[:5], 10 ** 1.5, 9.0)
    rows += [dict(r, method="exact", k=None, s=None, l=None, size=100, proven=True) for r in rows if r["k"] == 1]
    path = tmp_path / "val.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    main([str(path), "--out", str(tmp_path / "plots"), "--split", "val"])
    out = capsys.readouterr().out
    assert "validation" in out and "-1.50 [-1.50, -1.50]" in out and "over 30" in out
    assert "9.00" not in out    # the test-seed rows, 9% gap, never reach a table


def test_headline_takes_another_baseline():
    rows = [dict(r, method="lin-h2h4") if r["method"] == "gls" else r
            for r in _rows(FA, PA, TEST_SIZES[0], TEST_SEEDS, 10 ** 1.5, 6.0, method="gin-h2")]
    out = headline(rows, "quantum_runs", baseline="lin-h2h4")
    (row,) = [h for h in out if h["method"] == "gin-h2"]
    assert row["d_gap"] == pytest.approx(-1.5) and row["baseline"] == "lin-h2h4"


def test_plot_draws_composed_methods(tmp_path):
    rows = [dict(family="er", params={"p": 0.25}, n=100, method=m, k=k, s=2, l=10, restarts=1, gap=g, gap_lo=g - 1,
                 gap_hi=g + 1, quantum_runs=r, pasqal_runs=r)
            for m in ("gls", "lin-h2h4", "gin-h2h4h5", "lin-h2h4+gin-h5", "sage-h5") for k, r, g in ((2, 10, 9.0), (4, 30, 7.0))]
    assert len(plot(rows, tmp_path)) == 1
    assert len(plot(rows, tmp_path / "some", methods=["gls", "sage-h5"])) == 1


def test_cli_prints_oracle_references_on_val_only(tmp_path, capsys):
    path = tmp_path / "rows.jsonl"
    for split, seeds, n in (("val", VAL_SEEDS, 100), ("test", TEST_SEEDS, TEST_SIZES[0])):
        rows = [r for m in ("exact-h5", "lin-h2h4+exact-h5") for r in _rows(FA, PA, n, seeds, 10 ** 1.5, 6.0, method=m)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        main([str(path), "--out", str(tmp_path / "plots"), "--split", split])
        out = capsys.readouterr().out
        assert ("oracle references (exact value pruning, not a structure-track method)" in out) == (split == "val")
        if split == "val":
            assert out.split("oracle references")[1].count("exact-h5") >= 2 and "-1.58" in out


def test_headline_uses_a_methods_own_config():
    # the method has records at (4, 2, 100) only: with its config it is read, without it there is no row
    rows = [dict(r, l=100) if r["method"] == "m" else r
            for r in _rows(FA, PA, 100, SEL_SEEDS, 10 ** 1.5, 6.0, method="m")]
    assert headline(rows, "quantum_runs", seeds=SEL_SEEDS, methods=["m"]) == []
    (h,) = headline(rows, "quantum_runs", seeds=SEL_SEEDS, methods=["m"], configs={"m": (4, 2, 100, 1)})
    assert h["pool"] == "selection" and h["d_gap"] == pytest.approx(-1.5)
    assert h["d_clamp"] == pytest.approx(-1.5) and h["instances"] == len(SEL_SEEDS)


def test_clamped_d_gap_holds_the_envelope_flat_past_its_ends():
    # baseline points (10 runs, 10%) and (100 runs, 5%)
    assert _stats([1000.0, 10.0, 100.0], [6.0, 10.0, 5.0])[2] == pytest.approx(1.0)     # dearer: 6 - 5, d_gap undefined
    assert _stats([1.0, 10.0, 100.0], [6.0, 10.0, 5.0])[2] == pytest.approx(-4.0)       # cheaper: 6 - 10
    assert _stats([10 ** 1.5, 10.0, 100.0], [6.0, 10.0, 5.0])[2] == pytest.approx(-1.5)  # inside: the same as d_gap
    assert _stats([0.0, 10.0, 100.0], [6.0, 10.0, 5.0])[2] == pytest.approx(-4.0)       # no runs: the first point
    assert _stats([5.0], [6.0])[2] is None                                               # no envelope


def test_clamped_d_gap_is_defined_where_d_gap_is_not():
    rows = _rows(FA, PA, 100, TEST_SEEDS[:3], 1000, 6.0)
    (h,) = headline(rows, "quantum_runs", reps=50)
    assert h["d_gap"] is None and h["d_clamp"] == pytest.approx(1.0) and h["d_clamp_lo"] == pytest.approx(1.0)


def test_plot_and_tables_take_new_kinds_and_search_names(tmp_path, capsys):
    search, dual = "lin-h2+gin-h4+sage-h5+mcts", "dlin-h2h4+dsage-h5"
    rows = (_rows(FA, PA, 100, TEST_SEEDS[:3], 10 ** 1.5, 6.0, method=dual)
            + [r for r in _rows(FA, PA, 100, TEST_SEEDS[:3], 10 ** 1.5, 4.0, method=search) if r["method"] == search])
    for r in rows:
        if r["method"] == search:
            r["l"] = 100
    summary = summarize(rows, reps=50)
    assert len(plot(summary, tmp_path)) == 1
    configs = {search: (4, 2, 100, 1)}
    _print_settings(_tables(rows), summary, "gls", "quantum_runs", sorted(_tables(rows)), configs)
    out = capsys.readouterr().out
    assert dual in out and search in out
    assert [line for line in out.splitlines() if search in line][0].split()[1] == "3"
    (h,) = headline(rows, "quantum_runs", reps=50, methods=[search], configs=configs)
    assert h["method"] == search and h["d_gap"] == pytest.approx(-3.5)


def test_cli_sel_split_reads_each_method_at_its_config(tmp_path, capsys):
    rows = [dict(r, l=100) if r["method"] == "lin-h4" else r for r in
            _rows(FA, PA, 100, SEL_SEEDS, 10 ** 1.5, 6.0, method="lin-h4") + _rows(FA, PA, 100, TEST_SEEDS[:3], 10 ** 1.5, 9.0)]
    path = tmp_path / "sel.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    main([str(path), "--out", str(tmp_path / "plots"), "--split", "sel", "--methods", "lin-h4",
          "--config", "lin-h4=4,2,100"])
    out = capsys.readouterr().out
    assert "selection" in out and "-1.58 [-1.58, -1.58]" in out and "9.00" not in out


def test_load_passes_exact_duplicates_and_names_conflicting_records(tmp_path):
    from learned_selection.curves import load
    rows = _rows(FA, PA, 100, TEST_SEEDS[:2], 10 ** 1.5, 6.0)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("".join(json.dumps(r) + "\n" for r in rows))
    b.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert len(load([a, b])) == 2 * len(rows)
    for field in ("size", "quantum_runs"):
        b.write_text("".join(json.dumps(dict(r, **{field: r[field] + 1}) if r["method"] == "lin-h4" else r) + "\n"
                             for r in rows))
        with pytest.raises(ValueError, match="lin-h4"):
            load([a, b])


def test_style_and_plot_take_tilt_builder_and_policy_gradient_names(tmp_path):
    names = ["dgin-h2h4h5+tilt-0.1", "dgin-h2+pg-h4+dgin-h5", "bld-h3+bgin-h4+bgin-h5"]
    rows = [dict(family="er", params={"p": 0.25}, n=100, method=m, k=k, s=2, l=10, restarts=1, gap=g, gap_lo=g - 1,
                 gap_hi=g + 1, quantum_runs=r, pasqal_runs=r)
            for m in ("gls", *names) for k, r, g in ((2, 10, 9.0), (4, 30, 7.0))]
    assert len(plot(rows, tmp_path)) == 1
    assert _style("dgin-h2h4h5+tilt-0.1") != _style("dgin-h2h4h5")
    assert _style("dgin-h2h4h5+tilt-0.1")[2] == "--"
    assert _style("dgin-h2+pg-h4+dgin-h5")[0] == _style("bld-h3+bgin-h4+bgin-h5")[0] == COLORS["az"]
    assert _style("bld-h3+bgin-h4+bgin-h5")[1] == "X" and _style("bgin-h2h4h5")[0] == COLORS["gin"]


def _pair_rows(cell, seeds, method, base, config=(4, 2, 10)):
    """The method's and the baseline's (runs, gap) on each seed as rows at one config; best known 100."""
    (f, p, n), (k, s, l) = cell, config
    return [dict(family=f, params=p, n=n, seed=i, method=m, k=k, s=s, l=l, restarts=1, exploded=False,
                 quantum_runs=r, pasqal_runs=r, over_30=0, gap_pct=g, size=100 - g)
            for i, a, b in zip(seeds, method, base) for m, (r, g) in (("m", a), ("gls", b))]


def test_paired_gain_by_hand():
    cell, seeds = (FA, PA, TEST_SIZES[0]), TEST_SEEDS
    base = [(100, 5.0)] * len(seeds)
    for method, ratio, d, verdict in (([(50, 5.0)] * len(seeds), 0.5, 0.0, "saves runs"),
                                      ([(100, 6.0)] * len(seeds), 1.0, 1.0, "costs gap"),
                                      (base, 1.0, 0.0, "no clear change")):
        (h,) = paired_gain(_pair_rows(cell, seeds, method, base), "m", "gls", reps=50)
        assert (h["pool"], h["cells"], h["instances"], h["no_runs"]) == ("training", 1, len(seeds), 0)
        assert h["run_ratio"] == pytest.approx(ratio) and h["d_gap"] == pytest.approx(d)
        assert h["run_ratio_lo"] == pytest.approx(ratio) and h["d_gap_hi"] == pytest.approx(d)
        assert h["verdict"] == verdict
    assert PAIRED_MARGIN == 0.25


def test_paired_gain_leaves_out_exploded_and_zero_run_instances():
    cell, seeds = (FA, PA, TEST_SIZES[0]), TEST_SEEDS[:4]
    method = [(50, 5.0), (0, 9.0), (200, 5.0), (80, 7.0)]      # seed 1 made no runs, seed 3 exploded
    rows = _pair_rows(cell, seeds, method, [(100, 5.0)] * 4)
    for r in rows:
        r["exploded"] = r["method"] == "m" and r["seed"] == seeds[3]
    (h,) = paired_gain(rows, "m", "gls", reps=50)
    assert h["instances"] == 3 and h["no_runs"] == 1
    assert h["run_ratio"] == pytest.approx(1.0)                # geometric mean of 0.5 and 2
    assert h["d_gap"] == pytest.approx(4 / 3)                  # (0 + 4 + 0) / 3: the zero-run instance still counts
    assert paired_gain(rows, "m", "gls", seeds=VAL_SEEDS) == []


def test_paired_gain_reads_the_baseline_at_its_own_config():
    cell, seeds = (FA, PA, TEST_SIZES[0]), TEST_SEEDS[:4]
    ours = [r for r in _pair_rows(cell, seeds, [(50, 5.0)] * 4, [(0, 0.0)] * 4, config=(2, 2, 5)) if r["method"] == "m"]
    theirs = [r for r in _pair_rows(cell, seeds, [(0, 0.0)] * 4, [(100, 5.0)] * 4) if r["method"] == "gls"]
    assert paired_gain(ours + theirs, "m", "gls", config=(2, 2, 5, 1), reps=50) == []
    (h,) = paired_gain(ours + theirs, "m", "gls", config=(2, 2, 5, 1), base_config=(4, 2, 10, 1), reps=50)
    assert h["run_ratio"] == pytest.approx(0.5) and h["d_gap"] == pytest.approx(0.0)
