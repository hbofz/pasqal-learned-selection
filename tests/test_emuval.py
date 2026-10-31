import json, time
import numpy as np, pytest
from learned_selection import bandit, curves, emuval
from learned_selection.cloud import CloudPieceSolver, CountsCache, uniform_piece_solver, virtual_device
from learned_selection.emuval import done_keys, jobs, main, replay, run_grid
from learned_selection.engine import exact_piece_solver
from test_bandit import ROUND_TOY, _round_models
from test_cloud import FakeClient

SMALL_RUNS = [("lin-h2h4+lin-h5", 4, 2, 10), ("bandit-r", 4, 2, 10), ("bandit-r", 4, 2, 9002), ("gls", 4, 2, 10)]
ER = [("er", {"p": 0.25})]


@pytest.fixture
def models(tmp_path, monkeypatch):
    d = tmp_path / "models"
    d.mkdir()
    _round_models(d)
    (d / "bandit-r").mkdir()
    (d / "bandit-r" / "theta_10.json").write_text(json.dumps(dict(theta=np.zeros((4, 8)).tolist())))
    monkeypatch.setattr(bandit, "ROUND_STACK", ROUND_TOY)
    monkeypatch.setattr(emuval, "RUNS", SMALL_RUNS)
    return d


@pytest.fixture
def nosleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


def todo(seeds=(0,), **kw):
    return jobs(seeds, settings=ER, sizes=(30,), **kw)


def cloud(tmp_path, client=None):
    return CloudPieceSolver(client or FakeClient(), virtual_device(), cache=CountsCache(tmp_path / "c.jsonl"), poll_s=0)


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_jobs_order_and_filter(monkeypatch):
    all_jobs = jobs(range(1020, 1022))
    assert len(all_jobs) == 2 * 26 * 6
    assert [j[3] for j in all_jobs] == [1020] * 156 + [1021] * 156
    assert [(j[4], j[5:]) for j in all_jobs[:6]] == [(m, (k, s, l)) for m, k, s, l in emuval.RUNS]
    assert {j[4] for j in jobs([1020], methods=["bandit-r", "gls"])} == {"bandit-r", "gls"}
    assert len(jobs([1020], methods=["bandit-r"])) == 2 * 26
    assert [j[3] for j in jobs([9002, 1020], settings=ER, sizes=(30,))][:1] == [1020] and len(jobs([9002])) == 156


def test_runs_are_the_designs():
    assert emuval.RUNS == [("bld-h3+dgin-h4+dgin-h5", 4, 2, 10), ("bandit-r", 4, 2, 10), ("bandit-r", 4, 2, 9002),
                           ("bld-h3+sage-h4+dsage-h5+mcts-c0.5", 4, 2, 100), ("lin-h2h4+sage-h5", 4, 2, 10), ("gls", 4, 2, 10)]


def test_records_are_bench_records(tmp_path, models):
    out = tmp_path / "o.jsonl"
    assert run_grid(todo(), lambda j: uniform_piece_solver(), "uniform-repair", out, 2, models) == 0
    rows = curves.load([out])
    assert len(rows) == 4 and {r["piece_solver"] for r in rows} == {"uniform-repair"}
    assert {(r["method"], r["l"]) for r in rows} == {(m, l) for m, _, _, l in SMALL_RUNS}
    assert all(r["pasqal_runs"] > 0 for r in rows)


def test_resume_skips_done_and_keeps_widths_apart(tmp_path, models):
    out, seen = tmp_path / "o.jsonl", []
    all_jobs = todo()
    assert run_grid(all_jobs, lambda j: exact_piece_solver, "exact-top-s", out, 2, models) == 0
    done = done_keys(out)
    assert len(done) == 4 and {k[4:] for k in done} >= {("bandit-r", 4, 2, 10), ("bandit-r", 4, 2, 9002)}
    rest = [j for j in all_jobs if emuval._key(*j) not in done]
    assert rest == [] and run_grid(rest, lambda j: seen.append(j), "x", out, 2, models) == 0 and seen == []
    assert len(lines(out)) == 4


def test_failed_job_is_reported_and_others_finish(tmp_path, models, capsys):
    out, calls, base = tmp_path / "o.jsonl", tmp_path / "calls.jsonl", cloud(tmp_path)

    class Flaky:
        n = 0
        def answer(self, piece, mapping, s):
            self.n += 1
            if self.n > 1:
                raise RuntimeError("job lost")
            return base.answer(piece, mapping, s)
    flaky = Flaky()
    solve_for = lambda j: flaky if j[4] == "gls" else base
    all_jobs = todo()
    assert run_grid(all_jobs, solve_for, "emu-mps", out, 2, models, calls) == 1
    assert flaky.n >= 2 and "RuntimeError: job lost" in capsys.readouterr().err
    assert sorted(r["method"] for r in lines(out)) == ["bandit-r", "bandit-r", "lin-h2h4+lin-h5"]
    assert lines(calls) and "gls" not in {c["method"] for c in lines(calls)}
    rest = [j for j in all_jobs if emuval._key(*j) not in done_keys(out)]
    assert [j[4] for j in rest] == ["gls"] and run_grid(rest, lambda j: base, "emu-mps", out, 2, models, calls) == 0
    assert len(done_keys(out)) == 4


def test_torn_record_line(tmp_path, models, capsys):
    out, (a, b, *_) = tmp_path / "o.jsonl", todo()
    run_grid([a], lambda j: exact_piece_solver, "exact-top-s", out, 1, models)
    out.write_text(out.read_text() + '{"family": "er", "par')
    assert done_keys(out) == {emuval._key(*a)} and "not valid JSON" in capsys.readouterr().err
    run_grid([b], lambda j: exact_piece_solver, "exact-top-s", out, 1, models)
    text = out.read_text().splitlines()
    assert len(text) == 3 and text[1].startswith('{"family": "er", "par') and json.loads(text[2])["method"] == b[4]
    assert done_keys(out) == {emuval._key(*a), emuval._key(*b)}


def test_replay_matches_and_finds_a_miss(tmp_path, models, capsys):
    out, calls, solver = tmp_path / "o.jsonl", tmp_path / "calls.jsonl", cloud(tmp_path)
    all_jobs = todo()
    assert run_grid(all_jobs, lambda j: solver, "emu-mps", out, 2, models, calls) == 0
    cache = tmp_path / "c.jsonl"
    assert replay(out, cache, 1, models) == (4, 0)
    assert replay(out, cache, 2, models) == (2, 0)
    assert [j for j in all_jobs if emuval._key(*j) not in done_keys(out)] == []
    rows = cache.read_text().splitlines()
    cache.write_text("\n".join(rows[1:]) + "\n")
    same, differ = replay(out, cache, 1, models)
    assert differ >= 1 and same + differ == 4 and "DIFFER" in capsys.readouterr().out


def test_calls_log_line(tmp_path, models):
    out, calls, solver = tmp_path / "o.jsonl", tmp_path / "calls.jsonl", cloud(tmp_path)
    assert run_grid(todo(), lambda j: solver, "emu-mps", out, 2, models, calls) == 0
    log = lines(calls)
    fields = {"family", "params", "n", "seed", "method", "k", "s", "l", "key", "atoms", "returned", "asked"}
    assert log and all(set(c) == fields and len(c["key"]) == 64 and 5 < c["atoms"] <= 30 for c in log)
    assert all(c["returned"] <= c["asked"] for c in log)
    for r in lines(out):
        mine = [c for c in log if (c["method"], c["k"], c["s"], c["l"]) == (r["method"], r["k"], r["s"], r["l"])]
        assert len(mine) == r["pasqal_runs"] > 0 and sum(c["atoms"] for c in mine) == r["atoms"]
    assert {c["asked"] for c in log if (c["method"], c["l"]) == ("bandit-r", 10)} == {1}      # arm 0 of the round bandit
    assert {c["asked"] for c in log if (c["method"], c["l"]) == ("bandit-r", 9002)} == {2}


def test_cli_refuses(tmp_path, monkeypatch):
    def boom():
        raise AssertionError("client() was called")
    monkeypatch.setattr(emuval, "client", boom)
    base = ["--seeds", "1020", "--out", str(tmp_path / "o.jsonl")]
    for bad in (["--solver", "cloud"], ["--solver", "cloud", "--cache", "x"], ["--solver", "uniform", "--cache", "x"],
                ["--solver", "exact", "--calls", "x"]):
        with pytest.raises(SystemExit) as e:
            main([*bad, *base])
        assert e.value.code == 2
    assert not (tmp_path / "o.jsonl").exists()


def test_cli_runs_resumes_and_replays(tmp_path, models, monkeypatch, nosleep, capsys):
    fake = FakeClient()
    monkeypatch.setattr(emuval, "client", lambda: fake)
    out, cache, calls = tmp_path / "o.jsonl", tmp_path / "c.jsonl", tmp_path / "calls.jsonl"
    argv = ["--solver", "cloud", "--seeds", "0", "--out", str(out), "--cache", str(cache), "--calls", str(calls),
            "--workers", "2", "--models", str(models), "--settings", "3", "--sizes", "30"]
    assert main(argv) == 0 and len(lines(out)) == 12 and {r["piece_solver"] for r in lines(out)} == {"emu-mps"}
    assert fake.kwargs[0]["tags"] == ["plan5"] and len(cache.read_text().splitlines()) == fake.calls
    made = fake.calls
    assert main(argv) == 0 and len(lines(out)) == 12 and fake.calls == made and "0 jobs to run" in capsys.readouterr().out
    assert main(["--solver", "cloud", "--out", str(out), "--cache", str(cache), "--models", str(models), "--replay", "5"]) == 0
    assert "0 differ" in capsys.readouterr().out
    u = tmp_path / "u.jsonl"
    assert main(["--solver", "uniform", "--seeds", "0", "--out", str(u), "--models", str(models), "--settings", "1",
                 "--sizes", "30", "--methods", "gls"]) == 0
    assert [r["piece_solver"] for r in lines(u)] == ["uniform-repair"]
