import json, socket, threading, time
from types import SimpleNamespace
import networkx as nx, pytest, requests
from requests.adapters import HTTPAdapter
from learned_selection import cloud
from learned_selection.cloud import HTTP_TIMEOUT_S, CloudPieceSolver, CountsCache, _login, bound_http_calls, uniform_piece_solver
from learned_selection.solvers import is_independent
from pulser.exceptions.sequence import MinQubitNumberError
from learned_selection.cloud import EMULATION_DEVICE, decode, load_device, calibrated_layout, trap_map, piece_sequence, summarize_counts
from learned_selection.engine import gls_generator
from learned_selection.instances import erdos_renyi
from learned_selection.lattice import load_coords

def test_decode_uses_insertion_order():
    ids = [str(i) for i in range(12)]            # sorted(ids) would put "10", "11" before "2"
    assert decode("001000000001", ids) == ["2", "11"]

@pytest.mark.parametrize("dev", [EMULATION_DEVICE, load_device()], ids=["AnalogDevice", "FRESNEL"])
def test_trap_map_places_every_trap_on_the_calibrated_layout(dev):
    c = load_coords()
    layout, tm = calibrated_layout(dev), trap_map(dev)
    assert sorted(tm) == list(range(61)) and len(set(tm.values())) == 61
    traps = layout.traps_dict
    assert all(abs(traps[tm[t]][0] - c[t][0]) < 1e-3 and abs(traps[tm[t]][1] - c[t][1]) < 1e-3 for t in c)

def test_fresnel_rejects_pieces_under_42_atoms():
    G = erdos_renyi(40, 0.25, 0)
    m = gls_generator()(G, 1)[0]
    assert len(m) < 42
    with pytest.raises(MinQubitNumberError):
        piece_sequence(G.subgraph(m), m, load_device())

def test_piece_sequence_matches_mapping_geometry():
    G = erdos_renyi(40, 0.25, 0)
    m = gls_generator()(G, 1)[0]
    piece, dev, c = G.subgraph(m), EMULATION_DEVICE, load_coords()
    seq = piece_sequence(piece, m, dev)          # building it runs Pulser's device validation
    pos = seq.register.qubits
    assert list(pos) == [str(v) for v in piece]
    for v in piece:
        assert abs(float(pos[str(v)][0]) - c[m[v]][0]) < 1e-3 and abs(float(pos[str(v)][1]) - c[m[v]][1]) < 1e-3

def test_summarize_counts_separates_raw_quality_from_repair():
    piece = nx.path_graph(6)
    ids = [str(v) for v in piece]
    top_g, st_g = summarize_counts(piece, {"101010": 500}, ids, 2)
    top_n, st_n = summarize_counts(piece, {"111111": 250, "110011": 250}, ids, 2)
    assert top_g == [frozenset({0, 2, 4})] and st_g["invalid_rate"] == 0 and st_g["valid_sizes"] == {3: 500}
    assert st_n["invalid_rate"] == 1 and st_n["valid_sizes"] == {}
    assert all(len(S) == 3 for S in top_n)       # repair alone reaches the optimum: decisions must use raw stats

def test_virtual_device_takes_large_pieces():
    from learned_selection.cloud import virtual_device
    from learned_selection.instances import power_law
    G = power_law(150, 8, 0)
    m = next(m for m in gls_generator()(G, len(G)) if len(m) > 30)
    seq = piece_sequence(G.subgraph(m), m, virtual_device())
    assert len(seq.register.qubits) == len(m) and seq.to_abstract_repr()

def test_spread_keeps_evenly_spaced_items():
    from learned_selection.calibrate import spread
    xs = list(range(40))
    picked = spread(xs, 12)
    assert len(picked) == len(set(picked)) == 12 and picked[0] == 0 and picked[-1] == 39 and picked == sorted(picked)
    assert spread(xs, len(xs)) == xs
    assert spread(xs, 1) == [xs[0]]


# --- the cached, polled, retried solver, against a fake client (no test touches the cloud) ---

ODD = lambda ids: {"".join("1" if int(i) % 2 else "0" for i in ids): 500}


class FakeJob:
    def __init__(self, result):
        self.result = result


class FakeBatch:
    def __init__(self, id, plan, counts, refresh_errors):
        self.id, self.plan, self.refresh_errors = id, plan, refresh_errors
        self.status = "PENDING"
        self.created_at, self.start_datetime, self.end_datetime = "t0", "t1", "t2"
        self.ordered_jobs = [FakeJob(None if plan in ("none", "error") else counts)]

    def refresh(self):
        if self.refresh_errors:
            self.refresh_errors -= 1
            raise RuntimeError("network")
        self.status = {"stuck": "RUNNING", "error": "ERROR"}.get(self.plan, "DONE")


class FakeClient:
    """plan: one entry per create_batch call (default "ok"): ok, none, error, raise, stuck."""
    def __init__(self, plan=(), counts=ODD, delay=0.0, refresh_errors=0):
        self.plan, self.counts, self.delay, self.refresh_errors = list(plan), counts, delay, refresh_errors
        self.calls, self.kwargs, self.cancelled, self.batches = 0, [], [], {}
        self._lock = threading.Lock()

    def create_batch(self, seq, jobs, device_type=None, wait=False, tags=None):
        with self._lock:
            self.calls += 1
            n = self.calls
            self.kwargs.append(dict(jobs=jobs, device_type=device_type, wait=wait, tags=tags))
        time.sleep(self.delay)
        plan = self.plan[n - 1] if n <= len(self.plan) else "ok"
        if plan == "raise":
            raise RuntimeError("boom")
        ids = [q["name"] for q in json.loads(seq)["register"]]
        counts = {k: v * jobs[0]["runs"] // 500 for k, v in self.counts(ids).items()}
        b = FakeBatch(f"b{n}", plan, counts, self.refresh_errors)
        self.batches[b.id] = b
        return b

    def cancel_batch(self, id):
        self.cancelled.append(id)


PIECE = nx.path_graph(12)
MAP = {v: v for v in PIECE}
ODDS = frozenset(range(1, 12, 2))


def make(client, path=None, **kw):
    kw.setdefault("poll_s", 0)
    return CloudPieceSolver(client, cache=CountsCache(path) if path else None, **kw)


@pytest.fixture
def nosleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


def test_cache_makes_one_job_across_solvers(tmp_path):
    f, fake = tmp_path / "c.jsonl", FakeClient()
    s1 = make(fake, f)
    a = s1(PIECE, MAP, 3), s1(PIECE, MAP, 3)
    b = make(fake, f)(PIECE, MAP, 3)
    c = make(None, f)(PIECE, MAP, 3)
    assert fake.calls == 1 and a[0] == a[1] == b == c == [ODDS] and s1.log == []
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["batch_id"] == "b1" and rows[0]["runs"] == 500 and rows[0]["emulator"] == "EMU_MPS"
    assert {"key", "atoms", "ids", "edges", "counts", "created_at", "start_datetime", "end_datetime"} <= set(rows[0])
    assert rows[0]["atoms"] == 12 and rows[0]["ids"] == [str(v) for v in PIECE] and rows[0]["edges"][0] == [0, 1]
    assert rows[0]["key"] == s1.answer(PIECE, MAP, 3)[1]


def test_key_changes_with_the_register(tmp_path):
    f, fake = tmp_path / "c.jsonl", FakeClient()
    s = make(fake, f)
    k1 = s.answer(PIECE, MAP, 3)[1]
    k2 = s.answer(PIECE, {v: v + 1 for v in PIECE}, 3)[1]
    k3 = make(fake, f, runs=400).answer(PIECE, MAP, 3)[1]
    assert len({k1, k2, k3}) == 3 and fake.calls == 3


def test_replay_miss_raises(tmp_path):
    with pytest.raises(KeyError):
        make(None, tmp_path / "c.jsonl")(PIECE, MAP, 3)


def test_answer_decodes_by_insertion_order():
    assert sorted(map(str, range(12)))[2] == "10"            # a sorted decode would read different vertices
    assert make(FakeClient())(PIECE, MAP, 1) == [ODDS]


def test_wrong_length_bitstring_raises():
    with pytest.raises(ValueError):
        summarize_counts(PIECE, {"101": 500}, [str(v) for v in PIECE], 1)
    with pytest.raises(ValueError):
        make(FakeClient(counts=lambda ids: {"101": 500}), retries=0)(PIECE, MAP, 1)


def test_short_shots_are_retried(nosleep):
    short = FakeClient(counts=lambda ids: {next(iter(ODD(ids))): 499})
    with pytest.raises(ValueError):
        make(short, retries=1)(PIECE, MAP, 1)
    assert short.calls == 2


def test_failed_job_is_retried_and_not_cached(tmp_path, nosleep):
    f, fake = tmp_path / "c.jsonl", FakeClient(["none", "error", "raise"])
    assert make(fake, f, retries=3)(PIECE, MAP, 1) == [ODDS]
    assert fake.calls == 4 and len(f.read_text().splitlines()) == 1 and fake.batches == {}


def test_retries_exhausted_raises_and_caches_nothing(tmp_path, nosleep):
    f, fake = tmp_path / "c.jsonl", FakeClient(["none"] * 5)
    with pytest.raises(Exception):
        make(fake, f, retries=2)(PIECE, MAP, 1)
    assert fake.calls == 3 and (not f.exists() or f.read_text() == "")


def test_attempts_sleep_60_times_the_attempt(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    make(FakeClient(["none", "none", "ok"]), poll_s=7)(PIECE, MAP, 1)
    assert [x for x in slept if x >= 60] == [60, 120] and 7 in slept


def test_polling_survives_errors_on_one_batch(nosleep):
    fake = FakeClient(refresh_errors=2)
    assert make(fake)(PIECE, MAP, 1) == [ODDS] and fake.calls == 1


def test_deadline_cancels_and_retries(monkeypatch):
    ticks = iter(range(10**6))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    fake = FakeClient(["stuck"])
    assert make(fake, deadline_s=5)(PIECE, MAP, 1) == [ODDS]
    assert fake.cancelled == ["b1"] and fake.calls == 2 and fake.batches == {}


def test_client_store_is_emptied():
    fake = FakeClient()
    make(fake)(PIECE, MAP, 1)
    assert fake.calls == 1 and fake.batches == {}


def test_one_job_for_two_threads(tmp_path):
    f, fake, out = tmp_path / "c.jsonl", FakeClient(delay=0.2), []
    s = make(fake, f)
    ts = [threading.Thread(target=lambda: out.append(s(PIECE, MAP, 1))) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert fake.calls == 1 and out == [[ODDS], [ODDS]]


def test_cache_skips_torn_line(tmp_path, capsys):
    f = tmp_path / "c.jsonl"
    f.write_text('{"key": "a", "x": 1}\n{"key": "b", "x"')
    c = CountsCache(f)
    assert len(c) == 1 and "torn" in capsys.readouterr().err
    assert c.get_or_run("c", lambda: {"x": 3}) == {"x": 3}
    assert len(CountsCache(f)) == 2 and c.get_or_run("a", lambda: pytest.fail("ran")) == {"x": 1}


def test_cache_failing_fn_writes_nothing_and_releases_the_key(tmp_path):
    c = CountsCache(tmp_path / "c.jsonl")
    def bad():
        raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        c.get_or_run("a", bad)
    assert len(c) == 0 and c.get_or_run("a", lambda: {"x": 1}) == {"x": 1} and len(c) == 1


def test_uniform_solver():
    solve = uniform_piece_solver(runs=200)
    a, b = solve(PIECE, MAP, 4), solve(PIECE, MAP, 4)
    assert a == b and 1 <= len(a) <= 4
    assert all(is_independent(PIECE, S) and all(v in S or any(u in S for u in PIECE[v]) for v in PIECE) for S in a)


def test_solver_tags_reach_the_batch():
    fake = FakeClient()
    make(fake, tags=["ev", "x"])(PIECE, MAP, 1)
    kw = fake.kwargs[0]
    assert kw["tags"] == ["ev", "x"] and str(kw["device_type"]) == "EMU_MPS" and kw["jobs"] == [{"runs": 500}] and kw["wait"] is False


def test_log_keeps_entries_without_a_cache():
    s = make(FakeClient())
    s(PIECE, MAP, 2)
    e = s.log[-1]
    assert e["n_atoms"] == 12 and e["batch_id"] == "b1" and e["shots"] == 500 and len(e["key"]) == 64 and "top_sizes" in e


# --- every HTTP call is bounded (pasqal_cloud sets no timeout on the result download or the token query) ---

@pytest.fixture
def original_send(monkeypatch):
    seen = []
    monkeypatch.setattr(HTTPAdapter, "send", lambda self, request, **kw: seen.append(kw.get("timeout")))     # undone after the test
    return seen


def test_http_calls_without_a_timeout_get_the_default(original_send):
    bound_http_calls()
    adapter = HTTPAdapter()
    adapter.send(None, timeout=None)
    adapter.send(None)
    adapter.send(None, timeout=7)
    adapter.send(None, timeout=(3, 9))
    assert original_send == [HTTP_TIMEOUT_S, HTTP_TIMEOUT_S, 7, (3, 9)] and HTTP_TIMEOUT_S == 120


def test_bounding_twice_does_not_stack(original_send):
    bound_http_calls()
    once = HTTPAdapter.send
    bound_http_calls(5)
    assert HTTPAdapter.send is once
    HTTPAdapter().send(None)
    assert original_send == [HTTP_TIMEOUT_S]


def test_silent_server_times_out_under_the_bound(monkeypatch):
    monkeypatch.setattr(HTTPAdapter, "send", getattr(HTTPAdapter.send, "__wrapped__", HTTPAdapter.send))    # real send, rewrapped here only
    bound_http_calls(0.3)
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        t0 = time.monotonic()
        with pytest.raises(requests.Timeout):
            requests.get(f"http://127.0.0.1:{srv.getsockname()[1]}/")
        assert time.monotonic() - t0 < 2


def test_a_timeout_in_the_result_fetch_fails_the_attempt_and_is_retried(nosleep):
    class SlowJob:
        calls = 0
        @property
        def result(self):
            SlowJob.calls += 1
            if SlowJob.calls == 1:
                raise requests.ReadTimeout("result download")
            return ODD([str(v) for v in PIECE])
    fake = FakeClient()
    create = fake.create_batch
    def first_slow(*a, **kw):
        b = create(*a, **kw)
        if fake.calls == 1:
            b.ordered_jobs = [SlowJob()]
        return b
    fake.create_batch = first_slow
    assert make(fake, retries=1)(PIECE, MAP, 1) == [ODDS] and fake.calls == 2


# --- the login: Keychain on macOS, a private file elsewhere (never the real file, never the real client) ---

SECRET = "p@ss w0rd; $&' x "


@pytest.fixture
def linux_login(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.platform", "linux")
    path = tmp_path / "login"
    monkeypatch.setenv("PASQAL_LOGIN_FILE", str(path))
    return path


def test_login_reads_the_file_on_linux(linux_login):
    linux_login.write_text(f"alice\n{SECRET}\n")
    linux_login.chmod(0o600)
    assert _login() == ("alice", SECRET)
    linux_login.write_bytes(f"alice\r\n{SECRET}".encode())       # CRLF, no final newline
    assert _login() == ("alice", SECRET)


def test_login_refuses_an_open_file(linux_login):
    linux_login.write_text(f"alice\n{SECRET}\n")
    for mode in (0o644, 0o640, 0o602):
        linux_login.chmod(mode)
        with pytest.raises(RuntimeError) as e:
            _login()
        assert str(linux_login) in str(e.value) and SECRET not in str(e.value) and "alice" not in str(e.value)


@pytest.mark.parametrize("text", [None, f"{SECRET}\n", f"alice\n{SECRET}\nextra\n", f"alice\n\n", f"\n{SECRET}\n", ""],
                         ids=["missing", "one line", "three lines", "empty password", "empty username", "empty file"])
def test_login_refuses_missing_and_malformed(linux_login, text):
    if text is not None:
        linux_login.write_text(text)
        linux_login.chmod(0o600)
    with pytest.raises(RuntimeError) as e:
        _login()
    assert str(linux_login) in str(e.value) and SECRET.strip() not in str(e.value) and "alice" not in str(e.value) and "extra" not in str(e.value)


def test_login_uses_the_keychain_on_macos(monkeypatch, tmp_path):
    f = tmp_path / "login"
    f.write_text("fileuser\nfilepass\n")
    f.chmod(0o600)
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setenv("PASQAL_LOGIN_FILE", str(f))
    monkeypatch.setattr(type(f), "read_text", lambda self, *a, **k: pytest.fail("read the file"))
    calls = []
    def run(cmd, **kw):
        calls.append((cmd, kw))
        return SimpleNamespace(stdout="pw\n" if "-w" in cmd else '    "acct"<blob>="alice"\n')
    monkeypatch.setattr(cloud.subprocess, "run", run)
    assert _login() == ("alice", "pw")
    assert len(calls) == 2 and all(kw["timeout"] == 30 and c[0] == "security" for c, kw in calls)
