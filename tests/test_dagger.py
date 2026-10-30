import sys
from learned_selection import dagger
from learned_selection.dagger import commands

def test_first_iteration_rolls_out_the_starting_method():
    cmds = commands("lin", 1, ["b.pkl"], ["v.pkl"], "M", "C", 24, 6)
    assert len(cmds) == 3
    label = " ".join(cmds[0])
    assert "--policy lin-h2h4+sage-h5" in label and "--models models" in label and "--split train" in label
    assert "--jobs dlin-h2,dlin-h4,dsage-h5" in " ".join(cmds[2]) and "--out M/it1" in " ".join(cmds[2])

def test_later_iterations_use_the_last_models_and_aggregate():
    cmds = commands("gin", 3, ["b.pkl"], ["v.pkl"], "M", "C", 24, 6)
    label, train = " ".join(cmds[0]), cmds[2]
    assert "--policy dgin-h2h4h5" in label and "--models M/it2" in label
    i = train.index("--train")
    assert train[i + 1:i + 5] == ["b.pkl", "C/it1_train.pkl", "C/it2_train.pkl", "C/it3_train.pkl"]
    j = train.index("--val")
    assert train[j + 1:j + 5] == ["v.pkl", "C/it1_val.pkl", "C/it2_val.pkl", "C/it3_val.pkl"]

def test_shard_and_max_states_reach_only_the_label_steps():
    cmds = commands("lin", 1, ["b.pkl"], ["v.pkl"], "M", "C", 4, 2, max_states=5, shard="0/80")
    for c in cmds[:2]:
        assert c[c.index("--shard") + 1] == "0/80" and c[c.index("--max-states") + 1] == "5"
        assert c[c.index("--workers") + 1] == "4"
    assert "--shard" not in cmds[2] and cmds[2][cmds[2].index("--threads") + 1] == "2"
    assert cmds[0][cmds[0].index("--out") + 1] == "C/it1_train.pkl"
    assert cmds[1][cmds[1].index("--out") + 1] == "C/it1_val.pkl"
    assert cmds[0][:3] == [sys.executable, "-m", "learned_selection.graphdata"]

def test_main_resumes_past_finished_steps(tmp_path, monkeypatch):
    cache, out = tmp_path / "c", tmp_path / "m"
    cache.mkdir()
    (cache / "it1_train.pkl").write_bytes(b"done")
    seen = []
    monkeypatch.setattr(dagger.subprocess, "run", lambda argv, check: seen.append((argv, check)))
    dagger.main(["--stack", "lin", "--iters", "1", "--base-train", "b.pkl", "--base-val", "v.pkl",
                 "--out", str(out), "--cache", str(cache), "--workers", "2", "--threads", "1"])
    assert all(check for _, check in seen)
    splits = [a[a.index("--split") + 1] if "--split" in a else "train-step" for a, _ in seen]
    assert splits == ["val", "train-step"]
    # a finished training step is skipped too
    (cache / "it1_val.pkl").write_bytes(b"done")
    (out / "it1").mkdir(parents=True)
    (out / "it1" / "train_meta_dsage-h5.json").write_text("{}")
    seen.clear()
    dagger.main(["--stack", "lin", "--iters", "1", "--base-train", "b.pkl", "--base-val", "v.pkl",
                 "--out", str(out), "--cache", str(cache), "--workers", "2", "--threads", "1"])
    assert seen == []
