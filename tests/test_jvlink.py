import datetime as dt
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from keibaai import cli, db, jvlink


class FakeRun:
    """32bit Python の呼び出しの代わり。コマンドを記録し、書き出し先に空のファイルを作る。"""

    def __init__(self, code=0):
        self.cmds, self.code = [], code

    def __call__(self, cmd, **kw):
        self.cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, self.code, stdout="RACE: 3 件を書き出しました\n", stderr="")


@pytest.fixture
def fake(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(jvlink.subprocess, "run", run)
    monkeypatch.setattr(jvlink.jvexport, "RT_INTERVAL", 0)
    return run


def _args(cmd, name):
    return cmd[cmd.index(name) + 1:]


def test_export_clears_old_files_and_passes_specs(tmp_path, fake, capsys):
    (tmp_path / "RACE.txt").write_text("old", encoding="utf-8")
    (tmp_path / "RT_0B15.txt").write_text("old", encoding="utf-8")
    jvlink.export(tmp_path, "20261010", ["RACE"], option=2, rt=["0B15:20261011"], python32="py32")
    cmd = fake.cmds[0]
    assert cmd[:3] == ["py32", "-m", "keibaai.jvexport"]
    assert _args(cmd, "--option")[0] == "2" and _args(cmd, "--from")[0] == "20261010"
    assert _args(cmd, "--rt") == ["0B15:20261011"]
    assert not (tmp_path / "RACE.txt").exists() and not (tmp_path / "RT_0B15.txt").exists()
    assert "3 件" in capsys.readouterr().out


def test_export_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(jvlink.subprocess, "run", FakeRun(code=1))
    with pytest.raises(jvlink.JVError):
        jvlink.export(tmp_path, "20261010", ["RACE"])


def test_fetch_results_takes_quick_results_only_for_recent_race_days(tmp_path, fake, monkeypatch):
    got = {}
    monkeypatch.setattr("keibaai.jvrace.import_races",
                        lambda conn, raw, since, refresh_since=None: got.update(raw=raw, since=since, refresh=refresh_since) or (2, 30))
    conn = db.connect(":memory:")
    today = dt.date(2026, 10, 15)  # 木曜
    n = jvlink.fetch_results(conn, dt.date(2026, 10, 1), dt.date(2026, 10, 14), tmp_path, today=today)
    assert n == (2, 30)
    cmd = fake.cmds[0]
    assert _args(cmd, "--from")[0] == "20260930" and _args(cmd, "--option")[0] == "1"
    assert _args(cmd, "--rt") == ["0B15:20261010", "0B15:20261011", "0B15:20261012"]  # 直近7日の土日月だけ
    assert got == {"raw": tmp_path / "recent", "since": "2026-10-01", "refresh": "2026-10-01"}


def test_fetch_entries_skips_odds_when_no_numbers_yet(tmp_path, fake, monkeypatch):
    monkeypatch.setattr("keibaai.jventries.race_keys", lambda raw, date: ["2026101708040101"])  # 出走馬名表はある
    monkeypatch.setattr("keibaai.jventries.read_entries", lambda raw, date: [])  # 馬番はまだ
    monkeypatch.setattr("keibaai.jventries.import_entries", lambda conn, raw, date, at: 0)
    assert jvlink.fetch_entries(db.connect(":memory:"), "2026-10-17", raw_root=tmp_path) == 0
    assert len(fake.cmds) == 1  # 出馬表だけ。馬番がないのでオッズは開かない
    assert _args(fake.cmds[0], "--rt") == ["0B15:20261017", "0B14:20261017"] and _args(fake.cmds[0], "--from")[0] == "20261010"


def test_fetch_entries_takes_odds_per_race(tmp_path, fake, monkeypatch):
    from types import SimpleNamespace as NS

    monkeypatch.setattr("keibaai.jventries.race_keys",
                        lambda raw, date: ["2026101708040101", "2026101708040102", "2026101708040103"])
    monkeypatch.setattr("keibaai.jventries.read_entries",  # 3R はまだ馬番がない
                        lambda raw, date: [NS(race=NS(race_id="202608040101")), NS(race=NS(race_id="202608040102"))])
    monkeypatch.setattr("keibaai.jventries.import_entries", lambda conn, raw, date, at: 2)
    assert jvlink.fetch_entries(db.connect(":memory:"), "2026-10-17", raw_root=tmp_path) == 2
    assert _args(fake.cmds[1], "--rt") == ["0B31:2026101708040101", "0B31:2026101708040102"]


def test_auto_source(monkeypatch, tmp_path):
    assert cli.resolve_source("netkeiba", None) == "netkeiba"
    assert cli.resolve_source("auto", str(tmp_path / "none.exe")) == "netkeiba"
    py = tmp_path / "python.exe"
    py.write_text("", encoding="utf-8")
    monkeypatch.setattr(jvlink.sys, "platform", "win32")
    assert cli.resolve_source("auto", str(py)) == "jv"


def _odds_rec(code, length, entries, key="2026101705040611", announced="10162108"):
    """速報オッズのレコード(確定オッズと同じ形)。entries は {位置: 文字列}。"""
    s = list(" " * length)
    for pos, v in {0: code, 2: "1", 3: "20261016", 11: key, 27: announced, **entries}.items():
        s[pos:pos + len(v)] = v
    return "".join(s)


def _write_rt(raw, spec, recs):
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"RT_{spec}.txt").write_text("".join(f"x\t{r}\n" for r in recs), encoding="utf-8")


def _entry_race(conn, race_id="202605040611", date="2026-10-17"):
    conn.execute("INSERT INTO entry_races (race_id, date) VALUES (?, ?)", (race_id, date))


def test_race_key_needs_entries():
    conn = db.connect(":memory:")
    with pytest.raises(ValueError):
        jvlink.race_key(conn, "202605040611")
    _entry_race(conn)
    assert jvlink.race_key(conn, "202605040611") == "2026101705040611"


def test_read_bet_odds_maps_kinds(tmp_path):
    _write_rt(tmp_path, "0B31", [_odds_rec("O1", 960, {43: "01013306", 51: "02048312", 267: "010039006007"})])
    _write_rt(tmp_path, "0B33", [_odds_rec("O3", 2652, {40: "0102006750069706"})])
    _write_rt(tmp_path, "0B36", [_odds_rec("O6", 83283, {40: "01020300100620282", 57: "020113" + "0000000" + "2184"}),
                                 _odds_rec("O6", 83283, {40: "01020300000990001"}, key="2026101705040612")])
    odds, announced = jvlink.read_bet_odds(tmp_path, "2026101705040611")
    assert announced == "2026-10-16T21:08:00"
    assert odds["単勝"] == {(1,): 13.3, (2,): 48.3}
    assert odds["複勝"] == {(1,): 3.9}  # 下限
    assert odds["ワイド"] == {(1, 2): 67.5}
    assert odds["三連単"] == {(1, 2, 3): 1006.2}  # 票のない組(0000000)と、別のレースは入らない
    assert "馬連" not in odds


def test_fetch_bet_odds_opens_each_kind_once(tmp_path, fake):
    conn = db.connect(":memory:")
    _entry_race(conn)
    jvlink.fetch_bet_odds(conn, "202605040611", raw_root=tmp_path)
    assert _args(fake.cmds[0], "--rt") == [f"0B3{i}:2026101705040611" for i in range(1, 7)]


def test_snapshot_odds_records_announced_time(tmp_path, monkeypatch):
    conn = db.connect(":memory:")
    _entry_race(conn)
    _entry_race(conn, "202605040612")

    def export(out, from_ymd, specs, option=1, rt=(), python32=None):
        assert rt == ["0B31:2026101705040611", "0B31:2026101705040612"]
        _write_rt(out, "0B31", [_odds_rec("O1", 960, {43: "01013306", 51: "02----00"}),
                                _odds_rec("O1", 960, {}, key="2026101705040612")])  # 12R はまだオッズがない

    monkeypatch.setattr(jvlink, "export", export)
    now = dt.datetime(2026, 10, 16, 21, 10)
    assert jvlink.snapshot_odds(conn, "2026-10-17", raw_root=tmp_path, now=now) == 1
    assert conn.execute("SELECT race_id, number, snapped_at, official_at, win_odds, popularity FROM odds_snapshots").fetchall() == [
        ("202605040611", 1, "2026-10-16T21:10:00", "2026-10-16T21:08:00", 13.3, 6)]


def test_gui_fetch_bet_odds_uses_jv_when_available(monkeypatch):
    from keibaai import gui

    monkeypatch.setattr(jvlink, "available", lambda *a: True)
    monkeypatch.setattr(jvlink, "fetch_bet_odds", lambda conn, race_id: {"単勝": {(1,): 2.0}})
    assert gui.fetch_bet_odds(None, "r1", conn=object()) == {"単勝": {(1,): 2.0}}



def test_fetch_entries_odds_only_for_given_races(tmp_path, fake, monkeypatch):
    from types import SimpleNamespace as NS

    monkeypatch.setattr("keibaai.jventries.race_keys", lambda raw, date: ["2026101708040101", "2026101708040102"])
    monkeypatch.setattr("keibaai.jventries.read_entries",
                        lambda raw, date: [NS(race=NS(race_id="202608040101")), NS(race=NS(race_id="202608040102"))])
    monkeypatch.setattr("keibaai.jventries.import_entries", lambda conn, raw, date, at: 2)
    jvlink.fetch_entries(db.connect(":memory:"), "2026-10-17", raw_root=tmp_path, race_ids=["202608040102"])
    assert _args(fake.cmds[1], "--rt") == ["0B31:2026101708040102"]


def test_results_and_entries_are_jv_only(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):  # netkeiba は選べない(argparse のエラー)
        cli.main(["collect", "--from", "2026-10-10", "--to", "2026-10-11", "--source", "netkeiba"])
    monkeypatch.setattr(jvlink, "available", lambda *a: False)
    with pytest.raises(SystemExit, match="JV-Link"):
        cli.main(["entries", "--date", "2026-10-17", "--db", str(tmp_path / "k.db")])


def test_odds_history_stores_each_announcement_and_skips_done_races(tmp_path, monkeypatch):
    """時系列オッズは1行が1回の発表。発売前(オッズ0)と別レースは入れず、取り込み済みのレースは取り直さない。"""
    conn = db.connect(":memory:")
    conn.execute("INSERT INTO races (race_id, date) VALUES ('202605040611', '2026-10-17')")  # 終わったレースは成績の日付でキーを作る
    _entry_race(conn, "202605040612")
    calls = []

    def export(out, from_ymd, specs, option=1, rt=(), python32=None):
        calls.append(list(rt))
        _write_rt(out, "0B41", [
            _odds_rec("O1", 960, {43: "01000000", 51: "02000000"}, announced="10161837"),  # 発売前: オッズ0
            _odds_rec("O1", 960, {43: "01013306", 51: "02048312"}, announced="10170509"),
            _odds_rec("O1", 960, {43: "01011206", 51: "02061005"}, announced="10171400"),
            _odds_rec("O1", 960, {43: "01099999"}, key="2026101705040699"),  # 頼んでいないレース
        ])

    monkeypatch.setattr(jvlink, "export", export)
    assert jvlink.import_odds_history(conn, ["202605040611"], raw_root=tmp_path) == 1
    assert calls == [["0B41:2026101705040611"]]
    rows = conn.execute("SELECT race_id, number, snapped_at, official_at, win_odds, popularity, source FROM odds_snapshots "
                        "ORDER BY snapped_at, number").fetchall()
    assert rows == [("202605040611", 1, "2026-10-17T05:09:00", "2026-10-17T05:09:00", 13.3, 6, "ts"),
                    ("202605040611", 2, "2026-10-17T05:09:00", "2026-10-17T05:09:00", 48.3, 12, "ts"),
                    ("202605040611", 1, "2026-10-17T14:00:00", "2026-10-17T14:00:00", 11.2, 6, "ts"),
                    ("202605040611", 2, "2026-10-17T14:00:00", "2026-10-17T14:00:00", 61.0, 5, "ts")]
    # もう一度やっても、取り込み済みのレースは JV-Link を開かない
    assert jvlink.import_odds_history(conn, ["202605040611"], raw_root=tmp_path) == 0
    assert len(calls) == 1


def test_odds_history_runs_in_batches(tmp_path, monkeypatch):
    conn = db.connect(":memory:")
    ids = [f"2026050406{r:02d}" for r in range(1, 6)]
    for rid in ids:
        _entry_race(conn, rid)
    calls = []

    def export(out, from_ymd, specs, option=1, rt=(), python32=None):
        calls.append(len(rt))
        _write_rt(out, "0B41", [])

    monkeypatch.setattr(jvlink, "export", export)
    jvlink.import_odds_history(conn, ids, raw_root=tmp_path, batch=2)
    assert calls == [2, 2, 1]


def test_fetch_training_starts_before_latest_date_and_imports(tmp_path, monkeypatch):
    from tests.test_jvtraining import HC

    calls = []

    def export(out, from_ymd, specs, option=1, rt=(), python32=None):
        calls.append((out, from_ymd, specs, option))
        out.mkdir(parents=True, exist_ok=True)
        (out / "SLOP.txt").write_text(f"a.jvd\t{HC}", encoding="utf-8")

    monkeypatch.setattr(jvlink, "export", export)
    conn = db.connect(":memory:")
    conn.execute("INSERT INTO jv_training (horse_id, date, time, kind) VALUES ('x', '2026-10-30', '0600', 'slope')")
    assert jvlink.fetch_training(conn, tmp_path, today=dt.date(2026, 11, 1)) == 1
    assert calls == [(tmp_path / "training", "20261016", ["SLOP", "WOOD"], 1)]  # 最新 10/30 の14日前から
    assert conn.execute("SELECT COUNT(*) FROM jv_training").fetchone()[0] == 2


# ---- 取得の排他(fetch_lock)と、一時的なファイルロックへの再試行 ----

def test_fetch_lock_runs_one_at_a_time(tmp_path, monkeypatch):
    """同時に呼んでも、片方が終わるまで他方は入らない。両方とも完走する。"""
    monkeypatch.setattr(jvlink, "LOCK_POLL", 0.02)
    order = []

    def run(name, hold):
        with jvlink.fetch_lock(tmp_path):
            order.append(f"{name}-in")
            time.sleep(hold)
            order.append(f"{name}-out")

    a = threading.Thread(target=run, args=("a", 0.3))
    a.start()
    for _ in range(100):  # a が先に入ったことを確かめてから b を呼ぶ
        if order:
            break
        time.sleep(0.01)
    b = threading.Thread(target=run, args=("b", 0))
    b.start()
    a.join()
    b.join()
    assert order == ["a-in", "a-out", "b-in", "b-out"]


def test_fetch_lock_keeps_file_contents_unmixed(tmp_path, monkeypatch):
    """同じファイルを書いて読み直す処理を同時に呼んでも、内容が混ざらない。"""
    monkeypatch.setattr(jvlink, "LOCK_POLL", 0.02)
    shared = tmp_path / "shared.txt"
    seen = {}

    def run(name):
        with jvlink.fetch_lock(tmp_path):
            shared.write_text(name, encoding="utf-8")
            time.sleep(0.1)
            seen[name] = shared.read_text(encoding="utf-8")

    threads = [threading.Thread(target=run, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"a": "a", "b": "b"}


def test_fetch_lock_allows_nesting_in_same_thread(tmp_path):
    """同じスレッドからの入れ子は止まらない(取得の中で取得を呼んでもよい)。"""
    with jvlink.fetch_lock(tmp_path):
        with jvlink.fetch_lock(tmp_path):
            pass


def test_fetch_lock_waits_for_other_process_and_survives_its_death(tmp_path, monkeypatch):
    """ほかのプロセスが持っている間は待ち、持っていたプロセスが死ねば OS が解放するので入れる。"""
    monkeypatch.setattr(jvlink, "LOCK_POLL", 0.05)
    root = Path(__file__).resolve().parent.parent
    code = ("import sys, time\n"
            "from keibaai import jvlink\n"
            "with jvlink.fetch_lock(sys.argv[1]):\n"
            "    print('locked', flush=True)\n"
            "    time.sleep(60)\n")
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert child.stdout.readline().strip() == "locked"
        got = threading.Event()

        def wait_for_lock():
            with jvlink.fetch_lock(tmp_path):
                got.set()

        t = threading.Thread(target=wait_for_lock)
        t.start()
        assert not got.wait(0.8)  # 持たれている間は、エラーにならず待つ
        child.kill()
        assert got.wait(10)  # プロセスが死ねば、ロックは残らない
        t.join()
    finally:
        child.kill()
        child.wait()


def test_fetch_entries_waits_while_lock_is_held(tmp_path, fake, monkeypatch):
    """取得の関数も fetch_lock で囲まれている(2026-11-22 の競合の再発防止)。"""
    monkeypatch.setattr(jvlink, "LOCK_POLL", 0.02)
    monkeypatch.setattr("keibaai.jventries.race_keys", lambda raw, date: [])
    monkeypatch.setattr("keibaai.jventries.read_entries", lambda raw, date: [])
    monkeypatch.setattr("keibaai.jventries.import_entries", lambda conn, raw, date, at: 0)
    done = threading.Event()

    def call():
        jvlink.fetch_entries(db.connect(":memory:"), "2026-10-17", raw_root=tmp_path)
        done.set()

    with jvlink.fetch_lock(tmp_path):
        t = threading.Thread(target=call)
        t.start()
        assert not done.wait(0.5)  # ほかの取得が終わるまで始まらない
    t.join(timeout=10)
    assert done.is_set()


def test_concurrent_fetch_results_do_not_mix(tmp_path, monkeypatch):
    """成績の取得を同時に呼んでも、書き出しと取り込みの組が混ざらず、両方完走する。"""
    monkeypatch.setattr(jvlink.jvexport, "RT_INTERVAL", 0)
    monkeypatch.setattr(jvlink, "LOCK_POLL", 0.02)

    def run(cmd, **kw):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "RACE.txt").write_text(threading.current_thread().name, encoding="utf-8")
        time.sleep(0.2)  # 書き出しに時間がかかっている間に、もう片方が割り込まないことを見る
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(jvlink.subprocess, "run", run)
    seen = {}

    def import_races(conn, raw, since, refresh_since=None):
        seen[threading.current_thread().name] = (raw / "RACE.txt").read_text(encoding="utf-8")
        return (1, 1)

    monkeypatch.setattr("keibaai.jvrace.import_races", import_races)

    def call():
        jvlink.fetch_results(db.connect(":memory:"), dt.date(2026, 10, 1), dt.date(2026, 10, 2),
                             tmp_path, today=dt.date(2026, 10, 15))

    threads = [threading.Thread(target=call, name=n) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"a": "a", "b": "b"}


def _flaky_unlink(monkeypatch, target, fails):
    """target の unlink だけ、fails 回まで PermissionError にする。失敗した回数を返す。"""
    real_unlink = Path.unlink
    calls = {"n": 0}

    def flaky(self, missing_ok=False):
        if self == target and calls["n"] < fails:
            calls["n"] += 1
            raise PermissionError(32, "プロセスはファイルにアクセスできません")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky)
    return calls


def test_unlink_retries_until_unlocked(tmp_path, monkeypatch):
    target = tmp_path / "RT_0B31.txt"
    target.write_text("old", encoding="utf-8")
    calls = _flaky_unlink(monkeypatch, target, fails=2)
    monkeypatch.setattr(jvlink, "LOCK_RETRY_INTERVAL", 0)
    jvlink._unlink(target)
    assert calls["n"] == 2 and not target.exists()


def test_unlink_gives_up_after_retries(tmp_path, monkeypatch):
    target = tmp_path / "RT_0B31.txt"
    target.write_text("old", encoding="utf-8")
    calls = _flaky_unlink(monkeypatch, target, fails=100)
    monkeypatch.setattr(jvlink, "LOCK_RETRY", 3)
    monkeypatch.setattr(jvlink, "LOCK_RETRY_INTERVAL", 0)
    with pytest.raises(PermissionError):
        jvlink._unlink(target)
    assert calls["n"] == 3 and target.exists()


def test_export_retries_clearing_old_files(tmp_path, fake, monkeypatch):
    (tmp_path / "RACE.txt").write_text("old", encoding="utf-8")
    calls = _flaky_unlink(monkeypatch, tmp_path / "RACE.txt", fails=1)
    monkeypatch.setattr(jvlink, "LOCK_RETRY_INTERVAL", 0)
    jvlink.export(tmp_path, "20261010", ["RACE"], python32="py32")
    assert calls["n"] == 1


def test_rt_records_retries_read(tmp_path, monkeypatch):
    path = tmp_path / "RT_0B31.txt"
    path.write_text("x\tREC\n", encoding="utf-8")
    real_open = Path.open
    calls = {"n": 0}

    def flaky(self, *args, **kw):
        if self == path and calls["n"] < 1:
            calls["n"] += 1
            raise PermissionError(32, "プロセスはファイルにアクセスできません")
        return real_open(self, *args, **kw)

    monkeypatch.setattr(Path, "open", flaky)
    monkeypatch.setattr(jvlink, "LOCK_RETRY_INTERVAL", 0)
    assert list(jvlink._rt_records(path)) == ["REC"]
    assert calls["n"] == 1
