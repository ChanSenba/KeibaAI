import datetime as dt
import subprocess

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
    assert _args(fake.cmds[0], "--rt") == ["0B15:20261017"] and _args(fake.cmds[0], "--from")[0] == "20261010"


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
