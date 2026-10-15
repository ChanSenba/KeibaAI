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
