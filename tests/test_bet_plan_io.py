import json

import numpy as np
import pandas as pd
import pytest

from keibaai import db, gui
from keibaai.blend import Blend
from keibaai.entries import parse_bet_odds


def odds_json(kind_key: str, table: dict) -> str:
    return json.dumps({"status": "result", "data": {"official_datetime": "x", "odds": {kind_key: table}}})


def test_parse_win_and_place_from_type1():
    text = json.dumps({"status": "result", "data": {"odds": {
        "1": {"01": ["15.8", "0.0", "6"], "03": ["2.3", "0.0", "1"], "02": ["0.0", "0.0", "0"]},
        "2": {"01": ["3.3", "5.5", "7"], "03": ["1.2", "1.6", "1"]}}}})
    out = parse_bet_odds(text, 1)
    assert out["単勝"] == {(1,): 15.8, (3,): 2.3}  # 0.0 は発売なし
    assert out["複勝"] == {(1,): 3.3, (3,): 1.2}  # 下限を使う


def test_parse_combination_keys_and_comma_numbers():
    out = parse_bet_odds(odds_json("4", {"0102": ["334.6", "0.0", "50"], "1314": ["2,528.6", "0.0", "81"]}), 4)
    assert out["馬連"] == {(1, 2): 334.6, (13, 14): 2528.6}
    out = parse_bet_odds(odds_json("8", {"010203": ["2,463.0", "0.0", "515"], "141312": ["9.9", "0.0", "1"]}), 8)
    assert out["三連単"] == {(1, 2, 3): 2463.0, (14, 13, 12): 9.9}  # 着順どおり
    out = parse_bet_odds(odds_json("5", {"0102": ["87.8", "91.1", "54"]}), 5)
    assert out["ワイド"] == {(1, 2): 87.8}  # 下限
    assert list(parse_bet_odds(odds_json("6", {"0201": ["917.3", "0.0", "109"]}), 6)["馬単"]) == [(2, 1)]


def test_parse_unavailable_returns_empty():
    assert parse_bet_odds("", 4) == {} and parse_bet_odds('{"status":"x","data":[]}', 4) == {}
    assert parse_bet_odds(odds_json("4", {}), 4) == {}


class FakeFetcher:
    def __init__(self):
        self.asked = []

    def odds_json(self, race_id, odds_type=1):
        self.asked.append(odds_type)
        key = {4: "4", 5: "5", 6: "6", 7: "7", 8: "8"}.get(odds_type)
        if odds_type == 1:
            return json.dumps({"data": {"odds": {"1": {"01": ["3.0", "0.0", "1"], "02": ["4.0", "0.0", "2"]}, "2": {}}}})
        return odds_json(key, {"0102": ["6.0", "0.0", "1"]} if odds_type == 4 else {})


def test_fetch_bet_odds_asks_each_type_once_and_merges():
    f = FakeFetcher()
    seen = []
    out = gui.fetch_bet_odds(f, "r1", on_type=lambda i, n, t: seen.append(t))
    assert f.asked == [1, 4, 5, 6, 7, 8] == seen
    assert out["単勝"] == {(1,): 3.0, (2,): 4.0} and out["馬連"] == {(1, 2): 6.0}


def test_fetch_bet_odds_survives_snapshot_failure(monkeypatch):
    monkeypatch.setattr("keibaai.jvlink.available", lambda: False)  # JV-Link のある PC でも netkeiba 側の経路を試す
    out = gui.fetch_bet_odds(FakeFetcher(), "r1", conn=object())  # 保存できない conn でも、オッズ取得は止めない
    assert out["単勝"][(1,)] == 3.0


def test_fetch_bet_odds_saves_snapshot_when_conn_given(tmp_path, monkeypatch):
    monkeypatch.setattr("keibaai.jvlink.available", lambda: False)  # JV-Link のある PC でも netkeiba 側の経路を試す
    conn = db.connect(tmp_path / "t.db")
    out = gui.fetch_bet_odds(FakeFetcher(), "r1", conn=conn)
    assert out["単勝"] == {(1,): 3.0, (2,): 4.0}
    rows = conn.execute(
        "SELECT kind, combo, odds FROM bet_odds_snapshots WHERE race_id = 'r1' ORDER BY kind, combo").fetchall()
    assert rows == [("単勝", "01", 3.0), ("単勝", "02", 4.0), ("馬連", "01-02", 6.0)]
    stamps = {r[0] for r in conn.execute("SELECT snapped_at FROM bet_odds_snapshots WHERE race_id = 'r1'")}
    assert len(stamps) == 1 and stamps.pop()  # 取得時刻が入っている


def test_save_bet_odds_snapshot_appends_by_time_and_replaces_same_time(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    odds = {"単勝": {(1,): 3.0}, "三連単": {(1, 2, 3): 100.0}}
    assert db.save_bet_odds_snapshot(conn, "r1", odds, "2026-11-15T10:00:00") == 2
    assert db.save_bet_odds_snapshot(conn, "r1", {"単勝": {(1,): 2.5}}, "2026-11-15T10:00:00") == 1  # 同じ時刻は置き換え
    assert db.save_bet_odds_snapshot(conn, "r1", odds, "2026-11-15T11:00:00") == 2  # 別の時刻は追記
    assert db.save_bet_odds_snapshot(conn, "r1", {}, "2026-11-15T12:00:00") == 0  # 空は何も書かない
    rows = conn.execute(
        "SELECT snapped_at, kind, combo, odds FROM bet_odds_snapshots ORDER BY snapped_at, kind").fetchall()
    assert rows == [("2026-11-15T10:00:00", "三連単", "01-02-03", 100.0),
                    ("2026-11-15T10:00:00", "単勝", "01", 2.5),
                    ("2026-11-15T11:00:00", "三連単", "01-02-03", 100.0),
                    ("2026-11-15T11:00:00", "単勝", "01", 3.0)]


def test_bet_plan_for_mixes_probabilities_and_uses_race_odds():
    pred = pd.DataFrame({"race_id": ["r1"] * 4 + ["r2"], "number": [1, 2, 3, 4, 1],
                         "p_win_ability": [0.25, 0.25, 0.25, 0.25, 1.0],
                         "p_win_market": [0.55, 0.25, 0.15, 0.05, 1.0]})
    odds = {"単勝": {(1,): 2.4, (2,): 3.2, (3,): 4.8, (4,): 9.6}, "馬連": {(1, 2): 2.0}}  # オッズの逆数の合計は1.25(控除率20%)
    plan, mixed = gui.bet_plan_for(pred, "r1", odds, Blend(1.0, 0.0, 0, 0))  # モデルだけ。使うのはオッズ込みの勝率
    assert len(mixed) == 4 and np.isclose(mixed["p_win"].sum(), 1) and np.isclose(mixed["p_win"].iloc[0], 0.55)
    assert plan.bets and plan.bets[0].kind in ("単勝", "馬連")  # モデルの見立てでは有利
    assert list(gui.plan_table(plan.bets).columns) == ["券種", "買い目", "オッズ", "的中確率", "期待値", "金額", "的中時の払戻"]
    plan2, _ = gui.bet_plan_for(pred, "r1", odds, Blend(0.0, 1.0, 0, 0))  # オッズだけ → 控除率で期待値は1未満
    assert plan2.bets == [] and "見送り" in plan2.note


def test_bet_plan_for_falls_back_to_odds_when_market_model_is_missing():
    """出馬表を取ったときにオッズがなく、オッズ込みの勝率が出せていないレースは、いまのオッズの確率だけを使う。"""
    odds = {"単勝": {(1,): 2.4, (2,): 3.2, (3,): 4.8, (4,): 9.6}}
    for market in ([0.55, np.nan, 0.15, 0.05], None):
        pred = pd.DataFrame({"race_id": ["r1"] * 4, "number": [1, 2, 3, 4], "p_win_ability": [0.7, 0.1, 0.1, 0.1]})
        if market:
            pred["p_win_market"] = market
        plan, mixed = gui.bet_plan_for(pred, "r1", odds, Blend(1.0, 0.0, 0, 0))
        assert np.allclose(mixed["p_win"], [0.4, 0.3, 0.2, 0.1])  # 単勝オッズの逆数を合計1にしたもの
        assert plan.bets == []  # 能力の勝率(0.7)を重み1で使っていれば、1番の単勝が買い目になってしまう


def test_bet_plan_for_without_win_odds_explains():
    pred = pd.DataFrame({"race_id": ["r1"] * 3, "number": [1, 2, 3], "p_win_ability": [0.5, 0.3, 0.2]})
    with pytest.raises(ValueError, match="単勝オッズ"):
        gui.bet_plan_for(pred, "r1", {}, Blend(0.0, 1.0, 0, 0))


def test_cli_bets_runs_end_to_end(tmp_path, monkeypatch, capsys):
    from keibaai import cli, db
    from keibaai.blend import Blend
    from test_entries import _entries_for_future
    from test_model import synthetic_db

    path = tmp_path / "k.db"
    conn = synthetic_db(n_days=40)
    _entries_for_future(conn, "2030-01-05")
    disk = db.connect(path)
    conn.backup(disk)
    disk.close()
    monkeypatch.setattr(cli, "Fetcher", lambda cache_dir: object())
    monkeypatch.setattr(gui, "fetch_bet_odds", lambda f, race_id, on_type=None, conn=None: {
        "単勝": {(i + 1,): 2.0 + i for i in range(10)}, "馬連": {(1, 2): 2.0}})
    monkeypatch.setattr("keibaai.predict.blend_for_day", lambda conn, date: Blend(0.0, 1.0, 0, 0))
    cli.main(["bets", "--date", "2030-01-05", "--race-id", "202609990101", "--db", str(path)])
    out = capsys.readouterr().out
    assert "混ぜ方" in out and ("見送り" in out or "期待回収率" in out)
    with pytest.raises(SystemExit):
        cli.main(["bets", "--date", "2030-01-05", "--race-id", "nope", "--db", str(path)])
    with pytest.raises(SystemExit, match="出馬表がありません"):  # トレースバックではなくメッセージだけで終わる
        cli.main(["bets", "--date", "2031-01-05", "--race-id", "nope", "--db", str(path)])
