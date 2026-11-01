import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from keibaai import db
from keibaai.entries import apply_odds, parse_odds_json, parse_shutuba
from keibaai.predict import fetch_entries, predict_day, snapshot_odds

from test_model import synthetic_db

SHUTUBA = (Path(__file__).parent / "fixtures" / "shutuba_sample.html").read_text(encoding="utf-8")
ODDS = json.dumps({"status": "result", "data": {"official_datetime": "2024-11-24 15:52:26", "odds": {
    "1": {"01": ["15.8", "0.0", "6"], "03": ["2.3", "0.0", "1"]}, "2": {"01": ["3.3", "5.5", "7"]}}}})


def test_parse_shutuba_race_and_rows():
    p = parse_shutuba(SHUTUBA, "202405050812")
    r = p.race
    assert r.name == "ジャパンC(G1)" and r.date == "2024-11-24" and (r.venue, r.round) == ("東京", 12)
    assert (r.course_type, r.distance, r.direction) == ("芝", 2400, "左")
    assert (r.weather, r.ground, r.start_time) == ("晴", "良", "15:40")
    assert [e.number for e in p.entries] == [1, 3]  # 「登録/グループ」の行は馬ではない
    e = p.entries[1]
    assert (e.horse_id, e.horse_name, e.sex_age, e.weight_carried) == ("2019105219", "ドウデュース", "牡5", 58.0)
    assert (e.jockey, e.jockey_id, e.trainer, e.trainer_id) == ("武豊", "01088", "友道康夫", "01088")
    assert (e.body_weight, e.body_weight_diff, e.popularity) == (504, -6, None)  # 発売前の「**」は空
    first = p.entries[0]
    assert (first.body_weight, first.body_weight_diff) == (496, None)  # 「前計不」


def test_parse_odds_and_apply():
    odds = parse_odds_json(ODDS)
    assert odds == {1: (15.8, 6), 3: (2.3, 1)}
    p = parse_shutuba(SHUTUBA, "202405050812")
    apply_odds(p, odds)
    assert [(e.win_odds, e.popularity) for e in p.entries] == [(15.8, 6), (2.3, 1)]


@pytest.mark.parametrize("text", ["", "not json", "{}", '{"data": {"odds": {}}}'])
def test_parse_odds_bad_input_gives_empty(text):
    assert parse_odds_json(text) == {}


def test_missing_table_raises():
    with pytest.raises(ValueError):
        parse_shutuba("<html></html>", "202405050812")


def test_save_entries_replaces_and_stays_out_of_results():
    conn = db.connect(":memory:")
    p = parse_shutuba(SHUTUBA, "202405050812")
    db.save_entries(conn, p, "2024-11-23T10:00:00")
    apply_odds(p, parse_odds_json(ODDS))
    db.save_entries(conn, p, "2024-11-23T11:00:00")  # オッズを取り直した
    assert conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 2
    assert conn.execute("SELECT win_odds FROM entries WHERE number = 3").fetchone()[0] == 2.3
    assert conn.execute("SELECT fetched_at FROM entry_races").fetchone()[0] == "2024-11-23T11:00:00"
    assert not db.has_race(conn, "202405050812")  # 結果の取得を邪魔しない


class FakeFetcher:
    def __init__(self, ids, odds=ODDS):
        self.ids, self.odds = ids, odds

    def race_ids_on(self, date):
        return self.ids

    def shutuba_html(self, race_id):
        if race_id == "bad":
            raise ValueError("出馬表のないページです")
        return SHUTUBA

    def odds_json(self, race_id):
        if self.odds is None:
            raise RuntimeError("down")
        return self.odds


def test_fetch_entries_skips_bad_race_and_survives_odds_failure():
    conn = db.connect(":memory:")
    n = fetch_entries(conn, FakeFetcher(["202405050812", "bad"]), "2024-11-24", now=dt.datetime(2024, 11, 23))
    assert n == 1
    n = fetch_entries(conn, FakeFetcher(["202405050812"], odds=None), "2024-11-24", now=dt.datetime(2024, 11, 23))
    assert n == 1 and conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 2


def test_fetch_entries_only_given_races():
    conn = db.connect(":memory:")
    fetcher = FakeFetcher(["bad", "202405050812"])
    fetcher.race_ids_on = None  # レース一覧は読みにいかない
    n = fetch_entries(conn, fetcher, "2024-11-24", now=dt.datetime(2024, 11, 23), race_ids=["202405050812"])
    assert n == 1


def _entries_for_future(conn, date, odds=True):
    """合成データの馬・騎手で、未来の日の出馬表を作る。"""
    conn.execute("INSERT INTO entry_races (race_id, name, date, venue, round, course_type, distance, ground) "
                 "VALUES ('202609990101', '3歳未勝利', ?, '東京', 1, '芝', 1600, '良')", (date,))
    for i in range(10):
        conn.execute("INSERT INTO entries (race_id, number, bracket, horse_id, horse_name, sex_age, weight_carried, "
                     "jockey_id, trainer_id, body_weight, body_weight_diff, win_odds, popularity) "
                     "VALUES ('202609990101', ?, ?, ?, ?, '牡4', 56, ?, ?, 480, 0, ?, ?)",
                     (i + 1, 1 + i // 2, f"h{i}", f"馬{i}", f"j{i}", f"t{i % 7}",
                      (2.0 + i) if odds else None, i + 1 if odds else None))
    conn.commit()


def test_predict_day_gives_probabilities_and_expected_value():
    conn = synthetic_db(n_days=40)
    _entries_for_future(conn, "2030-01-05")
    out = predict_day(conn, "2030-01-05", rounds=30)
    assert len(out) == 10
    assert np.isclose(out["p_win_ability"].sum(), 1.0) and np.isclose(out["p_win_market"].sum(), 1.0)
    assert np.allclose(out["ev_market"], out["p_win_market"] * out["win_odds"])
    assert out["ev_ability"].notna().all()
    for col in ("p_win_quality", "p_win_human", "p_win_track"):  # 要素ごとの勝率
        assert np.isclose(out[col].sum(), 1.0)
    assert (out["venue"] + out["round"].astype(int).astype(str) + "R").iloc[0] == "東京1R"  # 画面で使う形
    flow = out.copy().attrs["flow"]  # 発走前の展開予想(weekly が予想と一緒に保存する)
    assert len(flow) == 1 and flow["pace"].iloc[0] in ("スロー", "平均", "ハイ", "不明")


def test_predict_day_without_odds_has_no_market_columns_values():
    conn = synthetic_db(n_days=40)
    _entries_for_future(conn, "2030-01-05", odds=False)
    out = predict_day(conn, "2030-01-05", rounds=30)
    assert out["p_win_market"].isna().all() and out["ev_market"].isna().all()
    assert np.isclose(out["p_win_ability"].sum(), 1.0)


def test_predict_day_uses_only_earlier_results():
    conn = synthetic_db(n_days=40)
    _entries_for_future(conn, "2025-01-04")  # 合成データの最初の日と同じ日。それより前の結果は何もない
    with pytest.raises(ValueError):
        predict_day(conn, "2025-01-05", rounds=30)


def test_predict_day_without_entries():
    with pytest.raises(ValueError):
        predict_day(synthetic_db(n_days=5), "2030-01-05")


def test_fetch_entries_skips_race_without_horses_yet(capsys):
    """馬番が決まる前のページ(馬が0頭)は保存しない。保存すると「取得済み」に見えてしまう。"""
    empty = "<html><body><table><tr><th>馬番</th><th>馬名</th></tr></table></body></html>"

    class F(FakeFetcher):
        def shutuba_html(self, race_id):
            return SHUTUBA if race_id == "202405050812" else empty

    conn = db.connect(":memory:")
    n = fetch_entries(conn, F(["202405050812", "202405050811"]), "2024-11-24", now=dt.datetime(2024, 11, 23))
    assert n == 1 and conn.execute("SELECT COUNT(*) FROM entry_races").fetchone()[0] == 1
    assert "確定していません" in capsys.readouterr().out


def test_snapshot_odds_appends_with_time_and_skips_empty(capsys):
    """オッズを取るたびに時刻つきで足していく(上書きしない)。発売前で空のレースは飛ばす。"""
    conn = db.connect(":memory:")
    fetch_entries(conn, FakeFetcher(["202405050812"]), "2024-11-24", now=dt.datetime(2024, 11, 23))
    f = FakeFetcher(["202405050812"])
    assert snapshot_odds(conn, f, "2024-11-24", now=dt.datetime(2024, 11, 24, 9, 0)) == 1
    assert snapshot_odds(conn, f, "2024-11-24", now=dt.datetime(2024, 11, 24, 15, 0)) == 1
    rows = conn.execute("SELECT snapped_at, number, win_odds, popularity, official_at FROM odds_snapshots ORDER BY snapped_at, number").fetchall()
    assert len(rows) == 4 and rows[0] == ("2024-11-24T09:00:00", 1, 15.8, 6, "2024-11-24 15:52:26")
    assert rows[3][:4] == ("2024-11-24T15:00:00", 3, 2.3, 1)

    class Empty(FakeFetcher):
        def odds_json(self, race_id, odds_type=1):
            return json.dumps({"status": "yet", "data": []})

    assert snapshot_odds(conn, Empty([]), "2024-11-24", now=dt.datetime(2024, 11, 24, 16, 0)) == 0
    assert "オッズがまだ" in capsys.readouterr().out
