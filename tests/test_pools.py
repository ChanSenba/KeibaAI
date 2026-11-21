import itertools

import numpy as np
import pandas as pd

from keibaai import db, jvodds, pools, predict
from keibaai.jvlink import read_pool_odds
from test_entries import _entries_for_future
from test_jvodds import odds_rec
from test_model import synthetic_db

DAY = "2030-01-05"


def _fill_final(oconn, race_id, odds_by_num):
    """単勝オッズから作った架空の確定オッズ(複勝・馬単・三連単)を jravan_odds.db の形で入れる。"""
    nums = sorted(odds_by_num)
    p = {n: 1 / odds_by_num[n] for n in nums}
    oconn.executemany("INSERT INTO odds_win VALUES (?,?,?,?)", [(race_id, n, odds_by_num[n], 1) for n in nums])
    oconn.executemany("INSERT INTO odds_place VALUES (?,?,?,?,?)",
                      [(race_id, n, 1 + 0.3 * odds_by_num[n], 1 + 0.6 * odds_by_num[n], 1) for n in nums])
    oconn.executemany("INSERT INTO odds_exacta VALUES (?,?,?,?,?)",
                      [(race_id, a, b, 0.75 / (p[a] * p[b]), 1) for a, b in itertools.permutations(nums, 2)])
    oconn.executemany("INSERT INTO odds_trifecta VALUES (?,?,?,?,?,?)",
                      [(race_id, a, b, c, 0.7 / (p[a] * p[b] * p[c]), 1) for a, b, c in itertools.permutations(nums, 3)])


def _snapshot(conn, race_id, odds_by_num, stamp="2030-01-04T21:00:00", kinds=("単勝", "複勝", "馬単", "三連単"),
              place_max=True):
    nums = sorted(odds_by_num)
    p = {n: 1 / odds_by_num[n] for n in nums}
    table = {"単勝": {(n,): odds_by_num[n] for n in nums},
             "複勝": {(n,): 1 + 0.3 * odds_by_num[n] for n in nums},
             "馬単": {(a, b): 0.75 / (p[a] * p[b]) for a, b in itertools.permutations(nums, 2)},
             "三連単": {(a, b, c): 0.7 / (p[a] * p[b] * p[c]) for a, b, c in itertools.permutations(nums, 3)}}
    hi = {"複勝": {(n,): 1 + 0.6 * odds_by_num[n] for n in nums}} if place_max else None
    db.save_bet_odds_snapshot(conn, race_id, {k: table[k] for k in kinds}, stamp, hi)


def test_final_and_snapshot_give_the_same_features(tmp_path):
    odds = {1: 2.0, 2: 4.0, 3: 9.0, 4: 20.0}
    oconn = jvodds.connect(tmp_path / "jravan_odds.db")
    _fill_final(oconn, "202605050511", odds)
    oconn.commit()
    conn = db.connect(":memory:")
    _snapshot(conn, "202605050511", odds, stamp="2030-01-04T21:00:00")
    _snapshot(conn, "202605050511", {1: 3.0, 2: 3.0, 3: 9.0, 4: 20.0}, stamp="2030-01-04T08:00:00")  # 古い記録は使わない
    # 買い方の画面が残す記録(複勝の上限がない)は、新しくても使わない
    _snapshot(conn, "202605050511", {1: 5.0, 2: 3.0, 3: 9.0, 4: 2.0}, stamp="2030-01-05T09:00:00", place_max=False)
    # 単勝の市場は、券種と同じ時点のオッズ(出馬表の単勝オッズは使わない)
    df = pd.DataFrame({"race_id": "202605050511", "number": [1, 2, 3, 4], "win_odds": [2.1, 3.9, 9.5, 18.0]})
    a = pools.add_pool_features(df, pools.final_shares(tmp_path / "jravan_odds.db"))
    b = pools.add_pool_features(df, pools.snapshot_shares(conn, ["202605050511"]))
    assert pools.has_pools(a).all()
    pd.testing.assert_frame_equal(a, b)
    # 単勝の市場の勝率との対数の差。複勝は下限と上限の幾何平均の逆数をレース内で合計1にしたもの
    s = 1 / np.sqrt((1 + 0.3 * np.array([2.0, 4, 9, 20])) * (1 + 0.6 * np.array([2.0, 4, 9, 20])))
    w = 1 / np.array([2.0, 4, 9, 20])
    mkt = w / w.sum()
    assert np.allclose(a["d_place"], np.log(s / s.sum()) - np.log(mkt))


def test_race_missing_a_pool_or_a_horse_has_no_features():
    conn = db.connect(":memory:")
    _snapshot(conn, "r1", {1: 2.0, 2: 4.0, 3: 6.0}, kinds=("単勝", "複勝", "馬単"))  # 三連単がない
    _snapshot(conn, "r2", {1: 2.0, 2: 4.0, 3: 6.0})
    df = pd.DataFrame({"race_id": ["r1"] * 3 + ["r2"] * 4, "number": [1, 2, 3, 1, 2, 3, 4],
                       "win_odds": [2.0, 4.0, 6.0, 2.0, 4.0, 6.0, 8.0]})
    out = pools.add_pool_features(df, pools.snapshot_shares(conn, ["r1", "r2"]))
    assert not pools.has_pools(out).any()  # r2 は4番の馬のオッズがない(取消の馬など)
    assert list(out.index) == list(df.index)


def test_read_pool_odds_keeps_place_range(tmp_path):
    key = "2026100305040111"
    o1 = odds_rec("place", {"01": "0012001501", "02": "0030004502"})
    o4 = odds_rec("exacta", {"0102": "000123001", "0201": "000456002"})
    o6 = odds_rec("trifecta", {"010203": "00012340001"})
    for spec, r in (("0B31", o1), ("0B34", o4), ("0B36", o6)):
        (tmp_path / f"RT_{spec}.txt").write_text(f"{key}\t{r}\n", encoding="utf-8")
    odds, hi = read_pool_odds(tmp_path, {key: "202605040111"})["202605040111"]
    assert odds["複勝"] == {(1,): 1.2, (2,): 3.0} and hi["複勝"] == {(1,): 1.5, (2,): 4.5}
    assert odds["馬単"] == {(1, 2): 12.3, (2, 1): 45.6} and odds["三連単"] == {(1, 2, 3): 123.4}
    assert "馬連" not in odds and "ワイド" not in odds


def test_predict_uses_pool_model_only_for_races_with_pool_odds(tmp_path, monkeypatch):
    mem = synthetic_db(n_days=40)
    _entries_for_future(mem, DAY)
    conn = db.connect(tmp_path / "keiba.db")
    mem.backup(conn)
    oconn = jvodds.connect(tmp_path / "jravan_odds.db")
    for rid, g in pd.read_sql_query("SELECT race_id, number, win_odds FROM results", conn).groupby("race_id"):
        _fill_final(oconn, rid, dict(zip(g["number"], g["win_odds"])))
    oconn.commit()
    monkeypatch.setattr(predict, "POOL_MIN_RACES", 10)
    base = predict.predict_day(conn, DAY)
    models = predict.models_for_day(conn, DAY, None)  # 保存したモデル(学習はしない)
    assert "residual_pools" in models
    odds = dict(zip(base["number"], base["win_odds"]))
    _snapshot(conn, "202609990101", odds)
    conn.commit()
    with_pools = predict.predict_day(conn, DAY)  # 券種のオッズを取ったら、予想を出し直す
    assert np.isclose(with_pools["p_win_market"].sum(), 1.0)
    assert not np.allclose(with_pools["p_win_market"], base["p_win_market"])
    pd.testing.assert_series_equal(with_pools["p_win_ability"], base["p_win_ability"])  # 印の勝率は変わらない


def test_partly_sold_trifecta_is_not_used():
    conn = db.connect(":memory:")
    _snapshot(conn, "r1", {1: 2.0, 2: 4.0, 3: 6.0, 4: 9.0})
    conn.execute("DELETE FROM bet_odds_snapshots WHERE kind = '三連単' AND combo = '04-03-02'")  # 発売直後で売れていない組
    assert pools.snapshot_shares(conn, ["r1"]).empty
