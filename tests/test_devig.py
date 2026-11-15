import numpy as np
import pandas as pd
import pytest

from keibaai.devig import market_probs, proportional, shin
from keibaai.devig_trials import compare, load_market
from keibaai import db


def test_shin_sums_to_one_and_shifts_toward_favorite():
    odds = np.array([1.5, 3.0, 6.0, 12.0, 30.0, 80.0])  # 逆数の合計は約1.25(JRAの単勝の控除に近い)
    p, z = shin(odds)
    assert p.sum() == pytest.approx(1.0)
    assert 0.0 < z < 1.0
    prop = proportional(odds)
    assert p[0] > prop[0]  # 人気馬は比例より厚く
    assert p[-1] < prop[-1]  # 人気薄は比例より薄く(控除が厚く乗っている分を剥がす)


def test_shin_matches_proportional_without_margin():
    odds = np.array([2.0, 4.0, 4.0])  # 逆数の合計がちょうど1(控除なし)
    p, z = shin(odds)
    assert z == 0.0 and p == pytest.approx(proportional(odds))


def test_shin_equal_odds_gives_equal_probs():
    p, z = shin(np.array([2.0, 2.0, 2.0]))  # 逆数の合計 1.5 で、Shin の計算を通る
    assert z > 0 and p == pytest.approx([1 / 3] * 3)


def test_shin_satisfies_defining_equation():
    odds = np.array([1.5, 3.0, 6.0, 12.0, 30.0, 80.0])
    p, z = shin(odds)
    inv = 1.0 / odds
    # Shin の定義式: π_i^2 / S = z p_i + (1-z) p_i^2 (π はオッズの逆数、S はその合計)
    assert inv**2 / inv.sum() == pytest.approx(z * p + (1 - z) * p**2, rel=1e-6)


def test_market_probs_groups_by_race_and_marks_incomplete_nan():
    df = pd.DataFrame({"race_id": ["a"] * 3 + ["b"] * 2, "win_odds": [2.0, 4.0, 4.0, 2.0, None]})
    prop = market_probs(df, "proportional")
    assert prop.iloc[:3].tolist() == pytest.approx([0.5, 0.25, 0.25])
    assert prop.iloc[3:].isna().all()  # オッズの欠けたレースは丸ごと NaN
    sh = market_probs(df, "shin")
    assert sh.iloc[:3].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        market_probs(df, "unknown")


def test_market_probs_with_duplicated_index():
    df = pd.DataFrame({"race_id": ["a", "a", "b", "b"], "win_odds": [2.0, 2.0, 3.0, 3.0]}, index=[0, 0, 1, 1])
    prop = market_probs(df, "proportional")
    assert prop.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])


def _results_db(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    with conn:
        conn.execute("INSERT INTO races (race_id, date) VALUES ('r1', '2024-06-01'), ('r2', '2024-06-01')")
        conn.executemany("INSERT INTO results (race_id, number, finish, win_odds) VALUES (?,?,?,?)",
                         [("r1", 1, 1, 2.0), ("r1", 2, 2, 4.0), ("r1", 3, 3, 4.0),
                          ("r2", 1, 2, 2.0), ("r2", 2, 1, None)])  # r2 は勝ち馬のオッズが欠けるので使わない
    return conn


def test_load_market_and_compare(tmp_path):
    conn = _results_db(tmp_path)
    df = load_market(conn, "2024-01-01", "2025-01-01")
    assert set(df["race_id"]) == {"r1"}
    out = compare(df, reps=100)
    assert list(out["方法"]) == ["比例", "Shin", "差 (Shin − 比例)"]
    assert out.loc[0, "レース数"] == 1
