import numpy as np
import pandas as pd
import pytest

from keibaai import backtest as bt
from keibaai.features import build_features, load_results

from test_model import synthetic_db


@pytest.fixture(scope="module")
def test_df():
    df = build_features(load_results(synthetic_db()))
    return bt.predict_test(df, "2025-09-01")


def toy():
    # 2レース。レース1は馬1が勝ち、レース2は馬2が勝つ
    return pd.DataFrame({
        "race_id": ["a", "a", "b", "b"], "date": pd.to_datetime(["2025-01-04"] * 2 + ["2025-01-11"] * 2),
        "number": [1, 2, 1, 2], "win": [1, 0, 0, 1], "win_odds": [2.0, 4.0, 3.0, 5.0],
        "popularity": [1, 2, 1, 2], "p_win": [0.6, 0.4, 0.7, 0.3],
    })


def test_settle_and_summary_by_hand():
    t = toy()
    s = bt.settle(bt.select_bets(t, "top1"))  # 馬1(2倍・的中) と 馬1(3倍・外れ)
    assert s["payout"].tolist() == [200, 0] and s["balance"].tolist() == [100, 0]
    m = bt.summarize("top1", s)
    assert (m.n_bets, m.n_hits, m.stake, m.payout, m.profit) == (2, 1, 200, 200, 0)
    assert m.roi == 100.0 and m.max_drawdown == 100


def test_ev_rule_and_odds_cap():
    t = toy()  # EV: 1.2, 1.6, 2.1, 1.5
    assert len(bt.select_bets(t, "ev", ev_min=1.5)) == 3
    assert len(bt.select_bets(t, "ev", ev_min=1.5, max_odds=4.0)) == 2
    assert len(bt.select_bets(t, "all")) == 4 and len(bt.select_bets(t, "favorite")) == 2
    with pytest.raises(ValueError):
        bt.select_bets(t, "nope")


def test_missing_odds_are_not_bought():
    t = toy()
    t.loc[0, "win_odds"] = np.nan
    assert len(bt.select_bets(t, "all")) == 3


def test_drawdown_and_empty():
    assert bt.max_drawdown(pd.Series([100, -200, -100, 300])) == 300  # 最高の100から、いちばん低い-200まで
    empty = bt.settle(bt.select_bets(toy(), "ev", ev_min=99))
    m = bt.summarize("ev", empty)
    assert m.n_bets == 0 and m.profit == 0 and m.max_drawdown == 0


def test_run_rules_on_model_output(test_df):
    table, settled = bt.run_rules(test_df, bt.DEFAULT_RULES)
    assert len(table) == len(bt.DEFAULT_RULES) and set(settled) == set(table["rule"])
    # 全頭買いは約 1 - 控除率(合成データのオッズは控除なしの目安なので、ここでは範囲だけ見る)
    assert (table["roi_low"] <= table["roi"]).all() and (table["roi"] <= table["roi_high"]).all()
    n_races = test_df["race_id"].nunique()
    assert table.set_index("rule").loc["◎(勝率1位)", "n_bets"] == n_races
    series = bt.balance_by_date(settled["◎(勝率1位)"])
    assert series.index.is_monotonic_increasing and series.iloc[-1] == table.set_index("rule").loc["◎(勝率1位)", "profit"]


def test_min_p_drops_low_probability_horses():
    import pandas as pd
    from keibaai.backtest import select_bets

    test = pd.DataFrame({"date": ["2026-01-01"] * 2, "race_id": ["a", "a"], "number": [1, 2],
                         "p_win": [0.02, 0.4], "win_odds": [100.0, 3.0], "popularity": [9, 1]})
    assert len(select_bets(test, "ev", 1.0)) == 2
    assert select_bets(test, "ev", 1.0, min_p=0.03)["number"].tolist() == [2]
