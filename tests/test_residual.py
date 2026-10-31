import numpy as np
import pandas as pd

from keibaai.features import add_relative, add_speed_figures
from keibaai.model import evaluate, market_logit, normalize_within_race


def _races(times, dates, horses, finishes, race_ids):
    n = len(times)
    return pd.DataFrame({"race_id": race_ids, "number": list(range(1, n + 1)), "horse_id": horses, "finish": finishes,
                         "time_sec": times, "date": pd.to_datetime(dates), "venue": "東京", "course_type": "芝",
                         "distance": 1000, "ground": "良"})


def test_speed_figure_uses_only_earlier_days_and_earlier_runs():
    # 5日ぶんの基準(勝ちタイム 60秒)のあと、6日目に x が 59秒で勝ち、7日目に x が走る
    days = [f"2025-01-0{d}" for d in range(1, 8)]
    rows = _races([60.0, 61.0] * 6 + [99.0, 61.0], [d for d in days for _ in range(2)],
                  ["a", "b"] * 5 + ["x", "y", "x", "y"], [1, 2] * 7, [f"r{d}" for d in range(1, 8) for _ in range(2)])
    rows.loc[10, "time_sec"] = 59.0
    f = add_speed_figures(rows)
    assert np.isnan(f.loc[10, "prev_fig"])  # 初めて走る馬には前走の指数がない
    assert f.loc[12, "prev_fig"] == 1.0  # 前走は基準 60秒より 1秒速い(1000m なので 1000m あたり 1秒)
    assert f.loc[12, "best_fig_3"] == 1.0  # 7日目の 99秒(その日の結果)は使わない


def test_relative_is_within_race():
    df = pd.DataFrame({"race_id": ["a", "a", "b", "b"], "top3_rate": [0.1, 0.3, 0.5, 0.5]})
    r = add_relative(df)
    assert list(r["top3_rate_rk"]) == [0.5, 1.0, 0.75, 0.75]
    assert np.isnan(r.loc[2, "top3_rate_z"])  # 全馬同じ値なら差は出さない


def test_market_logit_is_normalized_odds():
    df = pd.DataFrame({"race_id": ["a", "a"], "win_odds": [2.0, 2.0]})
    p = 1 / (1 + np.exp(-market_logit(df)))
    assert np.allclose(p, 0.5)


def test_market_model_starts_from_the_market(monkeypatch):
    from test_model import synthetic_db
    from keibaai import model
    from keibaai.features import build_features, load_results

    df = build_features(load_results(synthetic_db(n_days=40)))
    cut = str(df["date"].sort_values().iloc[len(df) * 3 // 4].date())
    # 木がほとんど何も足さなければ、市場の勝率そのものになる
    monkeypatch.setattr(model, "RESIDUAL_PARAMS", dict(model.RESIDUAL_PARAMS, learning_rate=1e-9))
    monkeypatch.setattr(model.fit_residual, "__defaults__", (1,))
    _, test = evaluate(df, cut, use_market=True)
    mkt = normalize_within_race(test, (1 / test["win_odds"]).to_numpy())
    assert np.allclose(test["p_win"].to_numpy(), mkt)
