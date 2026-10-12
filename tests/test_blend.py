import numpy as np
import pandas as pd
import pytest

from keibaai import blend as bl
from keibaai.features import build_features, load_results

from test_model import synthetic_db


def simulate(a, b, n_races=1500, n=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        pm = rng.dirichlet(np.ones(n) * 2)
        pq = rng.dirichlet(np.ones(n) * 2)
        z = a * np.log(pm) + b * np.log(pq)
        p = np.exp(z - z.max()); p /= p.sum()
        w = rng.choice(n, p=p)
        for i in range(n):
            rows.append({"race_id": f"r{r:05d}", "number": i + 1, "p_win": pm[i], "win_odds": 0.8 / pq[i],
                         "win": int(i == w), "date": pd.Timestamp("2025-01-01")})
    return pd.DataFrame(rows)


def test_fit_recovers_known_weights():
    fit = bl.fit_blend(simulate(0.5, 1.0))
    assert abs(fit.a - 0.5) < 0.2 and abs(fit.b - 1.0) < 0.2
    assert fit.loglik >= fit.baseline_loglik  # 混ぜたほうが市場だけより悪くならない


def test_market_only_when_model_is_noise():
    fit = bl.fit_blend(simulate(0.0, 1.0))
    assert fit.a < 0.2 and abs(fit.b - 1.0) < 0.2


def test_apply_keeps_probabilities_per_race():
    d = simulate(0.5, 1.0, n_races=50)
    out = bl.apply_blend(d, bl.Blend(0.5, 1.0, 0, 0))
    assert np.allclose(out.groupby("race_id")["p_win"].sum(), 1.0)
    assert np.allclose(out["p_model"], d.sort_values(["race_id", "number"])["p_win"].to_numpy())
    only_market = bl.apply_blend(d, bl.Blend(0.0, 1.0, 0, 0))
    assert np.allclose(only_market["p_win"], only_market["p_mkt"])


def test_races_without_odds_are_dropped_and_empty_fit_raises():
    d = simulate(0.5, 1.0, n_races=20)
    d.loc[d["race_id"] == "r00003", "win_odds"] = np.nan
    assert "r00003" not in set(bl.apply_blend(d, bl.Blend(0.5, 1.0, 0, 0))["race_id"])
    with pytest.raises(ValueError):
        bl.fit_blend(d.assign(win_odds=np.nan))


def test_predict_test_blend_uses_only_earlier_data():
    df = build_features(load_results(synthetic_db()))
    test, blend = bl.predict_test_blend(df, "2026-01-01", valid_months=6)
    assert test["date"].min() >= pd.Timestamp("2026-01-01")
    assert np.allclose(test.groupby("race_id")["p_win"].sum(), 1.0)
    assert blend.b > 0  # 合成データのオッズは強さを反映しているので、市場の重みは正


def test_cli_backtest_blend(tmp_path, capsys):
    import sqlite3

    from keibaai import cli

    path = tmp_path / "k.db"
    dst = sqlite3.connect(path)
    synthetic_db().backup(dst)
    dst.close()
    cli.main(["backtest", "--db", str(path), "--test-from", "2026-09-01", "--blend"])
    out = capsys.readouterr().out
    assert "混ぜ方: モデルの重み" in out and "混ぜた確率" in out and "単勝を毎回" in out
