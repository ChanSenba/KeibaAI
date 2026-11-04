import numpy as np
from test_model import synthetic_db

from keibaai.factors import FACTORS
from keibaai.features import build_features, load_results
from keibaai.nn_model import fit_predict


def test_fit_predict_gives_a_probability_per_race_and_per_factor():
    df = build_features(load_results(synthetic_db(n_days=30)))
    cut = sorted(df["date"].unique())[25]
    train, target = df[df["date"] < cut], df[df["date"] >= cut]
    p, parts = fit_predict(train, target)
    assert p.index.equals(target.index) and p.notna().all()
    assert np.allclose(p.groupby(target["race_id"]).sum(), 1.0)
    assert list(parts.columns) == list(FACTORS)
    for name in FACTORS:  # 要素ごとの勝率もレース内で合計1
        assert np.allclose(parts[name].groupby(target["race_id"]).sum(), 1.0)
    again, _ = fit_predict(train, target)
    assert np.allclose(p, again)  # 乱数を固定しているので、同じデータなら同じ予想


def test_past_runs_use_only_earlier_runs_of_the_same_horse():
    """過去走の塔の入力: k走前の結果だけが入り、その後の結果を書き換えても変わらない。"""
    import pandas as pd

    from keibaai.nn_model import past_runs

    df = build_features(load_results(synthetic_db(n_days=30)))
    X, M = past_runs(df)
    h = df["horse_id"].value_counts().index[0]
    rows = df.index[df["horse_id"] == h]
    first, later = rows[0], rows[-1]
    assert not M[first].any()  # 初めての走りには過去走がない
    n_before = (df.loc[rows, "date"] < df.loc[later, "date"]).sum()
    assert M[later].sum() == min(n_before, X.shape[1])
    changed = df.copy()
    changed.loc[changed["date"] >= df.loc[later, "date"], "finish"] = 1.0  # 今回以後の結果を変える
    X2, M2 = past_runs(changed)
    assert np.allclose(X[later], X2[later]) and (M[later] == M2[later]).all()
    # 1走前の着順の位置は、その走りの (着順 − 1) ÷ (完走頭数 − 1)
    prev = rows[n_before - 1]
    race = df[df["race_id"] == df.loc[prev, "race_id"]]
    size = pd.to_numeric(race["finish"], errors="coerce").gt(0).sum()
    assert np.isclose(X[later, 0, 0], (df.loc[prev, "finish"] - 1) / (size - 1))
