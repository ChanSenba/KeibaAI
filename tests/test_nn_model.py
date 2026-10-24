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
