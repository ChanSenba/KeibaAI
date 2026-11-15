import numpy as np
import pandas as pd

from keibaai import nn_model, nn_year


def frame():
    dates = pd.to_datetime(["2019-06-01", "2020-03-01", "2021-05-01", "2021-12-31", "2022-01-02"])
    rows = []
    for i, d in enumerate(dates):
        for n in (1, 2):
            rows.append({"race_id": f"r{i}", "date": d, "number": n, "horse_id": f"h{n}",
                         "finish": float(n) if i < 4 else np.nan, "win_odds": 2.0 * n})
    return pd.DataFrame(rows)


def test_year_net_learns_only_before_that_year(monkeypatch, tmp_path):
    seen = {}

    def fit(train, n_seeds=None):
        return [{"max": train["date"].max()}]

    def predict(models, train, rows):
        seen[int(rows["date"].dt.year.iloc[0])] = (models[0]["max"], set(rows.index) & set(train.index))
        return pd.Series(0.5, index=rows.index), None

    monkeypatch.setattr(nn_model, "fit_models", fit)
    monkeypatch.setattr(nn_model, "predict_models", predict)
    monkeypatch.setattr(nn_year, "CACHE_DIR", None)
    out = nn_year.add_nn_year(frame())
    assert set(seen) == {2020, 2021, 2022}  # 2019 は前で学べないので空
    for year, (last, overlap) in seen.items():
        assert last < pd.Timestamp(f"{year}-01-01") and not overlap
    assert out.loc[out["date"].dt.year == 2019, "nn_lp"].isna().all()
    assert np.allclose(out.loc[out["date"].dt.year > 2019, "nn_lp"], np.log(0.5))
    # 出馬表の行(結果なし)にも付き、市場との差は対数の差
    entry = out[out["finish"].isna()]
    assert entry["nn_lp"].notna().all()
    mkt = np.log((1 / entry["win_odds"]) / (1 / entry["win_odds"]).sum())
    assert np.allclose(entry["nn_vs_mkt"], entry["nn_lp"] - mkt)


def test_year_nets_are_saved_and_reused(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(nn_model, "fit_models", lambda train, n_seeds=None: calls.append(1) or [{}])
    monkeypatch.setattr(nn_model, "predict_models",
                        lambda m, train, rows: (pd.Series(0.5, index=rows.index), None))
    monkeypatch.setattr(nn_year, "CACHE_DIR", tmp_path)
    nn_year.add_nn_year(frame())
    n = len(calls)
    nn_year.add_nn_year(frame())
    assert n == 3 and len(calls) == n
    assert len(list(tmp_path.glob("nnyear-*.pkl"))) == 3
