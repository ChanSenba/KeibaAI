import numpy as np
import pandas as pd
import pytest

from keibaai import factors
from keibaai.features import (ARTICLE_FEATURES, BASE_FEATURES, BIAS_FEATURES, build_features, load_results,
                              parse_late_position)

from test_model import synthetic_db


@pytest.fixture(scope="module")
def raw():
    conn = synthetic_db(n_days=40, races_per_day=4)
    rng = np.random.default_rng(3)
    conn.execute("UPDATE results SET time_sec = 90 + finish * 0.2 + abs(random() % 10) / 10.0")
    conn.execute("UPDATE results SET passing = (number % 5 + 1) || '-' || (number % 5 + 1) || '-' || finish")
    conn.execute("UPDATE results SET last3f = 33 + (finish % 7) / 2.0")
    conn.execute("UPDATE races SET direction = CASE WHEN substr(race_id, 7, 2) % 2 = 0 THEN '右' ELSE '左' END")
    return load_results(conn)


def scramble_from(raw, day):
    out = raw.copy()
    m = out["date"] >= day
    n = int(m.sum())
    rng = np.random.default_rng(9)
    out.loc[m, "finish"] = rng.integers(1, 11, n)
    out.loc[m, "last3f"] = rng.uniform(32, 40, n)
    out.loc[m, "time_sec"] = rng.uniform(80, 100, n)
    out.loc[m, "passing"] = [f"{a}-{b}" for a, b in zip(rng.integers(1, 10, n), rng.integers(1, 10, n))]
    return out


def test_features_ignore_results_of_the_same_and_later_days(raw):
    day = sorted(raw["date"].unique())[25]
    a, b = build_features(raw.copy()), build_features(scramble_from(raw, day))
    cols = BASE_FEATURES + BIAS_FEATURES + ARTICLE_FEATURES
    key = ["race_id", "number"]
    a = a[a["date"] == day].sort_values(key).reset_index(drop=True)
    b = b[b["date"] == day].sort_values(key).reset_index(drop=True)
    assert len(a) > 0
    for c in cols:
        x, y = a[c].astype(float).to_numpy(), b[c].astype(float).to_numpy()
        assert np.allclose(x, y, equal_nan=True), f"{c} が当日以後の結果に左右されている"


def test_article_features_exist_and_are_mostly_filled(raw):
    df = build_features(raw.copy())
    assert set(ARTICLE_FEATURES) <= set(df.columns)
    late = df[df["n_prev"] >= 5]
    for c in ["horse_ct_top3_rate", "l3f_rank_3", "time_diff_5", "pos_gain_3", "weeks_since_last", "jockey_ct_top3_rate",
              "style_front_rate", "n_front_runners", "inner_pos"]:
        assert late[c].notna().mean() > 0.9, c
    first = df.groupby("horse_id").head(1)
    assert first["runs_since_layoff"].eq(1).all() and first["weeks_since_last"].isna().all()
    assert first["l3f_rank_3"].isna().all() and first["pair_n"].eq(0).all()


def test_rotation_and_layoff_by_hand():
    from keibaai.features import build_features as bf

    rows = []
    for i, (d, fin) in enumerate([("2025-01-05", 3), ("2025-01-12", 1), ("2025-01-19", 2), ("2025-06-01", 5), ("2025-06-15", 4)]):
        rows.append({"race_id": f"r{i}", "number": 1, "date": pd.Timestamp(d), "finish": fin, "horse_id": "h",
                     "sex_age": "牡4", "weight_carried": 56.0, "jockey_id": "j", "trainer_id": "t", "venue": "東京",
                     "course_type": "芝", "distance": 1600, "ground": "良", "round": 1, "race_name": "x", "body_weight": 480,
                     "body_weight_diff": 0, "win_odds": 5.0, "popularity": 3, "bracket": 1, "last3f": 34.0, "time_sec": 95.0,
                     "passing": "3-3", "direction": "右"})
    out = bf(pd.DataFrame(rows)).sort_values("date")
    assert out["runs_since_layoff"].tolist() == [1, 2, 3, 1, 2]  # 6月は休み明け(約19週)の初戦
    assert out["is_renso"].tolist()[1:3] == [1.0, 1.0] and out["is_renso"].tolist()[3] == 0.0  # 中0〜1週は連闘扱い
    assert out["is_layoff"].tolist()[3] == 1.0 and out["is_layoff"].tolist()[4] == 0.0
    assert out["pair_n"].tolist() == [0, 1, 2, 3, 4]
    assert out["horse_ct_top3_rate"].tolist()[1:4] == [1.0, 1.0, 1.0] and out["prev_same_ct_finish"].tolist()[1] == 3


def test_late_position_parser():
    assert parse_late_position("13-12-12-7") == 7 and np.isnan(parse_late_position(None))
    assert np.isnan(parse_late_position(float("nan"))) and parse_late_position("5") == 5


def test_every_factor_runs_with_new_features(raw):
    df = build_features(raw.copy())
    reports, uniform = factors.evaluate_factors(df, str(sorted(raw["date"].unique())[30].date()), rounds=30)
    names = [r.name for r in reports]
    assert {"適性", "ローテーション", "総合+全要素"} <= set(names)
    assert all(np.isfinite(r.win_logloss) for r in reports)


def test_course_flow_uses_only_earlier_days_at_the_same_course():
    from keibaai.features import _course_flow

    def day(date, rid, first, last, finishes, early):
        n = len(finishes)
        return pd.DataFrame({"race_id": rid, "venue": "東京", "course_type": "芝", "distance": 1600,
                             "date": pd.Timestamp(date), "finish": finishes, "early_pos": early,
                             "pace_first3f": first, "pace_last3f": last, "early_pos_rel_3": early})
    # 1日目: スロー(35.5-34.0=+1.5)で前の馬だけが3着以内 / 2日目: ハイ(33.0-36.0=-3.0)で後ろの馬だけが3着以内
    d1 = day("2030-01-05", "a", 35.5, 34.0, [1, 2, 3, 4, 5, 6], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    d2 = day("2030-01-12", "b", 33.0, 36.0, [4, 5, 6, 1, 2, 3], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    d3 = day("2030-01-19", "c", None, None, [np.nan] * 3, [0.1, 0.5, 0.9])  # これから走るレース
    d3["pace_first3f"] = d3["pace_last3f"] = np.nan
    out = _course_flow(pd.concat([d1, d2, d3], ignore_index=True))
    assert out.loc[:5, "course_pace_diff_5"].isna().all()  # 1日目は過去がない
    assert (out.loc[6:11, "course_pace_diff_5"] == 1.5).all()  # 2日目は1日目だけ
    assert (out.loc[12:, "course_pace_diff_5"] == (1.5 - 3.0) / 2).all()  # 3日目は両方の平均
    assert out.loc[12:, "course_front_edge_5"].iloc[0] == 0.0  # +1(前が有利)と −1(後ろが有利)の平均
    assert out.loc[12, "front_x_course_edge"] == 0.0
