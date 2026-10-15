import numpy as np
import pandas as pd
import pytest

from keibaai import feature_trials as ft
from keibaai.features import build_features, load_results

from test_model import synthetic_db

MID = "2025-06-01"


@pytest.fixture(scope="module")
def raw():
    conn = synthetic_db(n_days=40, races_per_day=6)
    conn.execute("UPDATE results SET passing = (number % 5 + 1) || '-' || finish")
    conn.execute("UPDATE races SET course = CASE WHEN round % 2 = 0 THEN '外' ELSE '内' END, rail = 'A'")
    return load_results(conn).merge(pd.read_sql_query("SELECT race_id, course, rail FROM races", conn), on="race_id")


def candidates(raw: pd.DataFrame) -> pd.DataFrame:
    df = build_features(raw.copy())
    df = ft.add_bias_candidates(df, ft.market_top3(df, MID))
    return ft.add_affinity_candidates(df)


def test_market_top3_sums_to_three_per_race(raw):
    df = build_features(raw.copy())
    q = ft.market_top3(df, MID)
    assert q.notna().all()
    s = q.groupby(df["race_id"]).sum()
    assert np.allclose(s, 3.0, atol=0.3)  # 1 を超えないよう切るので、ぴったりでなくてよい


def test_same_day_bias_uses_only_earlier_races_of_the_day(raw):
    day = raw["date"].sort_values().unique()[30]  # 市場の見込みを作る期間(MID より前)より後
    a = candidates(raw)
    # その日の 4R 以降と、翌日以降の結果をでたらめにしても、その日の 1〜4R の値は変わらない
    b_raw = raw.copy()
    m = (b_raw["date"] > day) | ((b_raw["date"] == day) & (b_raw["round"] >= 4))
    rng = np.random.default_rng(1)
    b_raw.loc[m, "finish"] = rng.integers(1, 11, int(m.sum()))
    b_raw.loc[m, "passing"] = [f"{x}-{y}" for x, y in rng.integers(1, 10, (int(m.sum()), 2))]
    b = candidates(b_raw)
    keep = (a["date"] < day) | ((a["date"] == day) & (a["round"] <= 4))
    cols = ft.BIAS_COLUMNS + ["tj_top3_shr10", "js_top3_shr10", "ts_win_shr5", "tj_top3_jshr20"]
    ka = a[keep].sort_values(["race_id", "number"])[cols].reset_index(drop=True)
    kb = b[keep].sort_values(["race_id", "number"])[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(ka, kb)


def test_same_day_bias_counts_earlier_races(raw):
    df = candidates(raw)
    r = df.groupby("race_id").first()
    assert (r.loc[r["round"] == 1, "today_n_races"] == 0).all()
    assert r.loc[r["round"] == 1, "today_front_bias"].isna().all()
    assert (r.loc[r["round"] == 6, "today_n_races"] == 5).all()
    assert r.loc[r["round"] == 6, "today_front_res"].notna().all()
    assert r["today_front_bias"].abs().max() <= 1


def test_same_day_bias_by_hand():
    # 1R: 前に付けた馬(序盤1番手)が勝った。2R の当日の先行有利度は、1R だけから計算する
    rows = []
    for rnd in (1, 2):
        for num in range(1, 7):
            rows.append(dict(race_id=f"r{rnd}", date=pd.Timestamp("2025-01-05"), round=rnd, venue="東京",
                             course_type="芝", course="", rail="", dist_bucket=4, number=num, field_size=6,
                             finish=num, early_pos=num / 6))
    df = pd.DataFrame(rows)
    out = ft.add_bias_candidates(df, pd.Series(0.5, index=df.index))
    r2 = out[out["race_id"] == "r2"].iloc[0]
    # 前の3分の1(1・2番)は2頭とも3着内、ほかの4頭は1頭(3番)が3着内 → 1 − 1/4
    assert r2["today_front_bias"] == pytest.approx(0.75)
    assert r2["today_front_res"] == pytest.approx(0.75)  # 見込みが一律なら、差し引いても同じ
    assert r2["today_inner_bias"] == pytest.approx(1.0)  # 内(1・2番)は 2/2、外(5・6番)は 0/2
    assert np.isnan(out[out["race_id"] == "r1"].iloc[0]["today_front_bias"])


def test_shrunk_rate_by_hand():
    df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
                       "trainer_id": "t", "jockey_id": "j", "finish": [1, 5, 2], "top3": [1, 0, 1]})
    s = ft.shrunk_rate(df, ["trainer_id", "jockey_id"], "top3", 10, pd.Series(0.2, index=df.index))
    assert s.tolist() == pytest.approx([0.2, (1 + 2) / 11, (1 + 2) / 12])


def test_run_compares_variants(raw, tmp_path):
    df = candidates(raw)
    variants = {"base": ([], []), "相性": (["tj_top3_shr10", "js_top3_shr10"], ["tj_top3_rate"])}
    res = ft.run(df, variants, seeds=1, rounds=5, out_dir=tmp_path, periods=[MID])
    assert set(res["variant"]) == {"base", "相性"} and len(res) == 4
    assert res["auc"].between(0, 1).all()
    assert "相性" in ft.to_markdown(res)
    again = ft.run(df, variants, seeds=1, rounds=5, out_dir=tmp_path, periods=[MID])  # 保存から読み直す
    pd.testing.assert_frame_equal(res, again)
