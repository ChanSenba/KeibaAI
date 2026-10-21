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
    return load_results(conn).merge(pd.read_sql_query("SELECT race_id, course, rail, weather FROM races", conn), on="race_id")


def candidates(raw: pd.DataFrame) -> pd.DataFrame:
    df = build_features(raw.copy())
    df = ft.add_bias_candidates(df, ft.market_top3(df, MID))
    return ft.add_more_candidates(ft.add_affinity_candidates(df))


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


def test_race_level_uses_only_next_runs_finished_before_the_day():
    # A と B が 1/1 のレースで一緒に走り、B は 1/8 に勝った。A の 1/8 の前走レベルには B の 1/8 は入らない
    d = pd.to_datetime
    df = pd.DataFrame({
        "race_id": ["r1", "r1", "r2", "r3", "r4"], "horse_id": ["A", "B", "B", "A", "A"],
        "date": d(["2025-01-01", "2025-01-01", "2025-01-08", "2025-01-08", "2025-01-15"]),
        "finish": [2, 1, 1, 5, 3], "top3": [1, 1, 1, 0, 1]})
    lvl, n = ft._race_level(df, 1)
    assert np.isnan(lvl[3]) and n[3] == 0  # A の 1/8: B の次走は同じ日なので数えない
    # A の 1/15 の前走は r3(1/8)で、ほかの馬はいない
    assert n[4] == 0
    lvl2, n2 = ft._race_level(df, 2)
    assert lvl2[4] == 1.0 and n2[4] == 1  # 2走前の r1 の B は、1/8 に3着以内


def test_more_candidates_ignore_results_of_the_same_and_later_days(raw):
    day = raw["date"].sort_values().unique()[30]
    a = candidates(raw)
    b_raw = raw.copy()
    m = b_raw["date"] >= day
    rng = np.random.default_rng(2)
    b_raw.loc[m, "finish"] = rng.integers(1, 11, int(m.sum()))
    b_raw.loc[m, "time_sec"] = rng.uniform(80, 100, int(m.sum()))
    b = candidates(b_raw)
    cols = [c for c in ft.MORE_COLUMNS if c not in ("bw_vs_season",)]
    keep = a["date"] <= day
    ka = a[keep].sort_values(["race_id", "number"])[cols].reset_index(drop=True)
    kb = b[keep].sort_values(["race_id", "number"])[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(ka, kb)
    assert a["meet_day"].notna().all() and a["jockey_rank_365"].notna().any()


def test_ability_model_uses_the_shrunk_trainer_jockey_rate(raw):
    from keibaai.features import ABILITY_MODEL_FEATURES, MARKET_MODEL_FEATURES, TJ_SHRINK_K

    assert "tj_top3_shr" in ABILITY_MODEL_FEATURES and "tj_top3_rate" not in ABILITY_MODEL_FEATURES
    assert "tj_top3_rate" in MARKET_MODEL_FEATURES and "tj_top3_shr" not in MARKET_MODEL_FEATURES
    df = candidates(raw)  # 検証用の計算(k ごと)と、予想に使う列が一致する
    assert np.allclose(df["tj_top3_shr"], df[f"tj_top3_shr{TJ_SHRINK_K}"], equal_nan=True)


def _training_db(rows):
    from keibaai import db
    conn = db.connect(":memory:")
    conn.executemany("INSERT INTO jv_training (horse_id, date, time, kind) VALUES (?,?,'0600',?)", rows)
    return conn


def _runs(rows):
    df = pd.DataFrame(rows, columns=["race_id", "horse_id", "date", "owner", "finish"])
    df["date"] = pd.to_datetime(df["date"])
    df["number"] = 1
    df["top3"] = (df["finish"] <= 3).astype(int)
    return df


def test_gaikyu_estimates_return_after_a_gap():
    days = pd.date_range("2024-01-01", "2024-03-01", freq="7D")  # 在厩で毎週
    back = pd.date_range("2024-05-01", "2024-05-12", freq="3D")  # 2か月空いて帰厩
    rows = [("h1", d.date().isoformat(), "wood") for d in days.append(back)]
    conn = _training_db(rows + [("h9", "2023-10-01", "wood")])  # 記録はこの日から
    df = _runs([("r1", "h1", "2024-02-25", "サンデーレーシング", 2),
                ("r2", "h1", "2024-05-15", "サンデーレーシング", 1),
                ("r3", "h1", "2024-05-10", "サンデーレーシング", 1)])  # 当日の記録は使わない
    out = ft.add_gaikyu_candidates(conn, df.iloc[:2].copy())
    r2 = out.iloc[1]
    assert r2["gk_in_days"] == 14 and r2["gk_short"] == 1.0
    assert r2["gk_gap_days"] == (pd.Timestamp("2024-05-01") - days[-1]).days
    assert r2["gk_n_since_in"] == len(back)
    assert r2["gk_stayed"] == 0.0 and r2["gk_short_change"] == 1.0  # 前走は在厩、今回は外厩帰り
    assert out.iloc[0]["gk_short"] == 0.0 and out.iloc[0]["owner_northern"] == 1.0
    one = ft.add_gaikyu_candidates(conn, df.iloc[[2]].copy()).iloc[0]
    assert one["gk_in_days"] == 9 and one["gk_n_since_in"] == len(back) - 1


def test_gaikyu_is_unknown_without_recent_or_early_records():
    rows = [("h1", "2024-01-01", "wood"), ("h1", "2024-01-08", "slope"), ("h2", "2024-03-01", "slope")]
    conn = _training_db(rows)
    df = _runs([("r1", "h1", "2024-01-10", "個人", 3),  # 記録の始まりの直後: 前に空白があったか分からない
                ("r2", "h1", "2024-03-01", "個人", 3),  # 最後の記録が30日より前
                ("r3", "h2", "2024-03-10", "個人", 3)])  # 初めての記録から(新しく入った馬)
    out = ft.add_gaikyu_candidates(conn, df)
    assert out["gk_in_days"].iloc[:2].isna().all()
    assert out["gk_in_days"].iloc[2] == 9 and np.isnan(out["gk_gap_days"].iloc[2])
    assert out["owner_northern"].eq(0.0).all()


def test_gaikyu_without_training_data(raw):
    from keibaai import db
    df = build_features(raw.copy())
    out = ft.add_gaikyu_candidates(db.connect(":memory:"), df)
    assert out[["gk_in_days", "gk_short", "owner_short_top3_shr"]].isna().all().all()
    assert set(ft.GAIKYU_COLUMNS) <= set(out.columns)
