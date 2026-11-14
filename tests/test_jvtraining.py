import pytest
import pandas as pd

from keibaai import db
from keibaai.jvtraining import (JV_FEATURES, VOLUME_FEATURES, add_course_relative, add_day_relative, add_jv_features,
                                import_raw, parse_record)

HC = "HC12026100502026100506112022103274064918304661710295151144\r\n"


def wood(horse="2022103274", f4="0520", l1="125"):
    """ウッド WC の1件(105文字): 10F〜2F の合計+ラップの9組と1Fのラップ。"""
    pairs = ["1200" "180", "1100" "170", "1000" "160", "0900" "150", "0800" "140", "0700" "130", f4 + "120", "0390" "130", "0260" "135"]
    return "WC1" + "20261005" + "1" + "20261006" + "0700" + horse + "2" + "1" + "0" + "".join(pairs) + l1 + "\r\n"


def test_parse_slope_record():
    r = parse_record(HC)
    assert r == ("2022103274", "2026-10-05", "0611", "slope", "0", None, None, 64.9, 46.6, 29.5, 14.4)
    assert round(r[7] - r[8], 1) == 18.3  # 4F − 3F = 最初の1Fのラップ(18.3)


def test_parse_wood_record():
    assert len(wood()) == 105
    r = parse_record(wood())
    assert r == ("2022103274", "2026-10-06", "0700", "wood", "1", "2", "1", 52.0, 39.0, 26.0, 12.5)


def test_parse_skips_deleted_and_unknown():
    assert parse_record("HC0" + HC[3:]) is None  # 区分 0 は削除
    assert parse_record("XX" + HC[2:]) is None
    assert parse_record(HC[:40]) is None  # 短すぎる


def test_import_raw_is_idempotent(tmp_path):
    (tmp_path / "SLOP.txt").write_text(f"a.jvd\t{HC}a.jvd\t{HC}", encoding="utf-8")
    (tmp_path / "WOOD.txt").write_text(f"b.jvd\t{wood()}", encoding="utf-8")
    conn = db.connect(":memory:")
    assert import_raw(conn, tmp_path) == 3
    import_raw(conn, tmp_path)
    assert conn.execute("SELECT COUNT(*) FROM jv_training").fetchone()[0] == 2


def _train(conn, rows):
    conn.executemany("INSERT INTO jv_training (horse_id, date, time, kind, f4, l1) VALUES (?,?,?,?,?,?)", rows)


def test_add_jv_features():
    conn = db.connect(":memory:")
    base = [("h1", f"2026-05-{d:02d}", "0600", "slope", f4, 12.0) for d, f4 in ((1, 54.0), (8, 55.0), (15, 56.0))]
    near = [("h1", "2026-09-28", "0600", "slope", 53.0, 12.2), ("h1", "2026-10-01", "0600", "slope", 52.0, 11.8),
            ("h1", "2026-10-07", "0600", "slope", 52.5, 11.5),  # レース当日なので使わない
            ("h2", "2026-10-01", "0600", "slope", 60.0, 14.0)]  # 4F58秒を超える流しは数えない
    _train(conn, base + near)
    df = pd.DataFrame({"race_id": ["r1", "r1", "r1"], "horse_id": ["h1", "h2", "h3"],
                       "date": pd.to_datetime(["2026-10-07"] * 3)})
    out = add_jv_features(conn, df)
    a = out.iloc[0]
    assert (a.jv_n14, a.jv_best_f4, a.jv_best_l1, a.jv_last_f4, a.jv_last_l1, a.jv_days_since) == (2, 52.0, 11.8, 52.0, 11.8, 6)
    assert a.jv_f4_vs_base == 52.0 - 55.0
    assert out.iloc[0].jv_f4_vs_field == 0.0  # 調教のある馬が1頭だけなので平均との差は0
    timed = [c for c in JV_FEATURES if c not in VOLUME_FEATURES]
    assert out.iloc[1][timed].isna().all() and out.iloc[2][timed].isna().all()
    # 本数: 流しも数える。レース当日の1本は数えない。調教のない馬は 0
    assert (a.jv_n_all14, a.jv_n_all28, a.jv_n_fast28, a.jv_wood_share28, a.jv_last_is_wood) == (2, 2, 2, 0.0, 0.0)
    b = out.iloc[1]
    assert (b.jv_n_all28, b.jv_n_fast28) == (1, 0) and pd.isna(b.jv_wood_share28)
    assert (out.iloc[2].jv_n_all28, out.iloc[2].jv_n_fast28) == (0, 0)


def test_volume_is_missing_before_wood_data_and_when_training_is_stale():
    conn = db.connect(":memory:")
    _train(conn, [("x", "2021-07-02", "0600", "slope", 55.0, 12.0), ("x", "2021-09-20", "0600", "slope", 55.0, 12.0)])
    df = pd.DataFrame({"race_id": ["r", "s", "t"], "horse_id": ["h", "h", "h"],
                       "date": pd.to_datetime(["2021-07-10", "2021-09-20", "2021-10-10"])})
    out = add_jv_features(conn, df)
    assert out.iloc[0][VOLUME_FEATURES].isna().all()  # ウッドの揃う前
    assert out.iloc[1].jv_n_all28 == 0  # 3日前までの調教が入っているので「調教なし」
    assert pd.isna(out.iloc[2].jv_n_all28)  # 取り込みが3日前まで届いていないので欠損


def test_volume_counts_up_to_three_days_before_and_no_recent_work():
    conn = db.connect(":memory:")
    _train(conn, [("h", "2026-10-04", "0600", "slope", 55.0, 12.0),  # 3日前: 数える
                  ("h", "2026-10-05", "0600", "slope", 70.0, 15.0),  # 2日前の流し: 数えない
                  ("g", "2026-08-01", "0600", "slope", 55.0, 12.0)])  # 28日より前だけの馬
    df = pd.DataFrame({"race_id": ["r", "r"], "horse_id": ["h", "g"], "date": pd.to_datetime(["2026-10-07"] * 2)})
    out = add_jv_features(conn, df)
    assert (out.iloc[0].jv_n_all28, out.iloc[0].jv_n_fast28) == (1, 1)
    assert (out.iloc[1].jv_n_all28, out.iloc[1].jv_n_fast28) == (0, 0)


def test_day_relative_compares_with_same_day_same_course_only():
    rows = [("h0", "2026-10-01", "0600", "slope", "1", None, 50.0 + i, 12.0 + i / 10) for i in range(9)]
    rows += [("w", "2026-10-01", "0600", "wood", "1", "2", 40.0, 11.0)]  # ウッドは別の基準
    rows += [("x", "2026-10-02", "0700", "slope", "1", None, 45.0, 11.0)]  # 別の日は別の基準(本数不足で欠損)
    tr = pd.DataFrame(rows, columns=["horse_id", "date", "time", "kind", "center", "course", "f4", "l1"])
    tr["date"] = pd.to_datetime(tr["date"])
    out = add_day_relative(tr)
    assert list(out["day_f4"].iloc[:9]) == [float(i - 4) for i in range(9)]
    assert out["day_f4"].iloc[9:].isna().all() and out["day_l1"].iloc[9:].isna().all()
    assert out["hour"].tolist() == [6] * 10 + [7]


def test_day_features_use_workouts_before_race_day():
    conn = db.connect(":memory:")
    rows = [(f"o{i}", "2026-10-01", "0600", "slope", 50.0 + i, 12.0) for i in range(8)]
    rows += [("h", "2026-10-01", "0730", "slope", 51.0, 11.9),
             ("h", "2026-10-07", "0600", "slope", 49.0, 11.0)]  # レース当日なので使わない
    _train(conn, rows)
    df = pd.DataFrame({"race_id": ["r"], "horse_id": ["h"], "date": pd.to_datetime(["2026-10-07"])})
    a = add_jv_features(conn, df).iloc[0]
    assert a.jv_last_day_f4 == 51.0 - 53.0 and a.jv_best_day_f4 == a.jv_last_day_f4
    assert a.jv_last_day_l1 == 11.9 - 12.0
    assert a.jv_last_hour == 7 and a.jv_n_all28 == 1


def test_add_jv_features_without_table_data():
    conn = db.connect(":memory:")
    df = pd.DataFrame({"race_id": ["r"], "horse_id": ["h"], "date": pd.to_datetime(["2026-10-07"])})
    assert add_jv_features(conn, df)[JV_FEATURES].isna().all(axis=None)


def test_course_relative_groups_by_center_kind_and_miho_renewal():
    tr = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-10-05", "2023-10-06", "2023-10-07", "2023-01-01"],
        "kind": ["slope"] * 6 + ["wood"], "center": ["0", "0", "0", "0", "0", "0", "0"], "course": [None] * 6 + ["A"],
        "f4": [52.0, 53.0, 54.0, 54.0, 55.0, 56.0, 50.0],
        "f3": [39.0, 40.0, 41.0, 41.0, 42.0, 43.0, 37.0], "f2": [25.0] * 7, "l1": [12.0, 12.0, 12.0, 13.0, 13.0, 13.0, 11.0]})
    out = add_course_relative(tr, min_history=1)
    # 各グループの初日は基準なし。改修後は改修前の時計を使わない。
    assert out["f4_z"].iloc[[0, 3, 6]].isna().all()
    assert out["f4_z"].iloc[[1, 2, 4, 5]].tolist() == [1.0, 1.5, 1.0, 1.5]
    assert out["f4_pct"].iloc[2] == 1.0
    assert out["accel"].iloc[0] == (52.0 - 39.0) - 12.0


def test_laps_of_the_4f_workout():
    conn = db.connect(":memory:")
    # 4F 52.0 / 3F 38.5 / 2F 25.0 / 1F 12.2 → ラップ 13.5 / 13.5 / 12.8 / 12.2
    conn.execute("INSERT INTO jv_training (horse_id, date, time, kind, center, f4, f3, f2, l1) "
                 "VALUES ('h1', '2026-10-01', '0600', 'slope', '1', 52.0, 38.5, 25.0, 12.2)")
    conn.execute("INSERT INTO jv_training (horse_id, date, time, kind, center, f4, f3, f2, l1) "
                 "VALUES ('h1', '2026-10-05', '0600', 'slope', '1', 54.0, 40.0, 26.5, 13.0)")
    df = pd.DataFrame({"race_id": ["r1"], "horse_id": ["h1"], "date": pd.to_datetime(["2026-10-07"])})
    a = add_jv_features(conn, df).iloc[0]
    assert (a.jv_last_lap1, a.jv_last_lap2, a.jv_last_lap3) == pytest.approx((14.0, 13.5, 13.5))  # 最後の追い切り
    assert (a.jv_best_lap1, a.jv_best_lap2, a.jv_best_lap3) == pytest.approx((13.5, 13.5, 12.8))  # 一番時計


def test_lap_z_is_relative_to_earlier_workouts_only():
    rows = [("h", f"2026-09-{d:02d}", "0600", "slope", "1", None, 52.0, 38.5, 25.5, 12.5) for d in range(1, 31)]
    rows += [("h", "2026-10-05", "0600", "slope", "1", None, 54.0, 40.0, 26.5, 13.0)]  # lap3 = 13.5(それまでは 13.0)
    tr = pd.DataFrame(rows, columns=["horse_id", "date", "time", "kind", "center", "course", "f4", "f3", "f2", "l1"])
    out = add_course_relative(tr)
    assert out["lap3"].iloc[-1] == pytest.approx(13.5)
    assert out["lap3_z"].iloc[-1] == pytest.approx(0.5)  # それまでの30本の中央値 13.0 との差
    assert out["lap3_z"].iloc[:30].isna().all()  # 同日・未来は使わないので、過去が足りない間は欠損
