import pandas as pd

from keibaai import db
from keibaai.jvtraining import JV_FEATURES, add_course_relative, add_jv_features, import_raw, parse_record

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
    assert out.iloc[1][JV_FEATURES].isna().all() and out.iloc[2][JV_FEATURES].isna().all()


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
    out = add_course_relative(tr)
    # 美浦坂路は改修の前(中央値53)と後(中央値55)で別グループ。ウッドは1本だけなので差は0
    assert out["f4_z"].tolist() == [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0]
    assert out["f4_pct"].iloc[0] < out["f4_pct"].iloc[2]
    assert out["accel"].iloc[0] == (52.0 - 39.0) - 12.0
