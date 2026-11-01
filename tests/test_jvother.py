import numpy as np
import pandas as pd

from keibaai.jvother import add_other_features, import_other, load_other, parse_other_ra, parse_other_se

from test_jvrace import ra, se, write


def other_ra(race, kubun="A", track="24", dist="1400"):
    return ra(race=race, kubun=kubun, track=track, over={697: dist})


def other_se(race, number, horse, finish="02", kubun="A", time="1272", abnormal="0"):
    return se(race=race, number=number, abnormal=abnormal,
              over={2: kubun, 30: horse, 334: finish, 338: time})


def test_parse_other_records():
    assert parse_other_ra(other_ra("2024050644010301")) == ("202444010301", "ダ", 1400, "富士ステークス")
    r = parse_other_se(other_se("2024050644010301", "03", "2020101234"))
    assert r[:6] == ("202444010301", 3, "2020101234", "2024-05-06", "地方", "44")
    assert r[8:] == (2, None, 87.2)
    abroad = parse_other_se(other_se("20240330C7010105", "05", "2020101234", kubun="B", time="0000"))
    assert abroad[4] == "海外" and abroad[10] is None
    assert parse_other_se(other_se("20240330C7010105", "06", "0000000000", kubun="B")) is None  # 外国馬
    assert parse_other_se(se()) is None and parse_other_ra(ra()) is None  # 中央は読まない


def test_import_other(tmp_path):
    import sqlite3
    write(tmp_path, "RACE.txt", [ra(), se(), other_ra("2024050644010301"),
                                 other_se("2024050644010301", "01", "2020101234", finish="01"),
                                 other_se("2024050644010301", "02", "2020105555", finish="", abnormal="4")])
    db = tmp_path / "o.db"
    conn = sqlite3.connect(db)
    assert import_other(conn, [tmp_path, tmp_path / "none"]) == 2
    assert import_other(conn, [tmp_path]) == 2  # 入れ直しても増えない
    conn.close()
    o = load_other(db)
    assert len(o) == 2 and set(o["field_size"]) == {2}
    row = o.set_index("horse_id").loc["2020105555"]
    assert row["status"] == "中" and np.isnan(row["finish"]) and row["course_type"] == "ダ" and row["distance"] == 1400


def _other(rows):
    o = pd.DataFrame(rows, columns=["race_id", "horse_id", "date", "finish", "field_size", "region"])
    o["date"] = pd.to_datetime(o["date"])
    return o


def test_add_other_features_uses_only_earlier_runs():
    df = pd.DataFrame({"horse_id": ["h1", "h1", "h1", "h2"],
                       "date": pd.to_datetime(["2024-01-10", "2024-06-01", "2024-07-01", "2024-03-01"])})
    o = _other([("o1", "h1", "2024-03-01", 1, 10, "地方"), ("o2", "h1", "2024-04-15", 4, 8, "地方"),
                ("o3", "h1", "2024-07-01", 1, 8, "地方"),  # 当日以後は使わない
                ("o4", "h3", "2024-01-01", 1, 8, "地方")])
    out = add_other_features(df, o)
    r = out.iloc[1]  # 中央(1/10) → 地方2走 → 中央(6/1)
    assert r["days_since_any"] == 47 and r["prev_is_other"] == 1.0
    assert r["prev_other_finish"] == 4 and r["prev_other_finish_rel"] == 0.5 and r["prev_other_abroad"] == 0.0
    assert r["n_other_prev"] == 2 and r["other_win_rate"] == 0.5 and r["runs_since_jra"] == 2 and r["n_any_prev"] == 3
    r = out.iloc[2]  # 前走は中央(6/1)
    assert r["prev_is_other"] == 0.0 and r["days_since_any"] == 30 and np.isnan(r["prev_other_finish"])
    assert r["n_other_prev"] == 2 and r["runs_since_jra"] == 0 and r["n_any_prev"] == 4
    first = out.iloc[0]
    assert np.isnan(first["days_since_any"]) and first["n_other_prev"] == 0 and first["n_any_prev"] == 0
    assert out.iloc[3]["n_other_prev"] == 0 and np.isnan(out.iloc[3]["prev_is_other"])


def test_transfer_from_local_has_history_on_first_jra_run():
    df = pd.DataFrame({"horse_id": ["h1"], "date": pd.to_datetime(["2024-05-01"])})
    o = _other([("o1", "h1", "2024-03-01", 1, 10, "地方"), ("o2", "h1", "2024-04-01", 2, 10, "地方")])
    r = add_other_features(df, o).iloc[0]
    assert r["prev_is_other"] == 1.0 and r["days_since_any"] == 30 and r["n_any_prev"] == 2
    assert r["runs_since_jra"] == 2 and r["other_top3_rate"] == 1.0 and r["is_layoff_any"] == 0.0


def test_features_from_db_know_the_overseas_run_and_the_jockey_gap():
    from keibaai import db
    from keibaai.features import features_from_db
    from keibaai.parse import ParsedRace, Race, Result

    conn = db.connect(":memory:")
    for i, (date, j1, j2) in enumerate([("2026-01-01", "J1", "J2"), ("2026-05-01", "J1", "J2"), ("2026-11-01", "J2", "J1")]):
        rid = f"20260501010{i + 1}"
        db.save_race(conn, ParsedRace(race=Race(race_id=rid, date=date, venue="東京", round=1, course_type="芝",
                                                distance=2000, name="テスト"), results=[
            Result(race_id=rid, number=1, finish=1, horse_id="h1", jockey_id=j1, sex_age="牡4"),
            Result(race_id=rid, number=2, finish=2, horse_id="h2", jockey_id=j2, sex_age="牡4"),
            Result(race_id=rid, number=3, finish=3, horse_id=f"x{i}", jockey_id="J3", sex_age="牡4"),
            Result(race_id=rid, number=4, finish=4, horse_id=f"y{i}", jockey_id="J2", sex_age="牡4")]))
    conn.execute("INSERT INTO other_runs (race_id, number, horse_id, date, region, venue_code, finish) "
                 "VALUES ('2026A6000005', 4, 'h1', '2026-07-25', '海外', 'A6', 7)")
    df = features_from_db(conn).set_index(["race_id", "number"])
    last = df.loc[("202605010103", 1)]
    assert last["prev_other_abroad"] == 1 and last["days_since_any"] == 99 and last["days_since_last"] == 184
    # 今回の騎手(J2: 4騎乗で3着内2回)と、前走の騎手(J1: 前走の時点で1騎乗1勝)の3着内率の差
    assert np.isclose(last["jockey_delta"], 0.5 - 1.0)
    assert np.isnan(df.loc[("202605010103", 2), "prev_other_abroad"])  # 前走が中央なら空
