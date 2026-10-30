import numpy as np
import pandas as pd

from keibaai import oddsmove


def _series(rows):
    s = pd.DataFrame(rows, columns=["race_id", "number", "t", "odds", "pop"])
    s["t"] = pd.to_datetime(s["t"])
    return s.sort_values("t", kind="stable").reset_index(drop=True)


def test_odds_move_uses_only_odds_published_before_each_cutoff():
    """発走 15:00。60・30・10分前の時点で最後に出ていたオッズと、確定オッズの比を取る。"""
    df = pd.DataFrame({"race_id": ["R", "R", "R"], "number": [1, 2, 3], "date": pd.Timestamp("2026-10-17"),
                       "win_odds": [2.0, 10.0, 20.0], "popularity": [1, 2, 3]})
    s = _series([
        ("R", 1, "2026-10-16 18:30", 4.0, 2), ("R", 1, "2026-10-17 13:50", 3.0, 2),  # 60分前までは 3.0
        ("R", 1, "2026-10-17 14:35", 2.5, 1), ("R", 1, "2026-10-17 14:55", 2.0, 1),  # 30分前は 3.0、10分前は 2.5
        ("R", 2, "2026-10-17 14:00", 10.0, 2), ("R", 2, "2026-10-17 14:56", 5.0, 1),  # 10分前は 10.0。14:56 は使わない
    ])
    out = oddsmove.add_odds_move(df, s, pd.Series({"R": "15:00"}))
    r1 = out.iloc[0]
    assert np.isclose(r1["om_60"], np.log(3.0 / 2.0)) and np.isclose(r1["om_30"], np.log(3.0 / 2.0))
    assert np.isclose(r1["om_10"], np.log(2.5 / 2.0)) and np.isclose(r1["om_open"], np.log(4.0 / 2.0))
    assert np.isclose(r1["om_late"], np.log(3.0 / 2.0) - np.log(2.5 / 2.0))
    assert r1["pop_chg_30"] == 1 - 2 and r1["om_n_ann"] == 4  # 人気は2位 → 1位(30分前は 2位)
    assert np.isclose(out.iloc[1]["om_10"], 0.0)  # 14:56 の 5.0 は 10 分前(14:50)より後なので見ない
    assert out.iloc[2][oddsmove.MOVE_COLUMNS].isna().all()  # 時系列オッズのない馬は欠損


def test_odds_move_is_missing_without_start_time():
    df = pd.DataFrame({"race_id": ["R"], "number": [1], "date": pd.Timestamp("2026-10-17"), "win_odds": [2.0], "popularity": [1]})
    s = _series([("R", 1, "2026-10-17 14:00", 3.0, 2)])
    out = oddsmove.add_odds_move(df, s, pd.Series({"R": None}))
    assert out[oddsmove.MOVE_COLUMNS].isna().all(axis=None)


def test_load_series_reads_only_time_series_rows(tmp_path):
    from keibaai import db

    path = tmp_path / "ts.db"
    conn = db.connect(path)
    conn.executemany("INSERT INTO odds_snapshots (race_id, number, snapped_at, win_odds, popularity, source) VALUES (?,?,?,?,?,?)",
                     [("R", 1, "2026-10-17T14:00:00", 3.0, 2, "ts"), ("R", 1, "2026-10-17T14:05:00", 2.9, 2, None),
                      ("R", 1, "2026-10-17T14:10:00", 0.0, 2, "ts")])
    conn.commit()
    conn.close()
    s = oddsmove.load_series(path, "2026-10-01")
    assert len(s) == 1 and s.iloc[0]["odds"] == 3.0
