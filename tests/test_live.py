import numpy as np
import pytest

from keibaai import db
from keibaai.live import clv_table, race_clv


_n = 0


def _db(tmp_path, saved_at="2026-11-14T21:00:00", after_result=0):
    global _n
    _n += 1
    conn = db.connect(tmp_path / f"t{_n}.db")
    with conn:
        conn.execute("INSERT INTO races (race_id, date, start_time) VALUES ('r1', '2026-11-15', '15:40')")
        # 保存時点: 1番が2.0倍。確定では1.6倍に下がった(市場が寄った) → ◎の CLV はプラス
        conn.executemany("INSERT INTO results (race_id, number, finish, win_odds) VALUES (?,?,?,?)",
                         [("r1", 1, 1, 1.6), ("r1", 2, 2, 4.0), ("r1", 3, 3, 8.0)])
        conn.executemany(
            "INSERT INTO predictions (race_id, number, saved_at, date, win_odds, mark, ev_market, after_result) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [("r1", 1, saved_at, "2026-11-15", 2.0, "◎", 0.9, after_result),
             ("r1", 2, saved_at, "2026-11-15", 4.0, "", 1.2, after_result),
             ("r1", 3, saved_at, "2026-11-15", 8.0, "", 0.5, after_result)])
    return conn


def test_clv_positive_when_market_moves_toward_pick(tmp_path):
    conn = _db(tmp_path)
    rc = race_clv(conn)
    honmei = rc[rc["対象"] == "◎"].iloc[0]
    assert honmei["clv"] > 0
    saved = np.array([1 / 2.0, 1 / 4.0, 1 / 8.0])
    final = np.array([1 / 1.6, 1 / 4.0, 1 / 8.0])
    expect = np.log((final / final.sum())[0] / (saved / saved.sum())[0])
    assert honmei["clv"] == pytest.approx(expect)
    table = clv_table(conn)
    assert set(table["対象"]) == {"◎", "期待値≧1", "1番人気", "全馬"}  # 基準の行も一緒に出る
    assert table.loc[table["対象"] == "◎", "平均CLV(/1000)"].iloc[0] > 0


def test_clv_includes_star_group_when_star_saved(tmp_path):
    conn = _db(tmp_path)
    with conn:
        conn.execute("UPDATE predictions SET star = '★' WHERE number = 2")
    table = clv_table(conn)
    star = table[table["対象"] == "★"]
    assert len(star) == 1 and star.iloc[0]["頭数"] == 1
    # 2番は保存時4.0倍→確定4.0倍。保存時と確定の市場確率の比が CLV
    saved = np.array([1 / 2.0, 1 / 4.0, 1 / 8.0])
    final = np.array([1 / 1.6, 1 / 4.0, 1 / 8.0])
    assert star.iloc[0]["平均CLV(/1000)"] == pytest.approx(round(np.log((final / final.sum())[1] / (saved / saved.sum())[1]) * 1000, 1))


def test_clv_has_no_star_group_when_star_not_saved(tmp_path):
    table = clv_table(_db(tmp_path))  # star は NULL のまま
    assert "★" not in set(table["対象"])


def test_clv_excludes_after_result_and_late_saves(tmp_path):
    assert race_clv(_db(tmp_path, after_result=1)).empty
    assert race_clv(_db(tmp_path, saved_at="2026-11-16T09:00:00")).empty  # 翌日の保存
    assert race_clv(_db(tmp_path, saved_at="2026-11-15T16:00:00")).empty  # 発走(15:40)より後
    assert not race_clv(_db(tmp_path, saved_at="2026-11-15T09:00:00")).empty  # 当日朝は発走前


def test_clv_start_time_with_single_digit_hour(tmp_path):
    conn = _db(tmp_path, saved_at="2026-11-15T10:30:00")
    with conn:
        conn.execute("UPDATE races SET start_time = '9:55' WHERE race_id = 'r1'")
    assert race_clv(conn).empty  # 9:55 発走のあと 10:30 の保存は、発走前と間違えない


def test_clv_start_time_unknown_uses_date_only(tmp_path):
    conn = _db(tmp_path, saved_at="2026-11-15T16:00:00")
    with conn:
        conn.execute("UPDATE races SET start_time = NULL WHERE race_id = 'r1'")
    assert not race_clv(conn).empty  # 発走時刻が分からなければ、日付だけで判定する(当日中は含める)
