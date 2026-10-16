import sqlite3

from keibaai import jrababa

ROWS = """２０２６年 第３回 東京競馬 クッション値・含水率一覧
開催日次 測定月日 曜日 芝コース（%） ダートコース（%）
6月 5日 金曜日 C 09:00 9.8 08:30 14.9 15.1 7.9 8.8
第 1日 6月 6日 土曜日 C 07:00 9.9 05:00 16.2 15.7 7.7 8.9
第 2日 6月 7日 日曜日 C 07:00 9.9 05:30 15.4 13.6 3.6 5.3
"""

WEEKS = """2018年 第4回阪神競馬 含水率
第８日・第９日の含水率
2018年9月28日から10月2日の含水率
場所 金曜日 土曜日 火曜日
芝コース含水率 ゴール前 10.2 13.1 10.9
（パーセント） ４コーナー 9.0 11.6 9.8
ダートコース含水率 ゴール前 12.1 17.3 11.7
（パーセント） ４コーナー 12.2 17.6 11.2
第１日・第２日の含水率
2018年7月27日（金）から29日（日）
場所 金曜日 土曜日 日曜日
芝コース含水率 ゴール前 11.7 12.6 12.4
（パーセント） ４コーナー 10.8 13.4 11.9
ダートコース含水率 ゴール前 2.8 2.8 2.3
（パーセント） ４コーナー 2.3 2.6 2.8
"""


def test_parse_rows_with_cushion():
    rows = jrababa.parse_text(ROWS, 2026)
    assert [r["date"] for r in rows] == ["2026-06-05", "2026-06-06", "2026-06-07"]
    assert rows[0]["day_no"] is None and rows[1]["day_no"] == 1
    r = rows[1]
    assert (r["course"], r["cushion_time"], r["cushion"], r["moisture_time"]) == ("C", "07:00", 9.9, "05:00")
    assert (r["turf_goal"], r["turf_4c"], r["dirt_goal"], r["dirt_4c"]) == (16.2, 15.7, 7.7, 8.9)
    assert not any("_bad" in r for r in rows)


def test_parse_rows_without_cushion():
    rows = jrababa.parse_rows("第 1日 9月 5日 土曜日 05:00 16.2 15.7 7.7 8.9", 2020)
    assert rows[0]["cushion"] is None and rows[0]["moisture_time"] == "05:00" and rows[0]["dirt_4c"] == 8.9


def test_parse_weeks_follows_weekdays():
    rows = jrababa.parse_text(WEEKS, 2018)
    dates = [(r["date"], r["weekday"]) for r in rows]
    assert dates[:3] == [("2018-09-28", "金"), ("2018-09-29", "土"), ("2018-10-02", "火")]  # 日曜の開催が火曜に延びた
    assert dates[3:] == [("2018-07-27", "金"), ("2018-07-28", "土"), ("2018-07-29", "日")]
    assert (rows[2]["turf_goal"], rows[2]["turf_4c"], rows[2]["dirt_goal"], rows[2]["dirt_4c"]) == (10.9, 9.8, 11.7, 11.2)
    assert not any("_bad" in r for r in rows)


def test_import_pdfs(tmp_path, monkeypatch):
    (tmp_path / "keiba_baba_archive_2026pdf_tokyo03.pdf").write_bytes(b"")
    (tmp_path / "other.pdf").write_bytes(b"")
    monkeypatch.setattr(jrababa, "pdf_text", lambda p: ROWS)
    conn = sqlite3.connect(":memory:")
    s = jrababa.import_pdfs(conn, tmp_path)
    assert s["pdfs"] == 1 and s["rows"] == 3 and not s["bad_rows"]
    assert conn.execute("SELECT venue, cushion FROM baba WHERE date='2026-06-06'").fetchone() == ("東京", 9.9)
    assert conn.execute("SELECT COUNT(*) FROM baba_long").fetchone()[0] == 12


BLOCKS = """2021年 第1回中京競馬 含水率・クッション値
第１日（2021年1月4日～5日）
月曜日 火曜日
芝コースクッション値 10.5 10.0
場所 月曜日 火曜日
芝コース含水率 ゴール前 13.7 12.7
（パーセント） ４コーナー 10.8 10.1
ダートコース含水率 ゴール前 4.2 4.2
（パーセント） ４コーナー 4.9 4.7
第２日・第３日（2020年10月9日～11日）
場所 金曜日 土曜日 日曜日
芝コース含水率 ゴール前 19.0 20.5 19.7
（パーセント） ４コーナー 17.6 19.5 19.0
ダートコース含水率 ゴール前 16.5 16.7 13.5
（パーセント） ４コーナー 15.8 16.0 13.1
金曜日 土曜日 日曜日
芝コースクッション値 8.5 7.0 7.8
"""


def test_parse_weeks_with_cushion_before_or_after():
    rows = jrababa.parse_text(BLOCKS, 2021)
    assert [(r["date"], r["cushion"], r["turf_goal"], r["dirt_4c"]) for r in rows] == [
        ("2021-01-04", 10.5, 13.7, 4.9), ("2021-01-05", 10.0, 12.7, 4.7),
        ("2020-10-09", 8.5, 19.0, 15.8), ("2020-10-10", 7.0, 20.5, 16.0), ("2020-10-11", 7.8, 19.7, 13.1)]
    assert not any("_bad" in r for r in rows)
