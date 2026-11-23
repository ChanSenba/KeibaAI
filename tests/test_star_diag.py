from keibaai import db, star_diag


def _insert(conn, race_id, date, saved_at, rows):
    """rows は (馬番, p_win_market, ev_market, 単勝オッズ) の並び。"""
    conn.executemany(
        "INSERT INTO predictions (race_id, date, saved_at, number, p_win_market, ev_market, win_odds) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(race_id, date, saved_at, n, pm, ev, odds) for n, pm, ev, odds in rows])
    conn.commit()


def _sample_conn():
    conn = db.connect(":memory:")
    # 前日に保存したレース: オッズがなく、p_win_market は全頭 NULL
    _insert(conn, "R1", "2025-03-01", "2025-02-28T20:00:00", [(i, None, None, 3.0) for i in range(1, 4)])
    # 当日朝に保存したレース: オッズがそろい、1番が★の基準(期待値1.1以上・勝率3%以上・30倍以内)を満たす
    _insert(conn, "R2", "2025-03-01", "2025-03-01T08:00:00",
            [(1, 0.30, 1.25, 4.0), (2, 0.25, 0.9, 3.6), (3, 0.2, 0.7, 3.5)])
    return conn


def test_save_band():
    assert star_diag.save_band("2025-02-28T20:00:00", "2025-03-01") == "前日以前"
    assert star_diag.save_band("2025-03-01T08:00:00", "2025-03-01") == "当日・9時前"
    assert star_diag.save_band("2025-03-01T09:00:00", "2025-03-01") == "当日・9時以降"
    assert star_diag.save_band("2025-03-02T10:00:00", "2025-03-01") == "翌日以降"
    assert star_diag.save_band(None, "2025-03-01") == "不明"
    assert star_diag.save_band("壊れた値", "2025-03-01") == "不明"


def test_band_and_criteria_stats():
    df = star_diag.load_predictions(_sample_conn())
    bands = star_diag.band_stats(df).set_index("帯")
    assert bands.loc["前日以前", "オッズなし率"] == 1.0 and bands.loc["前日以前", "レース数"] == 1
    assert bands.loc["当日・9時前", "オッズなし率"] == 0.0
    crit = star_diag.criteria_stats(df)
    assert crit["races"] == 1 and crit["horses"] == 3  # オッズがそろったのは R2 だけ
    assert crit["quantiles"][1.0] == 1.25  # レースごとの ev_market 最大値
    assert crit["hit_horses"] == 1 and crit["hit_races"] == 1 and crit["hit_race_rate"] == 1.0


def test_markdown_report_reading_guide():
    df = star_diag.load_predictions(_sample_conn())
    md = star_diag.markdown_report(df)
    assert "## 読み方" in md and "保存タイミングの問題" in md and "モデル・基準側の問題" in md
    assert "前日以前" in md and "当日・9時前" in md and "1.25" in md


def test_main_reads_db_readonly(tmp_path, capsys):
    path = tmp_path / "keiba.db"
    conn = db.connect(path)
    _insert(conn, "R2", "2025-03-01", "2025-03-01T08:00:00",
            [(1, 0.30, 1.25, 4.0), (2, 0.25, 0.9, 3.6), (3, 0.2, 0.7, 3.5)])
    conn.close()
    before = path.read_bytes()
    star_diag.main(["--db", str(path)])
    out = capsys.readouterr().out
    assert "★の切り分け診断" in out and "読み方" in out
    assert path.read_bytes() == before  # 読み取り専用なので、DB ファイルは変わらない


def test_no_full_odds_races():
    conn = db.connect(":memory:")
    _insert(conn, "R1", "2025-03-01", "2025-02-28T20:00:00", [(i, None, None, 3.0) for i in range(1, 4)])
    df = star_diag.load_predictions(conn)
    crit = star_diag.criteria_stats(df)
    assert crit["races"] == 0 and crit["quantiles"] == {}
    assert "オッズがそろったレースがありません" in star_diag.markdown_report(df)
