import datetime as dt
import os

from keibaai import db, progress


def _touch(path, age_sec, now):
    path.write_text("x", encoding="utf-8")
    t = now - age_sec
    os.utime(path, (t, t))


def test_scan_counts_files_last_update_and_recent(tmp_path):
    now = dt.datetime.now().timestamp()
    folder = tmp_path / "oikiri"
    folder.mkdir()
    _touch(folder / "a.html", 60, now)
    _touch(folder / "b.html", 600, now)
    _touch(folder / "c.html", 7200, now)
    (folder / "note.txt").write_text("x", encoding="utf-8")
    n, last, recent = progress.scan_folder(folder, now)
    assert (n, recent) == (3, 2)
    assert abs(last - (now - 60)) < 1


def test_missing_folder_is_not_started(tmp_path):
    p = progress.progress_of("training", tmp_path, 100)
    assert (p.done, p.last_saved, p.alive, p.eta_hours, p.fraction) == (0, None, False, None, 0.0)


def test_rate_eta_and_alive(tmp_path):
    now = dt.datetime.now().timestamp()
    folder = tmp_path / "oikiri"
    folder.mkdir()
    for i in range(30):  # 直近30分に30件 = 1件/分
        _touch(folder / f"{i}.html", 30 + i * 50, now)
    p = progress.progress_of("training", tmp_path, 90, now)
    assert p.done == 30 and abs(p.per_min - 1.0) < 1e-9
    assert p.alive and abs(p.fraction - 30 / 90) < 1e-9
    assert abs(p.eta_hours - 1.0) < 1e-9  # 残り60件 ÷ 1件/分 = 1時間


def test_stale_cache_is_reported_as_stopped(tmp_path):
    now = dt.datetime.now().timestamp()
    folder = tmp_path / "ped"
    folder.mkdir()
    _touch(folder / "h.html", 86400, now)
    p = progress.progress_of("pedigree", tmp_path, None, now)
    assert not p.alive and p.per_min == 0 and p.fraction is None and p.eta_hours is None


def test_total_is_read_only_and_tolerates_a_missing_db(tmp_path):
    path = tmp_path / "k.db"
    conn = db.connect(path)
    conn.execute("INSERT INTO races (race_id, date) VALUES ('A', '2024-01-01'), ('B', '2024-01-02')")
    conn.commit()
    conn.close()
    assert progress.read_total(path, progress.KINDS["training"][2]) == 2
    assert progress.read_total(tmp_path / "nothing.db", progress.KINDS["training"][2]) is None


def test_tail(tmp_path):
    log = tmp_path / "out.log"
    log.write_text("\n".join(f"line{i}" for i in range(20)), encoding="utf-8")
    assert progress.tail(log, 3) == "line17\nline18\nline19"
    assert progress.tail(tmp_path / "none.log") == ""


def test_data_status_counts_pedigree_coverage_from_the_db(tmp_path):
    path = tmp_path / "k.db"
    conn = db.connect(path)
    conn.execute("INSERT INTO races (race_id, date) VALUES ('A', '2026-10-12')")
    conn.executemany("INSERT INTO results (race_id, number, horse_id) VALUES ('A', ?, ?)",
                     [(1, "H1"), (2, "H2"), (3, "H3"), (4, None)])
    conn.executemany("INSERT INTO pedigree (horse_id) VALUES (?)", [("H1",), ("H2",), ("X",)])
    conn.execute("INSERT INTO jv_training (horse_id, date, time, kind) VALUES ('H1', '2026-10-15', '0600', 'slope')")
    conn.commit()
    conn.close()
    s = progress.read_data_status(path)
    assert (s.races, s.last_race_date, s.horses, s.horses_with_pedigree) == (1, "2026-10-12", 3, 2)
    assert abs(s.pedigree_fraction - 2 / 3) < 1e-9
    assert (s.training_rows, s.last_training_date) == (1, "2026-10-15")
    assert progress.read_data_status(tmp_path / "nothing.db") is None

def test_odds_status_stalled_only_during_race_hours():
    day = dt.datetime(2026, 10, 18)
    starts = ["10:05", "12:25", "15:40"]
    at = lambda h, m: day + dt.timedelta(hours=h, minutes=m)
    # 開催中に直近の記録がある → 動いている
    assert progress.odds_status_from(starts, 3, at(12, 5), at(12, 20)).state == "running"
    # 開催中に1時間以上記録がない → 止まっている
    assert progress.odds_status_from(starts, 3, at(10, 0), at(12, 20)).state == "stalled"
    # 開催中で今日の記録がまったくない(前日の記録だけ)。見張りの開始(最初の発走の1時間前)から1時間たてば止まっている
    prev = at(-5, 0)
    assert progress.odds_status_from(starts, 0, prev, at(9, 30)).state == "running"
    assert progress.odds_status_from(starts, 0, prev, at(10, 10)).state == "stalled"
    # 開催前・終了後・非開催日 → 待機中
    assert progress.odds_status_from(starts, 0, prev, at(7, 0)).state == "idle"
    assert progress.odds_status_from(starts, 3, at(15, 35), at(18, 0)).state == "idle"
    assert progress.odds_status_from([], 0, None, at(12, 0)).state == "idle"


def test_odds_status_reads_the_db_and_tolerates_missing_table(tmp_path):
    path = tmp_path / "k.db"
    conn = db.connect(path)
    conn.execute("INSERT INTO entry_races (race_id, date, start_time) VALUES ('A', '2026-10-18', '10:05'), ('B', '2026-10-18', '12:25')")
    conn.execute("INSERT INTO odds_snapshots (race_id, number, snapped_at, win_odds) VALUES ('A', 1, '2026-10-18T12:05:00', 3.0)")
    conn.commit()
    conn.close()
    s = progress.read_odds_status(path, dt.datetime(2026, 10, 18, 12, 20))
    assert (s.state, s.races_today, s.recorded_today, s.last_snapped) == ("running", 2, 1, dt.datetime(2026, 10, 18, 12, 5))
    assert progress.read_odds_status(tmp_path / "nothing.db", dt.datetime(2026, 10, 18, 12, 20)) is None
