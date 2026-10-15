import datetime as dt

import pandas as pd

from keibaai import db, weekly
from keibaai.parse import ParsedRace, Race, Result

from test_model import synthetic_db


def pred_table(race_id="202505010101", date="2025-03-01"):
    rows = []
    for i in range(1, 6):
        rows.append({"race_id": race_id, "date": pd.Timestamp(date), "venue": "東京", "round": 1, "race_name": "テスト特別",
                     "number": i, "horse_name": f"馬{i}", "jockey": "騎手", "win_odds": 2.0 * i, "popularity": i,
                     "p_win_ability": [0.4, 0.25, 0.2, 0.1, 0.05][i - 1], "p_win_market": None,
                     "ev_ability": [0.8, 1.25, 1.2, 0.8, 0.5][i - 1], "ev_market": None})
    return pd.DataFrame(rows)


def add_result(conn, race_id, date, order):
    race = Race(race_id=race_id, date=date, venue="東京", round=1, course_type="芝", distance=1600, ground="良")
    results = [Result(race_id=race_id, number=n, finish=f, win_odds=2.0 * n, popularity=n, horse_id=f"h{n}")
               for f, n in enumerate(order, start=1)]
    db.save_race(conn, ParsedRace(race=race, results=results))


def test_save_is_idempotent_and_frozen_after_race_day_morning():
    conn = db.connect(":memory:")
    assert weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 27, 20)) == 5
    t = pred_table(); t["win_odds"] = 99.0
    weekly.save_predictions(conn, t, dt.datetime(2025, 3, 1, 8))  # 当日朝: 更新できる
    assert conn.execute("SELECT win_odds FROM predictions LIMIT 1").fetchone()[0] == 99.0
    t["win_odds"] = 1.0
    assert weekly.save_predictions(conn, t, dt.datetime(2025, 3, 1, 13)) == 0  # 当日昼: 凍結
    assert weekly.save_predictions(conn, t, dt.datetime(2025, 3, 2, 8)) == 0  # 翌日以降も凍結
    assert conn.execute("SELECT win_odds FROM predictions LIMIT 1").fetchone()[0] == 99.0


def test_no_save_when_result_exists():
    conn = db.connect(":memory:")
    add_result(conn, "202505010101", "2025-03-01", [1, 2, 3, 4, 5])
    assert weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 3, 1, 7)) == 0


def test_review_matches_predictions_with_results():
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    assert weekly.review_day(conn, "2025-03-01").summary["races"] == 0  # 結果が出る前
    add_result(conn, "202505010101", "2025-03-01", [3, 1, 2, 5, 4])  # 3番が勝ち、◎(1番)は2着
    r = weekly.review_day(conn, "2025-03-01")
    s = r.summary
    assert s["races"] == 1 and s["honmei_wins"] == 0 and s["top3_hits"] == 3  # ◎○▲(1,2,3番)が全員3着以内
    assert s["ev_bets"] == 2 and s["ev_wins"] == 1 and s["ev_roi"] == 100 * 6.0 / 2  # 3番(6倍)が勝ち
    x = r.races.iloc[0]
    assert x["winner"] == "馬3" and x["winner_rank"] == 3 and x["honmei"] == "馬1" and x["honmei_finish"] == 2
    md = weekly.render_markdown(r)
    assert "2025-03-01 回顧" in md and "勝ち馬は馬3(予想3位" in md


def test_unreviewed_dates_and_upcoming_weekend(tmp_path):
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    assert weekly.unreviewed_dates(conn, dt.date(2025, 3, 3), tmp_path) == ["2025-03-01"]
    (tmp_path / "2025-03-01.md").write_text("x")
    assert weekly.unreviewed_dates(conn, dt.date(2025, 3, 3), tmp_path) == []
    assert weekly.unreviewed_dates(conn, dt.date(2025, 3, 1), tmp_path) == []  # 当日はまだ
    up = weekly._upcoming_weekend
    assert up(dt.date(2026, 10, 14)) == [dt.date(2026, 10, 17), dt.date(2026, 10, 18)]
    assert up(dt.date(2026, 10, 17)) == [dt.date(2026, 10, 17), dt.date(2026, 10, 18)]
    assert up(dt.date(2026, 10, 18)) == [dt.date(2026, 10, 18)]


class FakeFetcher:
    def __init__(self, conn=None):
        self.conn = conn

    def race_ids_on(self, date):
        return []


def test_run_weekly_reviews_then_forecasts(tmp_path, monkeypatch):
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    add_result(conn, "202505010101", "2025-03-01", [3, 1, 2, 5, 4])
    monkeypatch.setattr(weekly, "forecast_day", lambda conn, f, d, now, jv=None: 0)
    out = weekly.run_weekly(conn, FakeFetcher(), dt.date(2025, 3, 3), tmp_path / "r", now=dt.datetime(2025, 3, 3, 21))
    assert out["reviewed"] == ["2025-03-01"] and (tmp_path / "r" / "2025-03-01.md").exists()
    assert list(out["forecast"]) == ["2025-03-08", "2025-03-09"]
    out = weekly.run_weekly(conn, FakeFetcher(), dt.date(2025, 3, 4), tmp_path / "r", now=dt.datetime(2025, 3, 4, 21))
    assert out["reviewed"] == []  # 済んだ日は飛ばす


def test_weekly_command_runs_run_weekly(tmp_path, monkeypatch, capsys):
    from keibaai import cli

    seen = {}

    def fake(conn, fetcher, today, reviews_dir, jv=None):
        seen.update(today=today, reviews=str(reviews_dir), jv=jv)
        return {"reviewed": ["2025-03-01"], "forecast": {"2025-03-08": 0, "2025-03-09": 12}}

    monkeypatch.setattr(weekly, "run_weekly", fake)
    cli.main(["weekly", "--db", str(tmp_path / "k.db"), "--cache", str(tmp_path / "c"),
              "--reviews", str(tmp_path / "r"), "--today", "2025-03-03", "--source", "netkeiba"])
    out = capsys.readouterr().out
    assert seen["today"] == dt.date(2025, 3, 3) and "reviewed: 2025-03-01" in out and seen["jv"] is None
    assert "forecast 2025-03-08: 出馬表がまだありません" in out and "forecast 2025-03-09: 12頭" in out


def test_weekly_with_jv_takes_results_and_entries_from_jravan(tmp_path, monkeypatch):
    from keibaai import jvlink

    calls = []
    monkeypatch.setattr(weekly, "unreviewed_dates", lambda conn, today, d: ["2025-03-01"])
    monkeypatch.setattr(weekly, "review_day", lambda conn, d: weekly.Review(d, None, None, {"races": 0}))
    monkeypatch.setattr(jvlink, "fetch_results", lambda conn, s, e, raw, py, today=None: calls.append(("results", s)))
    monkeypatch.setattr(jvlink, "fetch_entries", lambda conn, d, odds, raw, py, now=None: calls.append(("entries", d)) or 0)
    conn = db.connect(":memory:")
    out = weekly.run_weekly(conn, None, dt.date(2025, 3, 4), tmp_path / "r", now=dt.datetime(2025, 3, 4, 21),
                            jv=("raw", "py32"))
    assert calls == [("results", dt.date(2025, 3, 1)), ("entries", "2025-03-08"), ("entries", "2025-03-09")]
    assert out["forecast"] == {"2025-03-08": 0, "2025-03-09": 0}
