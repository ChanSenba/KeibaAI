import datetime as dt

from keibaai import gui
from keibaai.parse import ParsedRace, Race, Result

from test_model import synthetic_db


def test_summary_and_tables():
    conn = synthetic_db(n_days=3, races_per_day=2)
    s = gui.summary(conn)
    assert s["races"] == 6 and s["results"] == 60 and 20 <= s["horses"] <= 60
    assert s["first"] == "2025-01-04" and s["last"] == "2025-01-18"
    assert gui.races_by_month(conn)["races"].sum() == 6
    assert gui.recent_races(conn, limit=3)["runners"].tolist() == [10, 10, 10]
    assert gui.date_bounds(conn) == (dt.date(2025, 1, 4), dt.date(2025, 1, 18))


def test_summary_empty():
    from keibaai import db
    conn = db.connect(":memory:")
    assert gui.summary(conn)["races"] == 0
    assert gui.date_bounds(conn) == (None, None)


def test_race_days_match_cli():
    # 2024-11-22(金)〜11-25(月): 土日月の3日
    days = gui.race_days(dt.date(2024, 11, 22), dt.date(2024, 11, 25))
    assert [d.isoformat() for d in days] == ["2024-11-23", "2024-11-24", "2024-11-25"]


class FakeFetcher:
    def race_ids_on(self, date):
        return [f"{date}01"]

    def race_html(self, race_id):
        return race_id


def test_collect_reports_progress(monkeypatch):
    from keibaai import cli, db
    monkeypatch.setattr(cli, "parse_race", lambda html, race_id: ParsedRace(
        Race(race_id=race_id, date="2024-11-23"), [Result(race_id=race_id, number=1, finish=1)]))
    conn = db.connect(":memory:")
    seen = []
    n = gui.collect(conn, FakeFetcher(), dt.date(2024, 11, 23), dt.date(2024, 11, 24),
                    lambda i, total, d: seen.append((i, total)))
    assert n == 2
    assert seen == [(0, 2), (1, 2), (2, 2)]


def test_train_reports_table():
    conn = synthetic_db()
    report = gui.train_reports(conn, "2026-07-01")
    assert report["モデル"].tolist()[0] .startswith("基準")
    assert len(report) == 3


def test_preset_range():
    mon = dt.date(2024, 11, 25)
    assert gui.preset_range("先週末", mon) == (dt.date(2024, 11, 23), dt.date(2024, 11, 25))
    wed = dt.date(2024, 11, 27)
    assert gui.preset_range("先週末", wed) == (dt.date(2024, 11, 23), dt.date(2024, 11, 25))
    sat = dt.date(2024, 11, 30)  # 当日はまだ結果が出ていないので前の週末
    assert gui.preset_range("先週末", sat) == (dt.date(2024, 11, 23), dt.date(2024, 11, 25))
    assert gui.preset_range("今年", wed) == (dt.date(2024, 1, 1), wed)
