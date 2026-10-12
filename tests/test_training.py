from pathlib import Path

import pytest

from keibaai.training import parse_oikiri

HTML = (Path(__file__).parent / "fixtures" / "oikiri_sample.html").read_text(encoding="utf-8")


def test_parse_oikiri_rows_per_horse():
    runs = parse_oikiri(HTML, "202608040511")
    assert [(r.number, r.seq) for r in runs] == [(1, 0), (1, 1), (1, 2), (2, 0)]
    first = runs[0]
    assert first.horse_id == "2023105685"
    assert first.date == "2026-10-08" and first.course == "ＣＷ" and first.slope == 0
    assert first.ground == "良" and first.rider == "助手"
    assert (first.f6, first.f5, first.f4, first.f3, first.f1) == (None, None, 51.8, 36.4, 11.2)
    assert first.position == 5 and first.load == "馬也"
    assert first.critic == "気配上昇" and first.rank == "B"
    assert first.review.startswith("ひと叩きして")
    assert runs[1].review is None


def test_parse_oikiri_slope_and_partner():
    runs = parse_oikiri(HTML, "202608040511")
    slope = runs[1]
    assert slope.course == "栗坂" and slope.slope == 1
    assert (slope.f4, slope.f3, slope.f2, slope.f1) == (55.4, 40.0, 25.9, 13.2)
    assert slope.load == "一杯"
    assert "ロングトールサリー" in slope.partner and "遅れ" in slope.partner
    assert runs[2].partner.count("先着") == 2
    assert runs[3].rank == "A" and runs[3].position is None and runs[3].horse_id == "2022105071"


def test_parse_oikiri_without_table():
    with pytest.raises(ValueError):
        parse_oikiri("<html><body>no table</body></html>", "x")


NOTYPE = (Path(__file__).parent / "fixtures" / "oikiri_notype_sample.html").read_text(encoding="utf-8")


def test_parse_oikiri_evaluation_only_page():
    runs = parse_oikiri(NOTYPE, "202608040511")
    assert [(r.number, r.seq, r.horse_id, r.critic, r.rank) for r in runs] == [
        (1, 0, "2023105685", "気配上昇", "B"), (2, 0, "2022105071", None, None), (3, 0, "2020110155", "前走並み", "C")]
    assert runs[0].date is None and runs[0].course is None and runs[0].f1 is None and runs[0].load is None


class FakeFetcher:
    def __init__(self, pages):
        self.pages, self.calls, self.discarded = pages, [], []

    def oikiri_html(self, race_id, cache=True):
        self.calls.append((race_id, cache))
        page = self.pages[race_id]
        if isinstance(page, Exception):
            raise page
        return page

    def discard_oikiri(self, race_id):
        self.discarded.append(race_id)


def _db_with_races(ids_dates):
    from keibaai import db
    conn = db.connect(":memory:")
    for rid, d in ids_dates:
        conn.execute("INSERT INTO races (race_id, date) VALUES (?, ?)", (rid, d))
    conn.commit()
    return conn


def test_collect_training_newest_first_and_resumable():
    from keibaai.fetch import PageUnavailable
    from keibaai.training import collect_training, pending_race_ids

    conn = _db_with_races([("A", "2024-01-01"), ("B", "2025-01-01"), ("C", "2023-01-01")])
    f = FakeFetcher({"B": NOTYPE, "A": PageUnavailable("no table"), "C": "<html>OikiriTable but broken</html>"})
    assert collect_training(conn, f, log=lambda *_: None) == 1
    assert [c[0] for c in f.calls] == ["B", "A", "C"]
    assert conn.execute("SELECT COUNT(*) FROM training WHERE race_id='B'").fetchone()[0] == 3
    # 表のないレースは読んだ印が付いて取り直さない。読めなかったページはキャッシュを捨て、次回また試す
    assert pending_race_ids(conn) == ["C"] and f.discarded == ["C"]


def test_collect_training_stops_after_consecutive_missing():
    from keibaai.fetch import PageUnavailable
    from keibaai.training import collect_training

    conn = _db_with_races([(f"R{i}", f"2024-01-0{i}") for i in range(1, 7)])
    f = FakeFetcher({f"R{i}": PageUnavailable("x") for i in range(1, 7)})
    with pytest.raises(RuntimeError):
        collect_training(conn, f, log=lambda *_: None)
    assert len(f.calls) == 5


def test_entry_training_is_not_cached():
    from keibaai.training import fetch_entry_training

    conn = _db_with_races([])
    conn.execute("INSERT INTO entry_races (race_id, date) VALUES ('E1', '2026-10-17')")
    f = FakeFetcher({"E1": NOTYPE})
    assert fetch_entry_training(conn, f, "2026-10-17", log=lambda *_: None) == 1
    assert f.calls == [("E1", False)]
