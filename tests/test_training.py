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
