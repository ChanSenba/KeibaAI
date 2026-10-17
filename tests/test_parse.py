from pathlib import Path

import pytest

from keibaai.parse import parse_body_weight, parse_race, parse_time

HTML = (Path(__file__).parent / "fixtures" / "race_sample.html").read_text(encoding="utf-8")


def test_race_info():
    race = parse_race(HTML, "202405050812").race
    assert race.name == "第44回ジャパンC(GI)"
    assert (race.course_type, race.direction, race.distance) == ("芝", "左", 2400)
    assert (race.weather, race.ground, race.start_time) == ("晴", "良", "15:40")
    assert race.date == "2024-11-24"
    assert (race.venue, race.round) == ("東京", 12)


def test_winner_row():
    r = parse_race(HTML, "202405050812").results[0]
    assert (r.finish, r.bracket, r.number) == (1, 3, 3)
    assert r.horse_id == "2019105219" and r.horse_name == "ドウデュース"
    assert r.jockey_id == "01088" and r.trainer == "友道康夫"
    assert r.time_sec == pytest.approx(145.5)
    assert (r.last3f, r.win_odds, r.popularity) == (32.7, 2.3, 1)
    assert (r.body_weight, r.body_weight_diff) == (510, 6)
    assert r.prize == pytest.approx(50298.2)


def test_non_finisher_keeps_status():
    r = parse_race(HTML, "202405050812").results[1]
    assert r.finish is None and r.status == "中"
    assert r.time_sec is None and r.body_weight_diff == 0


def test_demoted_horse_keeps_official_finish():
    r = parse_race(HTML.replace('>中<', '>3(降)<'), "202405050812").results[1]
    assert r.finish == 3 and r.status == "3(降)"


def test_missing_table():
    with pytest.raises(ValueError):
        parse_race("<html></html>", "202405050812")


@pytest.mark.parametrize("s,exp", [("2:25.5", 145.5), ("1:09.0", 69.0), ("", None), ("**", None)])
def test_parse_time(s, exp):
    assert parse_time(s) == exp


@pytest.mark.parametrize("s,exp", [("510(+6)", (510, 6)), ("498(-12)", (498, -12)), ("計不", (None, None))])
def test_body_weight(s, exp):
    assert parse_body_weight(s) == exp


def test_steeplechase_course_text():
    html = HTML.replace("芝左2400m", "障芝ダート2910m")
    race = parse_race(html, "202408060704").race
    assert (race.course_type, race.direction, race.distance) == ("障", None, 2910)


@pytest.mark.parametrize("text,dist", [("障芝外-内3170m", 3170), ("障芝内-外3330m", 3330), ("芝右外2周3200m", None)])
def test_course_variants(text, dist):
    race = parse_race(HTML.replace("芝左2400m", text), "202408060704").race
    assert race.distance == (dist or 3200) and race.course_type in ("障", "芝")


def test_trainer_name_keeps_leading_character_that_looks_like_a_region():
    html = HTML.replace("[西] 友道康夫", "[東] 西村真幸")
    assert parse_race(html, "202405050812").results[0].trainer == "西村真幸"


@pytest.mark.parametrize("text,course", [("芝右 外1600m", "外"), ("障芝外-内3170m", "外-内"), ("芝右外2周3200m", "外2周"),
                                         ("芝左2400m", None)])
def test_course_inner_outer(text, course):
    assert parse_race(HTML.replace("芝左2400m", text), "202408060704").race.course == course
