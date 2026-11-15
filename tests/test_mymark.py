from keibaai.mymark import mark_url


def test_mark_url_encodes_marks():
    url = mark_url("202605050811", {3: "◎", 7: "○", 1: "▲", 5: "△", 9: "△", 12: "☆"})
    assert url == ("https://race.netkeiba.com/race/shutuba.html?race_id=202605050811"
                   "#keibaai=3-1,7-2,1-3,5-4,9-4,12-5")


def test_mark_url_skips_blank_and_unknown():
    assert mark_url("202605050811", [(2, ""), (4, "★"), (6.0, "◎")]).endswith("#keibaai=6-1")
    assert mark_url("202605050811", {}) == "https://race.netkeiba.com/race/shutuba.html?race_id=202605050811"
