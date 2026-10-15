from test_jvrace import ra, record, se, write

from keibaai import db
from keibaai.jventries import import_entries, parse_o1, race_keys, read_entries

RACE = "2026101708040511"  # 10/17 京都 4回5日 11R → race_id 202608040511


def o1(race=RACE, wins=(("01", "0061", "02"), ("02", "0228", "10"), ("03", "----", "**"))):
    f = {0: "O1", 2: "1", 3: "20261017", 11: race, 39: "7"}
    for i, (number, odds, pop) in enumerate(wins):
        f[43 + 8 * i] = number + odds + pop
    return record(962, f)


def card(kubun="2"):
    """出馬表の段階のレース1つと出走馬3頭(3番は取消)。"""
    pre = {2: kubun, 334: "00", 338: "0000", 351: "00000000", 359: "0000", 363: "00", 365: "00000000", 390: "000",
           324: "   ", 327: " ", 328: "   "}
    return [ra(race=RACE, kubun=kubun, title="スワンステークス"),
            se(race=RACE, number="01", over=pre), se(race=RACE, number="02", over={**pre, 40: "スズハローム"}),
            se(race=RACE, number="03", abnormal="1", over=pre)]


def test_parse_win_odds():
    race_id, odds = parse_o1(o1())
    assert race_id == "202608040511"
    assert odds == {1: (6.1, 2), 2: (22.8, 10), 3: (None, None)}  # 取消の馬はオッズなし
    assert parse_o1(card()[0]) is None


def test_read_entries_before_the_race(tmp_path):
    write(tmp_path, "RT_0B15.txt", card() + [ra(race="2026101805040601", kubun="2")])
    write(tmp_path, "RT_0B31.txt", [o1()])
    assert race_keys(tmp_path, "2026-10-17") == [RACE]
    (p,) = read_entries(tmp_path, "2026-10-17")  # 別の日のレースと、出走馬のいないレースは入らない
    assert (p.race.race_id, p.race.name, p.race.date, p.race.venue, p.race.round) == (
        "202608040511", "スワンステークス(GII)", "2026-10-17", "京都", 11)
    assert (p.race.course_type, p.race.distance, p.race.start_time) == ("芝", 1600, "15:45")
    assert [e.number for e in p.entries] == [1, 2]  # 取消の3番は入れない
    e = p.entries[0]
    assert (e.horse_id, e.horse_name, e.sex_age, e.weight_carried, e.jockey_id, e.trainer_id) == (
        "2024105705", "アークレイリ", "牡2", 56.0, "01091", "01141")
    assert (e.body_weight, e.body_weight_diff) == (None, None)  # 馬体重の発表前
    assert (e.win_odds, e.popularity) == (6.1, 2)
    assert (p.entries[1].horse_name, p.entries[1].win_odds) == ("スズハローム", 22.8)


def test_weekly_data_card_is_read_and_the_later_stage_wins(tmp_path):
    """木曜〜金曜の「今週のデータ」(RACE.txt)からも読む。出走馬名表(1)のあと出馬表(2)が来たら出馬表を使う。"""
    names = [ra(race=RACE, kubun="1", title="スワンステークス"), se(race=RACE, number="00", over={2: "1"}),
             se(race=RACE, number="00", over={2: "1", 30: "2024100000"})]
    write(tmp_path, "RACE.txt", names)
    assert race_keys(tmp_path, "2026-10-17") == [RACE]  # 馬番の前でも、オッズを取るレースは分かる
    assert read_entries(tmp_path, "2026-10-17") == []  # 木曜: 馬番がないので、まだ入れない
    write(tmp_path, "RACE.txt", names + card())  # 枠順が出た(区分 2)
    (p,) = read_entries(tmp_path, "2026-10-17")
    assert [e.number for e in p.entries] == [1, 2]
    write(tmp_path, "RT_0B15.txt", [ra(race=RACE, kubun="2", title="スワンステークス"),
                                    se(race=RACE, number="05", over={2: "2"})])  # 速報にも同じ区分がある
    (p,) = read_entries(tmp_path, "2026-10-17")
    assert [e.number for e in p.entries] == [1, 2, 5]  # 同じ区分なら両方の馬を合わせる


def test_entries_without_numbers_or_odds(tmp_path):
    rows = card()
    write(tmp_path, "RT_0B15.txt", [rows[0], se(race=RACE, number="00", over={2: "1"})])  # 木曜: 馬番の確定前
    assert read_entries(tmp_path, "2026-10-17") == []
    write(tmp_path, "RT_0B15.txt", rows)
    (p,) = read_entries(tmp_path, "2026-10-17")  # オッズのファイルがなくても出馬表は読める
    assert [e.win_odds for e in p.entries] == [None, None]


def test_import_entries_replaces_the_race(tmp_path):
    conn = db.connect(":memory:")
    write(tmp_path, "RT_0B15.txt", card())
    assert import_entries(conn, tmp_path, "2026-10-17", "2026-10-16T19:00:00") == 1
    write(tmp_path, "RT_0B31.txt", [o1()])
    assert import_entries(conn, tmp_path, "2026-10-17", "2026-10-17T09:00:00") == 1
    assert conn.execute("SELECT COUNT(*), MAX(fetched_at) FROM entry_races").fetchone() == (1, "2026-10-17T09:00:00")
    assert conn.execute("SELECT number, win_odds, popularity FROM entries ORDER BY number").fetchall() == [
        (1, 6.1, 2), (2, 22.8, 10)]
    assert import_entries(conn, tmp_path, "2026-10-18", "x") == 0
