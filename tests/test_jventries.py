from test_jvrace import ra, record, se, write

from keibaai import db
from keibaai.jventries import import_entries, latest_conditions, parse_o1, parse_we, race_keys, read_entries

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


def we(announced, weather, turf, dirt, venue="08", ymd="20261017"):
    """天候・馬場の速報(WE)。京都 4回5日。"""
    return record(42, {0: "WE", 2: "1", 3: ymd, 11: ymd, 19: venue, 21: "04", 23: "05", 25: announced, 33: "3",
                       34: weather, 35: turf, 36: dirt, 37: "111"})


def test_parse_weather_and_ground():
    assert parse_we(we("10170930", "3", "2", "3")) == ("2026-10-17", "京都", "10170930", "雨", "稍重", "重")
    assert parse_we(we("10170930", "0", "0", "0"))[3:] == (None, None, None)  # 未設定
    assert parse_we(card()[0]) is None


def test_latest_weather_and_ground_replace_the_card(tmp_path):
    """当日の天候・馬場の変更(0B14)があれば、いちばん新しい発表で出馬表の天候・馬場を置き換える。"""
    dirt = ra(race="2026101708040501", kubun="2", title="", track="23")
    write(tmp_path, "RT_0B15.txt", card() + [dirt, se(race="2026101708040501", number="01", over={2: "2"})])
    assert {p.race.race_id: (p.race.weather, p.race.ground) for p in read_entries(tmp_path, "2026-10-17")} == {
        "202608040511": ("晴", "稍重"), "202608040501": ("晴", "稍重")}  # 速報がなければ RA のまま
    write(tmp_path, "RT_0B14.txt", [we("10171215", "3", "3", "4"), we("10170900", "2", "2", "2"),
                                    we("10171300", "1", "1", "1", venue="05"),  # 別の競馬場
                                    we("10181300", "1", "1", "1", ymd="20261018")])  # 別の日
    assert {p.race.race_id: (p.race.weather, p.race.ground) for p in read_entries(tmp_path, "2026-10-17")} == {
        "202608040511": ("雨", "重"), "202608040501": ("雨", "不良")}  # 芝は芝、ダートはダートの馬場


# 2026-10-25 京都の実データ(RT_0B14.txt。末尾の CRLF は落としてある)。朝の発表のあと、天候の変更と馬場の変更が別々に来る
KYOTO_1025 = ["WE12026102420261025080409000000001224000", "WE12026102520261025080409102506553023024",
              "WE12026102520261025080409102508272300200", "WE12026102520261025080409102509063024023",
              "WE12026102520261025080409102509223034024", "WE12026102520261025080409102509542400300",
              "WE12026102520261025080409102510262200400", "WE12026102520261025080409102511403024034"]


def test_weather_and_ground_are_taken_field_by_field(tmp_path):
    """天候の変更(2)には馬場が、馬場の変更(3)には天候が入らない(0)。欄ごとにいちばん新しい値を使う。"""
    recs = [r.encode() + b"\r\n" for r in reversed(KYOTO_1025)]  # 並びに頼らない
    assert parse_we(recs[0]) == ("2026-10-25", "京都", "10251140", None, "稍重", "不良")  # 11:40 の馬場の変更
    write(tmp_path, "RT_0B14.txt", recs)
    assert latest_conditions(tmp_path, "2026-10-25") == {"京都": ("曇", "稍重", "不良")}  # 天候は 10:26 の発表
