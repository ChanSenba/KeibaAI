from keibaai import db
from keibaai.features import parse_class
from keibaai.jvrace import (PED_START, PED_WIDTH, from_bytes, import_pedigree, import_races, parse_ra, parse_se,
                            parse_um, to_bytes)


def record(length: int, fields: dict[int, str]) -> bytes:
    """空白で埋めた固定長レコードに、位置(0始まり)ごとの値を Shift_JIS で書き込む。末尾は CRLF。"""
    b = bytearray(b" " * (length - 2) + b"\r\n")
    for pos, value in fields.items():
        v = value.encode("cp932")
        b[pos:pos + len(v)] = v
    return bytes(b)


def ra(race="2026100305040111", kubun="7", title="富士ステークス", grade="B", cond="999", track="11", over=None):
    f = {0: "RA", 2: kubun, 3: "20261005", 11: race, 32: title, 614: grade, 616: "13", 634: cond, 697: "1600",
         705: track, 873: "1545", 887: "1", 888: "2", 889: "0", 890: "123110118", 969: "345", 975: "339"}
    f.update(over or {})
    return record(1272, f)


def se(race="2026100305040111", number="01", abnormal="0", over=None):
    f = {0: "SE", 2: "7", 3: "20261005", 11: race, 27: "1", 28: number, 30: "2024105705", 40: "アークレイリ",
         78: "1", 82: "02", 85: "01141", 90: "西村真幸", 104: "伊藤　正樹", 288: "560", 296: "01091", 306: "丹内祐次",
         324: "458", 327: "-", 328: "004", 331: abnormal, 334: "03", 338: "1394", 351: "00001010", 359: "0807",
         363: "07", 365: "00015000", 373: "00000000", 390: "364"}
    f.update(over or {})
    return record(555, f)


def um(horse="2017100464", made="20261001", sire=("1120002268", "ルーラーシップ"), damsire=("1120002086", "ディープインパクト")):
    anc = [sire, ("1220062257", "ヒーラ"), ("1120002006", "キングカメハメハ"), ("1220043789", "エアグルーヴ"), damsire]
    f = {0: "UM", 2: "4", 3: made, 11: horse}
    for i in range(14):
        hn, name = anc[i] if i < len(anc) else ("1140003006", "Kingmambo")
        f[PED_START + PED_WIDTH * i] = hn
        f[PED_START + PED_WIDTH * i + 10] = name
    return record(1609, f)


def write(tmp_path, name, records):
    lines = ["x.jvd\t" + from_bytes(r[:-2]) + "\n" for r in records]  # jvexport と同じく末尾の CRLF は落とす
    (tmp_path / name).write_text("".join(lines), encoding="utf-8", newline="\n")


def test_bytes_round_trip():
    raw = bytes(range(0x20, 0x100))  # Shift_JIS の全バイト(1バイト1文字で渡ってくる)
    assert to_bytes(from_bytes(raw)) == raw
    assert to_bytes(from_bytes("皐月賞".encode("cp932"))).decode("cp932") == "皐月賞"


def test_parse_race():
    r = parse_ra(ra())
    assert r == ("202605040111", "富士ステークス(GII)", "2026-10-03", "東京", 11, "芝", "左", 1600, "晴", "稍重",
                 "15:45", "12.3-11.0-11.8", 34.5, 33.9, None)
    assert parse_class(r[1]) == 7.0


def test_race_name_without_title_uses_age_and_class():
    r = parse_ra(ra(title="", grade=" ", cond="703", track="24", over={616: "11", 888: "0", 889: "3"}))
    assert (r[1], r[5], r[6], r[9]) == ("2歳未勝利", "ダ", "右", "重")
    assert parse_class(r[1]) == 1.0
    assert parse_ra(ra(title="清滝特別", grade="E", cond="010"))[1] == "清滝特別(2勝)"


def test_quick_results_replace_until_final_arrives(tmp_path):
    """速報成績(区分6)で入れたレースは、直近のあいだは確定成績(区分7)で入れ替わる。古いレースは触らない。"""
    conn = db.connect(":memory:")
    quick = [ra(kubun="6", title="速報の名前"), se(over={2: "6"}), se(number="02", over={2: "6"})]
    write(tmp_path, "RT_0B15.txt", quick)
    assert import_races(conn, tmp_path, refresh_since="2026-10-01") == (1, 2)
    assert conn.execute("SELECT name FROM races").fetchone()[0] == "速報の名前(GII)"
    write(tmp_path, "RACE.txt", [ra(), se()])  # 確定成績が出た(出走は1頭に減った想定)
    assert import_races(conn, tmp_path, refresh_since="2026-10-01") == (1, 1)  # 両方のファイルにあっても確定を使う
    assert conn.execute("SELECT name FROM races").fetchone()[0] == "富士ステークス(GII)"
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1
    assert import_races(conn, tmp_path, refresh_since="2026-10-10") == (0, 0)  # 入れ替えの期間より前は、あれば触らない
    assert import_races(conn, tmp_path) == (0, 0)


def test_parse_race_skips_other_data():
    assert parse_ra(ra(kubun="2")) is None  # 出馬表の段階
    assert parse_ra(ra(kubun="6")) is not None  # 速報成績は使う
    assert parse_ra(ra(race="2026100330040111")) is None  # 地方競馬の場コード
    assert parse_ra(se()) is None


def test_parse_result():
    assert parse_se(se()) == ("202605040111", 1, 3, None, 1, "2024105705", "アークレイリ", "牡2", 56.0, "丹内祐次",
                              "01091", 99.4, None, "10-10", 36.4, 80.7, 7, 458, -4, "西村真幸", "01141", "伊藤正樹",
                              150.0)


def test_parse_result_scratched_and_first_start():
    r = parse_se(se(abnormal="1", over={334: "00", 338: "0000", 359: "0000", 363: "00", 390: "000", 365: "00000000"}))
    assert (r[2], r[3], r[11], r[14], r[15], r[22]) == (None, "取", None, None, None, None)
    r = parse_se(se(over={327: " ", 328: "   "}))  # 初出走: 前走比なし。netkeiba と同じく 0
    assert (r[17], r[18]) == (458, 0)
    r = parse_se(se(over={365: "00114000", 373: "00004270"}))  # 本賞金 + 付加賞金
    assert r[22] == 1182.7


def test_import_races_adds_only_new_races(tmp_path):
    conn = db.connect(":memory:")
    conn.execute("INSERT INTO races (race_id, name) VALUES ('202605040111', 'netkeiba の名前')")
    old = "2019010606010101"
    write(tmp_path, "RACE.txt", [ra(), se(), se(number="02"), ra(race=old), se(race=old), se(race="2026100305040112")])
    assert import_races(conn, tmp_path, since="2019-01-01") == (1, 1)  # すでにあるレースと、RA のない出走馬は入れない
    assert conn.execute("SELECT name FROM races WHERE race_id='202605040111'").fetchone()[0] == "netkeiba の名前"
    assert conn.execute("SELECT date FROM races WHERE race_id='201906010101'").fetchone()[0] == "2019-01-06"
    assert import_races(conn, tmp_path, since="2019-01-01") == (0, 0)  # 2回目は何も増えない
    assert import_races(conn, tmp_path, since="2020-01-01", replace=True) == (1, 2)  # since より前は入れない
    assert conn.execute("SELECT name FROM races WHERE race_id='202605040111'").fetchone()[0] == "富士ステークス(GII)"
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 3


def test_parse_horse_master():
    row, made = parse_um(um())
    assert row == ("2017100464", "1120002268", "ルーラーシップ", "1220062257", "ヒーラ", "1120002086",
                   "ディープインパクト", "1120002006", "キングカメハメハ")
    assert made == "20261001"


def test_import_pedigree_updates_listed_horses_and_drops_netkeiba_rows(tmp_path):
    conn = db.connect(":memory:")
    conn.executemany("INSERT INTO pedigree (horse_id, sire_id, sire, damsire_id) VALUES (?,?,?,?)", [
        ("2015100001", "000a00033a", "サンデーサイレンス", "1998101786"),  # netkeiba の番号(輸入馬・生まれ年)
        ("2016100002", "2002100816", "ディープインパクト", None),
        ("2014100003", "1120002086", "ディープインパクト", "1120001232"),  # 前に JRA-VAN から入れた馬
    ])
    write(tmp_path, "DIFN.txt", [um(made="20260903", sire=("1120009999", "古い父")), um(), um(horse="2018100442")])
    assert import_pedigree(conn, tmp_path) == 2
    rows = dict(conn.execute("SELECT horse_id, sire FROM pedigree"))
    # 新しい作成日を使う。差分に入っていない JRA-VAN の馬は残り、netkeiba の番号の行だけ消える
    assert rows == {"2017100464": "ルーラーシップ", "2018100442": "ルーラーシップ", "2014100003": "ディープインパクト"}


def test_import_without_files_does_nothing(tmp_path):
    conn = db.connect(":memory:")
    conn.execute("INSERT INTO pedigree (horse_id, sire) VALUES ('2015100001', 'x')")
    assert import_races(conn, tmp_path) == (0, 0)
    assert import_pedigree(conn, tmp_path) == 0
    assert conn.execute("SELECT COUNT(*) FROM pedigree").fetchone()[0] == 1


def test_course_inner_outer_from_track_code():
    assert parse_ra(ra(track="18", race="2026100308040111"))[3:7] == ("京都", 11, "芝", "右")
    assert parse_ra(ra(track="18", race="2026100308040111"))[14] == "外"
    assert parse_ra(ra(track="17", race="2026100308040111"))[14] == "内"  # 京都は内外の別がある
    assert parse_ra(ra(track="11"))[14] is None  # 東京は内外の別がない
    assert parse_ra(ra(track="20", race="2026100306040111"))[14] == "外-内"