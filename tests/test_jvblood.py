from keibaai import db
from keibaai.jvblood import POSITIONS, Blood, import_blood, parse_bt, parse_hn, parse_sk, read_blood
from keibaai.jvrace import import_pedigree

from test_jvrace import record, um, write


def hn(hid, name, father="", mother="", made="20261001", roman=""):
    return record(251, {0: "HN", 2: "1", 3: made, 11: hid, 40: name, 116: roman, 229: father, 239: mother})


def sk(horse, anc, made="20261001"):
    f = {0: "SK", 2: "1", 3: made, 11: horse}
    for i, a in enumerate(anc):
        f[66 + 10 * i] = a
    return record(208, f)


def bt(hid, line_id, name, made="20261001"):
    return record(6889, {0: "BT", 2: "1", 3: made, 11: hid, 21: line_id, 51: name, 87: "説明"})


def blood_of(records) -> Blood:
    b = Blood()
    for r in records:
        b.add(r[:-2])
    return b


def test_parse_records():
    assert parse_hn(hn("1120002268", "ルーラーシップ", "1120002006", "1220043789")[:-2]) == (
        "1120002268", "ルーラーシップ", "1120002006", "1220043789", "20261001")
    assert parse_hn(hn("1140003006", "", roman="Kingmambo")[:-2])[1:4] == ("Kingmambo", None, None)
    h, anc, _ = parse_sk(sk("2017100464", ["1120002268", "1220062257"])[:-2])
    assert h == "2017100464" and anc[:3] == ["1120002268", "1220062257", None]
    assert parse_bt(bt("1120002006", "0101", "キングマンボ系")[:-2])[:3] == ("1120002006", "0101", "キングマンボ系")
    assert parse_hn(sk("2017100464", [])[:-2]) is None


def test_sire_line_walks_up_the_male_line():
    b = blood_of([hn("1100000013", "孫", "1100000012", ""), hn("1100000012", "子", "1100000011", ""), hn("1100000011", "祖", "", ""), bt("1100000011", "01", "祖系")])
    assert b.sire_line("1100000013") == "祖系"
    b.add(bt("1100000012", "0101", "子系")[:-2])
    assert b.sire_line("1100000013") == "子系"
    assert b.sire_line("1100009999") is None


def test_sire_line_stops_on_loops():
    b = blood_of([hn("1100000011", "a", "1100000012", ""), hn("1100000012", "b", "1100000011", "")])
    assert b.sire_line("1100000011") is None


def test_ancestors_use_sk_then_hn():
    # 3代は SK、4代目は HN をたどる(pos 15 = 父父父の父)
    b = blood_of([sk("2017100000", ["1100000001", "1200000002", "1100000003", "1200000004", "1100000005", "1200000006", "1100000007"]), hn("1100000007", "父父父", "1100000015", "1200000016"), hn("1100000015", "X名")])
    anc = b.ancestors("2017100000")
    assert len(anc) == POSITIONS
    assert anc[:7] == ["1100000001", "1200000002", "1100000003", "1200000004", "1100000005", "1200000006", "1100000007"]
    assert anc[14] == "1100000015" and anc[15] == "1200000016"  # pos 15・16
    assert anc[30] is None  # pos 31 = X の父(HN に父がない)
    assert b.name("1100000015") == "X名"


def test_ancestors_without_sk_start_from_given_parents():
    b = blood_of([hn("1100000001", "父", "1100000003", "1200000004")])
    anc = b.ancestors("2017100000", "1100000001", "1200000002")
    assert anc[:4] == ["1100000001", "1200000002", "1100000003", "1200000004"]


def test_import_blood_fills_sire_line_and_pedigree5(tmp_path):
    conn = db.connect(":memory:")
    conn.execute("INSERT INTO pedigree (horse_id, sire_id, dam_id, sire_line) VALUES ('2017100001','1100000001','1200000002',NULL), "
                 "('2017100002','1100000099','1200000099','前の値')")
    write(tmp_path, "BLDN.txt", [sk("2017100001", ["1100000001", "1200000002"]), hn("1100000001", "父", "1100000003", "1200000004"), hn("1100000003", "父の父"),
                                 bt("1100000003", "01", "父父系")])
    r = import_blood(conn, tmp_path)
    assert r["horses"] == 2 and r["sire_line"] == 1
    lines = dict(conn.execute("SELECT horse_id, sire_line FROM pedigree"))
    assert lines == {"2017100001": "父父系", "2017100002": "前の値"}  # 分からない馬は前の値を残す
    assert conn.execute("SELECT COUNT(*) FROM pedigree5").fetchone()[0] == 2 * POSITIONS
    assert conn.execute("SELECT ancestor_id, name FROM pedigree5 WHERE horse_id='2017100001' AND pos=3").fetchone() == ("1100000003", "父の父")
    assert 0 < r["missing_rate"] < 1


def test_import_blood_without_file_does_nothing(tmp_path):
    conn = db.connect(":memory:")
    assert import_blood(conn, tmp_path)["horses"] == 0
    assert read_blood(tmp_path).hn == {}


def test_import_pedigree_keeps_sire_line(tmp_path):
    conn = db.connect(":memory:")
    conn.execute("INSERT INTO pedigree (horse_id, sire, sire_line) VALUES ('2017100464', '古い', 'キングマンボ系')")
    write(tmp_path, "DIFN.txt", [um()])
    assert import_pedigree(conn, tmp_path) == 1
    assert conn.execute("SELECT sire, sire_line FROM pedigree").fetchone() == ("ルーラーシップ", "キングマンボ系")
