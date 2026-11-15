from keibaai import db
from keibaai.jvsales import import_sales, parse_hs

from test_jvrace import record, write


def hs(horse="2017100464", market="010001", start="20180910", price="0001080000", kubun="1",
       made="20181001", age="1", host="日本競走馬協会", name="セレクトセール"):
    return record(200, {0: "HS", 2: kubun, 3: made, 11: horse, 21: "1120002268", 31: "1220062257",
                        41: "2017", 45: market, 51: host, 91: name, 171: start, 179: "20180911",
                        187: age, 188: price})


def test_parse_hs():
    key, made, row = parse_hs(hs()[:-2])
    assert key == ("2017100464", "010001", "2018-09-10") and made == "20181001"
    assert row[0:3] == ("2017100464", "010001", "2018-09-10")
    assert row[3] == "2018-09-11" and row[4] == "日本競走馬協会" and row[5] == "セレクトセール"
    assert row[6] == 1 and row[7] == 1080000  # 年齢と価格(円)
    assert row[8] == "1120002268" and row[9] == "1220062257" and row[10] == 2017
    assert parse_hs(hs(kubun="0")[:-2])[2] is None  # 削除レコード
    assert parse_hs(b"XX" + hs()[2:-2]) is None


def test_import_sales_dedupes_and_deletes(tmp_path):
    conn = db.connect(":memory:")
    write(tmp_path, "HOSN.txt", [hs(), hs(price="0002160000", made="20181101"),  # 同じキーは新しい作成日が勝つ
                                 hs(start="20200706", age="2", price="0000540000")])  # 2回目の取引は別行
    assert import_sales(conn, tmp_path) == 2
    rows = conn.execute("SELECT start_date, age, price FROM sales ORDER BY start_date").fetchall()
    assert rows == [("2018-09-10", 1, 2160000), ("2020-07-06", 2, 540000)]
    write(tmp_path, "HOSN.txt", [hs(kubun="0", made="20181201")])
    import_sales(conn, tmp_path)
    assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 1  # 削除レコードで1件消える


def test_import_sales_without_file(tmp_path):
    conn = db.connect(":memory:")
    assert import_sales(conn, tmp_path) == 0


def test_import_sales_older_record_does_not_overwrite(tmp_path):
    conn = db.connect(":memory:")
    write(tmp_path, "HOSN.txt", [hs(price="0002160000", made="20181101")])
    import_sales(conn, tmp_path)
    write(tmp_path, "HOSN.txt", [hs(price="0001080000", made="20181001")])  # 古い作成日では巻き戻らない
    import_sales(conn, tmp_path)
    assert conn.execute("SELECT price FROM sales").fetchone()[0] == 2160000


def test_import_sales_older_delete_does_not_remove(tmp_path):
    conn = db.connect(":memory:")
    write(tmp_path, "HOSN.txt", [hs(made="20181101")])
    import_sales(conn, tmp_path)
    write(tmp_path, "HOSN.txt", [hs(kubun="0", made="20181001")])  # 行より古い削除レコードでは消えない
    import_sales(conn, tmp_path)
    assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 1
