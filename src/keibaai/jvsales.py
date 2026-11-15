"""JRA-VAN の競走馬市場取引価格(HOSN の HS レコード)を取り込む。データ層だけ(特徴量にはまだ使わない)。

レコードは Shift_JIS の固定長 200 バイト(末尾の改行を除いて 198 文字)。下の位置はすべて 0 始まりのバイト位置。
項目: 11-20 血統登録番号(results.horse_id と同じ体系), 21-30 父の繁殖登録番号,
31-40 母の繁殖登録番号, 41-44 生年, 45-50 主催者・市場コード, 51-90 主催者名称, 91-170 市場の名称,
171-178 開催期間(開始日), 179-186 終了日, 187 取引時の年齢, 188-197 取引価格(円)。

キーは (血統登録番号, 市場コード, 開始日) で、同じ馬の複数回の取引は別々に残る。
データ区分 0 は削除(提供ミス等)のレコードで、同じキーの行を取り消す。
特徴量に使うときは、対象レースより前の取引だけを使うこと(取引日が開始日。リーク防止は使う側の責任)。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .jvrace import _records, _txt


def _id(b: bytes) -> str | None:
    s = b.decode("ascii", errors="replace")
    return s if s.isdigit() and int(s) > 0 else None


def _int(b: bytes) -> int | None:
    s = b.decode("ascii", errors="replace").strip()
    return int(s) if s.isdigit() else None


def parse_hs(b: bytes) -> tuple | None:
    """HS の1件を (キー, 作成日, 行) にする。削除レコードは行が None。形が違うものは None を返す。"""
    if b[:2] != b"HS" or len(b) < 198:
        return None
    horse = _id(b[11:21])
    market = b[45:51].decode("ascii", errors="replace").strip()
    start = b[171:179].decode("ascii", errors="replace")
    if not horse or not market or not start.isdigit():
        return None
    key = (horse, market, f"{start[:4]}-{start[4:6]}-{start[6:]}")
    made = b[3:11].decode("ascii", errors="replace")
    if b[2:3] == b"0":
        return key, made, None
    end = b[179:187].decode("ascii", errors="replace")
    row = (horse, market, key[2], f"{end[:4]}-{end[4:6]}-{end[6:]}" if end.isdigit() else None,
           _txt(b[51:91]) or None, _txt(b[91:171]) or None, _int(b[187:188]), _int(b[188:198]),
           _id(b[21:31]), _id(b[31:41]), _int(b[41:45]), made)
    return key, made, row


def import_sales(conn: sqlite3.Connection, raw_dir: str | Path) -> int:
    """HOSN.txt を sales に取り込む。ファイルがなければ 0。同じキーは作成日が新しいものを使う。"""
    path = Path(raw_dir) / "HOSN.txt"
    if not path.exists():
        return 0
    latest: dict[tuple, tuple] = {}
    for b in _records(path):
        parsed = parse_hs(b)
        if parsed is None:
            continue
        key, made, row = parsed
        if key not in latest or made >= latest[key][0]:
            latest[key] = (made, row)
    rows = [row for _, row in latest.values() if row is not None]
    gone = [(*key, made) for key, (made, row) in latest.items() if row is None]
    before = conn.total_changes
    with conn:
        # 既にある行より古い作成日のレコードでは上書きしない(取り込む範囲を変えて再実行したときの巻き戻し防止)
        conn.executemany(
            "INSERT INTO sales (horse_id, market_code, start_date, end_date, host, market_name, "
            "age, price, sire_id, dam_id, birth_year, made) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT (horse_id, market_code, start_date) DO UPDATE SET "
            "end_date=excluded.end_date, host=excluded.host, market_name=excluded.market_name, age=excluded.age, "
            "price=excluded.price, sire_id=excluded.sire_id, dam_id=excluded.dam_id, birth_year=excluded.birth_year, "
            "made=excluded.made WHERE excluded.made >= COALESCE(sales.made, '')", rows)
        # 削除も作成日で判定する(古い範囲の取り込み直しで、新しい行を消さない)
        conn.executemany("DELETE FROM sales WHERE horse_id = ? AND market_code = ? AND start_date = ? "
                         "AND COALESCE(made, '') <= ?", gone)
    return conn.total_changes - before
