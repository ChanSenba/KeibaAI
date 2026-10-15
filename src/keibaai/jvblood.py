"""JRA-VAN の血統データ(BLDN)を読んで、pedigree.sire_line(父の系統)と pedigree5(5代血統)を埋める。

書き出し(jvexport.py の BLDN.txt)を読む。レコードは Shift_JIS の固定長で、位置は 0 始まり(jvrace と同じ読み方)。

  共通: 0-1 種別, 2 区分, 3-10 作成日
  HN(繁殖馬マスタ, 251バイト): 11-20 繁殖登録番号, 29-38 血統登録番号, 40-75 馬名, 116-195 馬名欧字,
      229-238 父の繁殖登録番号, 239-248 母の繁殖登録番号
  SK(産駒マスタ, 208バイト): 11-20 血統登録番号(netkeiba の馬 ID と同じ), 66- 3代血統の繁殖登録番号(10桁)が14頭ぶん。
      並びは 父, 母, 父の父, 父の母, 母の父, 母の母, …(UM と同じ)
  BT(系統情報, 6889バイト): 11-20 繁殖登録番号, 21-50 系統ID, 51-86 系統名, 87- 系統説明

位置は書き出したデータで確かめた(JV-Data 仕様書は手元になかった)。
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

from .jvrace import _records, _txt

GENERATIONS = 5
POSITIONS = 2 ** (GENERATIONS + 1) - 2  # 5代で62頭
SK_START, SK_COUNT = 66, 14
MAX_SIRE_LINE_DEPTH = 40  # 父系をさかのぼる上限(循環したデータで止まらないように)


def _id(b: bytes) -> str | None:
    s = b.decode("ascii", errors="replace")
    return s if s.isdigit() and int(s) > 0 else None


def parse_hn(b: bytes) -> tuple | None:
    """HN の1件を (繁殖登録番号, 馬名, 父, 母, 作成日) にする。馬名は日本語がなければ欧字。"""
    if b[:2] != b"HN" or len(b) < 249 or b[2:3] == b"0":
        return None
    hid = _id(b[11:21])
    if not hid:
        return None
    name = _txt(b[40:76]) or _txt(b[116:196]) or None
    return hid, name, _id(b[229:239]), _id(b[239:249]), b[3:11].decode("ascii", errors="replace")


def parse_sk(b: bytes) -> tuple | None:
    """SK の1件を (血統登録番号, 3代血統の繁殖登録番号14個, 作成日) にする。"""
    if b[:2] != b"SK" or len(b) < SK_START + 10 * SK_COUNT or b[2:3] == b"0":
        return None
    horse = _id(b[11:21])
    if not horse:
        return None
    anc = [_id(b[SK_START + 10 * i:SK_START + 10 * (i + 1)]) for i in range(SK_COUNT)]
    return horse, anc, b[3:11].decode("ascii", errors="replace")


def parse_bt(b: bytes) -> tuple | None:
    """BT の1件を (繁殖登録番号, 系統ID, 系統名, 作成日) にする。"""
    if b[:2] != b"BT" or len(b) < 87 or b[2:3] == b"0":
        return None
    hid = _id(b[11:21])
    name = _txt(b[51:87])
    if not hid or not name:
        return None
    return hid, b[21:51].decode("ascii", errors="replace").strip(), name, b[3:11].decode("ascii", errors="replace")


class Blood:
    """BLDN を読んだ結果。同じ馬が何度も出てくるので、作成日が新しいものを使う。"""

    def __init__(self) -> None:
        self.hn: dict[str, tuple] = {}  # 繁殖登録番号 → (馬名, 父, 母)
        self.sk: dict[str, list] = {}  # 血統登録番号 → 3代血統14頭
        self.bt: dict[str, tuple] = {}  # 繁殖登録番号 → (系統ID, 系統名)
        self._made: dict[tuple, str] = {}
        self.counts: Counter = Counter()

    def _newer(self, kind: str, key: str, made: str) -> bool:
        if made < self._made.get((kind, key), ""):
            return False
        self._made[(kind, key)] = made
        return True

    def add(self, b: bytes) -> None:
        kind = b[:2].decode("ascii", errors="replace")
        self.counts[kind] += 1
        if kind == "HN" and (r := parse_hn(b)) and self._newer(kind, r[0], r[4]):
            self.hn[r[0]] = r[1:4]
        elif kind == "SK" and (r := parse_sk(b)) and self._newer(kind, r[0], r[2]):
            self.sk[r[0]] = r[1]
        elif kind == "BT" and (r := parse_bt(b)) and self._newer(kind, r[0], r[3]):
            self.bt[r[0]] = r[1:3]

    def sire_line(self, sire_id: str | None) -> str | None:
        """父の系統名。父に系統情報がなければ、父の父、その父…とさかのぼって最初に見つかった系統。"""
        seen = set()
        hid = sire_id
        while hid and hid not in seen and len(seen) < MAX_SIRE_LINE_DEPTH:
            if hid in self.bt:
                return self.bt[hid][1]
            seen.add(hid)
            hid = (self.hn.get(hid) or (None, None, None))[1]
        return None

    def ancestors(self, horse_id: str, sire_id: str | None = None, dam_id: str | None = None) -> list[str | None]:
        """5代血統の繁殖登録番号(pos 1〜62 の順。たどれない先祖は None)。

        3代までは産駒マスタ、その先は繁殖馬マスタの父・母でたどる。産駒マスタにない馬は、与えた父・母から始める。
        """
        anc: list[str | None] = [None] * (POSITIONS + 1)  # 0 は馬自身(使わない)
        for i, a in enumerate(self.sk.get(horse_id) or []):
            anc[i + 1] = a
        anc[1] = anc[1] or sire_id
        anc[2] = anc[2] or dam_id
        for k in range(3, POSITIONS + 1):
            if anc[k]:
                continue
            parent = anc[(k - 1) // 2]
            hn = self.hn.get(parent) if parent else None
            if hn:
                anc[k] = hn[1] if k % 2 else hn[2]  # 奇数が父、偶数が母
        return anc[1:]

    def name(self, hid: str | None) -> str | None:
        return (self.hn.get(hid) or (None,))[0] if hid else None


def read_blood(raw_dir: str | Path) -> Blood:
    blood = Blood()
    path = Path(raw_dir) / "BLDN.txt"
    if path.exists():
        for b in _records(path):
            blood.add(b)
    return blood


def import_blood(conn: sqlite3.Connection, raw_dir: str | Path, blood: Blood | None = None) -> dict:
    """pedigree の馬と成績・出馬表に出た馬について、sire_line と pedigree5 を入れる。結果の数をまとめて返す。

    sire_line は分かった馬だけ書き換える(分からなければ前の値を残す)。pedigree5 は対象の馬の行を入れ替える。
    """
    blood = blood or read_blood(raw_dir)
    if not blood.hn and not blood.sk:
        return {"horses": 0}
    ped = {h: (s, d) for h, s, d in conn.execute("SELECT horse_id, sire_id, dam_id FROM pedigree")}
    horses = set(ped)
    for table in ("results", "entries"):
        try:
            horses |= {h for (h,) in conn.execute(f"SELECT DISTINCT horse_id FROM {table} WHERE horse_id IS NOT NULL")}
        except sqlite3.OperationalError:  # 表がない(古い DB)
            pass
    horses = sorted(h for h in horses if h and h.strip())
    lines: dict[str, str] = {}
    rows = []
    missing = 0
    for h in horses:
        sire, dam = ped.get(h, (None, None))
        anc = blood.ancestors(h, sire, dam)
        line = blood.sire_line(anc[0] or sire)
        if line and h in ped:
            lines[h] = line
        missing += sum(a is None for a in anc)
        rows += [(h, k + 1, a, blood.name(a)) for k, a in enumerate(anc)]
    with conn:
        conn.executemany("UPDATE pedigree SET sire_line = ? WHERE horse_id = ?", [(v, h) for h, v in lines.items()])
        conn.executemany("INSERT OR REPLACE INTO pedigree5 (horse_id, pos, ancestor_id, name) VALUES (?,?,?,?)", rows)
    return {"horses": len(horses), "pedigree": len(ped), "sire_line": len(lines),
            "missing_rate": missing / (len(horses) * POSITIONS) if horses else 0.0,
            "top_lines": Counter(lines.values()).most_common(10), "records": dict(blood.counts)}
