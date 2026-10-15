"""JRA-VAN のデータから、出馬表と単勝オッズを entry_races / entries に入れる(netkeiba の代わり)。

書き出し(jvexport.py、32bit Python)の出力を読む。
  RACE.txt: 「今週のデータ」(RACE, option 2)。木曜に出走馬名表(区分 1、馬番なし)、枠順の発表後に出馬表(区分 2)が出る
  RT_0B15.txt: その日のレース(RA)と出走馬(SE)の速報。当日の出馬表や成績。過去の日は確定成績が出ると取れなくなる
  どちらも同じ形で読める。同じレースが両方にあれば、区分の進んだほう(数字の大きいほう)を使う
  RT_0B31.txt: 単勝・複勝・枠連のオッズ(O1)。レースごとに1件
O1(962バイト、位置は 0 始まり): 0-1 種別, 2 区分, 11-26 レース, 43- 単勝(馬番2桁 + オッズ4桁(0.1倍単位) + 人気2桁)が28頭ぶん。
取消などでオッズのない馬は、数字の代わりに記号が入っている。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from . import db
from .entries import Entry, EntryRace, ParsedEntries
from .jvrace import VENUES, _num, _records, parse_ra, parse_se, race_id_of

ENTRY_KUBUNS = tuple(str(i).encode() for i in range(1, 8))  # 出走馬名表・出馬表・速報成績・確定成績のどれでも読む
WIN_START, WIN_WIDTH, WIN_COUNT = 43, 8, 28


def parse_o1(b: bytes) -> tuple[str, dict[int, tuple[float | None, int | None]]] | None:
    """O1 の1件から (race_id, {馬番: (単勝オッズ, 人気)}) を作る。対象外は None。"""
    if b[:2] != b"O1" or len(b) < WIN_START + WIN_WIDTH * WIN_COUNT or b[19:21].decode("ascii", "replace") not in VENUES:
        return None
    out = {}
    for i in range(WIN_COUNT):
        e = b[WIN_START + WIN_WIDTH * i:WIN_START + WIN_WIDTH * (i + 1)]
        number = _num(e[:2])
        if number:
            out[number] = (_num(e[2:6], 10), _num(e[6:8]))
    return race_id_of(b), out


CARD_FILES = ("RACE.txt", "RT_0B15.txt")  # 今週のデータ、速報


def _card_records(raw_dir: Path):
    for name in CARD_FILES:
        path = raw_dir / name
        if path.exists():
            yield from _records(path)


def race_keys(raw_dir: str | Path, date: str) -> list[str]:
    """書き出しにある、その日(YYYY-MM-DD)の中央のレースのキー(年月日+場+回+日+R の16桁)。オッズを取るのに使う。"""
    ymd = date.replace("-", "").encode()
    keys = [b[11:27].decode() for b in _card_records(Path(raw_dir))
            if b[:2] == b"RA" and b[11:19] == ymd and b[19:21].decode("ascii", "replace") in VENUES]
    return sorted(set(keys))


def read_entries(raw_dir: str | Path, date: str) -> list[ParsedEntries]:
    """書き出しから、その日(YYYY-MM-DD)の出馬表をレースごとに組み立てる。オッズがあれば付ける。

    同じレースの出走馬が、出走馬名表(区分 1)と出馬表(区分 2)のように複数の段階で入っていたら、進んだほうだけを使う。
    """
    raw_dir = Path(raw_dir)
    races: dict[str, EntryRace] = {}
    horses: dict[str, dict[int, Entry]] = {}
    level: dict[tuple[str, bytes], bytes] = {}
    for b in _card_records(raw_dir):
        if b[:2] == b"RA":
            r = parse_ra(b, ENTRY_KUBUNS)
            if r and r[2] == date and b[2:3] >= level.get((r[0], b"RA"), b"0"):
                level[(r[0], b"RA")] = b[2:3]
                races[r[0]] = EntryRace(*r[:11])
        elif b[:2] == b"SE":
            s = parse_se(b, ENTRY_KUBUNS)
            if not s:
                continue
            kubun = b[2:3]
            if kubun < level.get((s[0], b"SE"), b"0"):  # 古い段階のデータは使わない
                continue
            if kubun > level.get((s[0], b"SE"), b"0"):  # 新しい段階が来たら、そのレースの古い行は捨てる
                level[(s[0], b"SE")] = kubun
                horses.pop(s[0], None)
            if not s[1] or s[3] in ("取", "除"):  # 馬番の確定前(0)と、取消・除外の馬は入れない
                continue
            horses.setdefault(s[0], {})[s[1]] = Entry(
                    race_id=s[0], number=s[1], bracket=s[4], horse_id=s[5], horse_name=s[6], sex_age=s[7],
                    weight_carried=s[8], jockey=s[9], jockey_id=s[10], trainer=s[19], trainer_id=s[20],
                    body_weight=s[17], body_weight_diff=s[18] if s[17] else None, win_odds=s[15], popularity=s[16])
    odds = raw_dir / "RT_0B31.txt"
    if odds.exists():
        for b in _records(odds):
            o = parse_o1(b)
            if not o or o[0] not in horses:
                continue
            for number, (win, pop) in o[1].items():
                e = horses[o[0]].get(number)
                if e and win is not None:
                    e.win_odds, e.popularity = win, pop if pop is not None else e.popularity
    return [ParsedEntries(races[i], [h[n] for n in sorted(h)]) for i, h in sorted(horses.items()) if i in races and h]


def import_entries(conn: sqlite3.Connection, raw_dir: str | Path, date: str, fetched_at: str) -> int:
    """その日の出馬表と単勝オッズを保存する(同じレースは置き換え)。保存したレース数を返す。"""
    parsed = read_entries(raw_dir, date)
    for p in parsed:
        db.save_entries(conn, p, fetched_at)
    return len(parsed)
