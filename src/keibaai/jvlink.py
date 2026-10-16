"""JRA-VAN(JV-Link)からの取得を、64bit 側(CLI・GUI・毎週の処理)からまとめて呼ぶ。

JV-Link は 32bit の部品なので、書き出しは 32bit Python で jvexport を動かし、ここではその結果を DB に入れる。
netkeiba と違ってアクセス間隔の待ちはない(JV-Link の内部サービスのため、開き直しの間は数秒あける)。
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from . import jvexport

def _find_python32() -> str:
    """pywin32 を入れた 32bit Python。環境変数 KEIBAAI_PYTHON32 があればそれ、なければ決まった場所で見つかったもの。"""
    candidates = [os.environ.get("KEIBAAI_PYTHON32"), r"C:\Python311-32\jvenv\Scripts\python.exe",
                  r"C:\Python311-32\python.exe"]
    return next((c for c in candidates if c and Path(c).exists()), candidates[1])


PYTHON32 = _find_python32()
RAW_DIR = "data/jv_raw"
RECENT_DAYS = 14  # この日数より新しいレースは、取り込み直すたびに入れ替える(速報成績 → 確定成績)


class JVError(RuntimeError):
    pass


_last_run = 0.0  # 前の書き出しが終わった時刻(続けて開き直すと JV-Link の内部サービスが落ちることがあったため)


def available(python32: str = PYTHON32) -> bool:
    """JV-Link を呼ぶ 32bit Python があるか(Windows でセットアップ済みの PC だけ)。"""
    return sys.platform == "win32" and Path(python32).exists()


def export(out: str | Path, from_ymd: str, specs: list[str], option: int = 1, rt: list[str] = (),
           python32: str = PYTHON32) -> None:
    """jvexport を 32bit Python で動かす。出力(件数など)はそのまま print する。失敗したら JVError。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in [f"{s}.txt" for s in specs] + [f"RT_{r.split(':')[0]}.txt" for r in rt]:
        (out / name).unlink(missing_ok=True)  # 「該当データなし」のとき、前の日の書き出しが残らないように
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent), "PYTHONIOENCODING": "utf-8"}
    cmd = [python32, "-m", "keibaai.jvexport", "--from", from_ymd, "--out", str(out), "--option", str(option),
           "--specs", *specs, "--rt", *rt]
    global _last_run
    time.sleep(max(0.0, _last_run + jvexport.RT_INTERVAL - time.monotonic()))
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    _last_run = time.monotonic()
    for line in (p.stdout + p.stderr).splitlines():
        print(line)
    if p.returncode:
        raise JVError("JV-Link からの書き出しに失敗しました(上のメッセージを確認)。"
                      "JV-Link の内部サービスが止まっている場合は、PC の再起動で直ります")


def fetch_entries(conn: sqlite3.Connection, date: str, odds: bool = True, raw_root: str | Path = RAW_DIR,
                  python32: str = PYTHON32, now: dt.datetime | None = None, race_ids: list[str] | None = None) -> int:
    """その日(YYYY-MM-DD)の出馬表と単勝オッズを JRA-VAN から取って保存する。保存したレース数を返す。

    出馬表はその日の全レースぶんをまとめて取る。race_ids を渡すと、オッズはそのレースだけ取る(1レースだけ取得するとき)。

    木曜〜金曜の出走馬名表・出馬表は「今週のデータ」(RACE, option 2)に、当日の出馬表は速報(0B15)に出る。両方取る。
    馬番は枠順の発表(金曜)のあとに入るので、それより前は 0 を返す。
    """
    from .jventries import import_entries, race_keys, read_entries

    raw = Path(raw_root) / "entries"  # 成績の取り込みが使う RACE.txt と混ぜない
    ymd = date.replace("-", "")
    week = (dt.date.fromisoformat(date) - dt.timedelta(days=7)).strftime("%Y%m%d")
    export(raw, week, ["RACE"], option=2, rt=[f"0B15:{ymd}"], python32=python32)
    (raw / "RT_0B31.txt").unlink(missing_ok=True)  # 前に取った別の日のオッズを混ぜない
    numbered = {p.race.race_id for p in read_entries(raw, date)}  # 馬番の入ったレースだけ(枠順の発表前はオッズもない)
    keys = [k for k in race_keys(raw, date) if k[:4] + k[8:] in numbered and (race_ids is None or k[:4] + k[8:] in race_ids)]
    if keys and odds:
        try:
            export(raw, ymd, [], rt=[f"0B31:{k}" for k in keys], python32=python32)
        except JVError:
            print("オッズの書き出しに失敗しました。出馬表だけ保存します")
    now = now or dt.datetime.now()
    return import_entries(conn, raw, date, now.isoformat(timespec="seconds"))


def fetch_results(conn: sqlite3.Connection, start: dt.date, end: dt.date, raw_root: str | Path = RAW_DIR,
                  python32: str = PYTHON32, today: dt.date | None = None) -> tuple[int, int]:
    """start〜end の成績を JRA-VAN から取って入れる。(レース数, 出走数) を返す。

    確定成績は通常データ(RACE の差分)に、開催直後の速報成績は速報(0B15)に出る。直近の日は両方取り、
    確定成績が出たら速報の内容と入れ替える。
    """
    from .jvrace import import_races

    today = today or dt.date.today()
    raw = Path(raw_root) / "recent"  # セットアップの全件の書き出し(raw_root/RACE.txt)は上書きしない
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    # 速報が残っているのは直近だけ。開催のある土日と月曜(祝日開催)だけ開く(開く回数を減らすため)
    recent = [d for d in days if today - dt.timedelta(days=7) <= d <= today and (d.weekday() >= 5 or d.weekday() == 0)]
    export(raw, f"{start - dt.timedelta(days=1):%Y%m%d}", ["RACE"], option=1,
           rt=[f"0B15:{d:%Y%m%d}" for d in recent], python32=python32)
    refresh = max(start, today - dt.timedelta(days=RECENT_DAYS)).isoformat()
    return import_races(conn, raw, start.isoformat(), refresh_since=refresh)


# 速報オッズの種別(JVRTOpen のデータ種別)→ 入っているレコード。2026-10-16 に実機で、10/17 の前日発売のオッズが
# 6種別とも1レース1件ずつ取れることを確かめた(レコードの形は確定オッズ O1〜O6 と同じ)
RT_ODDS_SPECS = {"0B31": "O1", "0B32": "O2", "0B33": "O3", "0B34": "O4", "0B35": "O5", "0B36": "O6"}
# jvodds の券種 → 買い目(betting)の券種名。枠連は枠番の組なので使わない。複勝・ワイドは安全側に下限を使う
BET_KINDS = {"win": "単勝", "place": "複勝", "quinella": "馬連", "wide": "ワイド", "exacta": "馬単",
             "trio": "三連複", "trifecta": "三連単"}


def race_key(conn: sqlite3.Connection, race_id: str) -> str:
    """race_id(12桁)を JV-Link のレースのキー(年+月日+場+回+日+R の16桁)にする。月日は出馬表(entry_races)から。"""
    row = conn.execute("SELECT date FROM entry_races WHERE race_id = ?", (race_id,)).fetchone()
    if not row or not row[0]:
        raise ValueError(f"{race_id} の出馬表がありません。先に entries で取得してください")
    return race_id[:4] + row[0][5:7] + row[0][8:10] + race_id[4:]


def announced_at(rec: str) -> str | None:
    """オッズのレコードの発表月日時分(27-34)を ISO の日時にする。年はレースの年。"""
    s = rec[27:35]
    if not s.isdigit() or s == "00000000":
        return None
    return f"{rec[11:15]}-{s[:2]}-{s[2:4]}T{s[4:6]}:{s[6:8]}:00"


def _rt_records(path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                _, _, rec = line.rstrip("\n").partition("\t")
                if rec:
                    yield rec


def read_bet_odds(raw_dir: str | Path, key: str) -> tuple[dict, str | None]:
    """書き出した速報オッズ(RT_0B31〜0B36.txt)から、そのレースの {券種: {馬番の組: オッズ}} と発表時刻(単勝の)を作る。"""
    from .jvodds import ODDS, parse_odds

    out: dict = {}
    announced = None
    for spec, code in RT_ODDS_SPECS.items():
        for rec in _rt_records(Path(raw_dir) / f"RT_{spec}.txt"):
            if rec[:2] != code or rec[11:27] != key:
                continue
            if code == "O1":
                announced = announced_at(rec)
            for kind, name in BET_KINDS.items():
                if ODDS[kind][0] != code:
                    continue
                tail = len(ODDS[kind][6]) + 1  # 組のあとに、オッズ(複勝・ワイドは下限と上限)と人気
                table = {r[:-tail]: r[-tail] for r in parse_odds(rec, kind)}
                if table:
                    out[name] = table
    return out, announced


def fetch_bet_odds(conn: sqlite3.Connection, race_id: str, raw_root: str | Path = RAW_DIR,
                   python32: str = PYTHON32) -> dict:
    """1レースの全券種の速報オッズを JRA-VAN から取って {券種: {馬番の組: オッズ}} にする(netkeiba の6回の取得の代わり)。

    JV-Link を種別ごとに1回ずつ、間をあけて開く(6種別で30秒ほど)。発売前などで取れない券種は入らない。
    """
    key = race_key(conn, race_id)
    raw = Path(raw_root) / "bet_odds"
    export(raw, key[:8], [], rt=[f"{s}:{key}" for s in RT_ODDS_SPECS], python32=python32)
    return read_bet_odds(raw, key)[0]


def snapshot_odds(conn: sqlite3.Connection, date: str, race_ids: list[str] | None = None,
                  now: dt.datetime | None = None, raw_root: str | Path = RAW_DIR, python32: str = PYTHON32) -> int:
    """date(YYYY-MM-DD)の各レースの単勝の速報オッズを JRA-VAN から取り、odds_snapshots に追記する。記録したレース数を返す。

    snapped_at は取った時刻、official_at は JRA-VAN のオッズの発表時刻。レースは出馬表を取り込み済みのもの(entry_races)。
    JV-Link は1回だけ起動し、レースごとに間をあけて開く。オッズのないレースは飛ばす。
    """
    from .jventries import parse_o1
    from .jvrace import to_bytes

    stamp = (now or dt.datetime.now()).isoformat(timespec="seconds")
    ids = race_ids or [r[0] for r in conn.execute("SELECT race_id FROM entry_races WHERE date = ? ORDER BY race_id", (date,))]
    keys = {}
    for rid in ids:
        try:
            keys[race_key(conn, rid)] = rid
        except ValueError as e:
            print(e)
    if not keys:
        return 0
    raw = Path(raw_root) / "snapshot"
    export(raw, date.replace("-", ""), [], rt=[f"0B31:{k}" for k in keys], python32=python32)
    n = 0
    seen = set()
    for rec in _rt_records(raw / "RT_0B31.txt"):
        o = parse_o1(to_bytes(rec))
        rid = keys.get(rec[11:27])
        if not o or not rid or rid in seen:
            continue
        rows = [(rid, num, stamp, announced_at(rec), win, pop) for num, (win, pop) in o[1].items() if win is not None]
        if not rows:
            print(f"skip {rid}: オッズがまだありません")
            continue
        with conn:
            conn.executemany("INSERT OR REPLACE INTO odds_snapshots (race_id, number, snapped_at, official_at, win_odds, "
                             "popularity) VALUES (?,?,?,?,?,?)", rows)
        seen.add(rid)
        n += 1
    return n
