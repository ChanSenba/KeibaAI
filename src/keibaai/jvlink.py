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
                  python32: str = PYTHON32, now: dt.datetime | None = None) -> int:
    """その日(YYYY-MM-DD)の出馬表と単勝オッズを JRA-VAN から取って保存する。保存したレース数を返す。

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
    keys = [k for k in race_keys(raw, date) if k[:4] + k[8:] in numbered]
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
