"""取得の進み具合。取得中の DB には触れず、保存済みページのフォルダのファイル数と更新時刻だけから見積もる。

(取得中に画面から DB を読むと、書き込み側が database is locked で落ちることがあったため。)
DB から読むのは総数だけで、読み取り専用で1回だけ。
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

ALIVE_SEC = 120  # 最後のページの保存からこの秒数以内なら「動いている」とみなす(取得は3〜5秒おき)
RATE_WINDOW_SEC = 1800  # 速さは直近30分から出す

# 取得の種類: (画面での名前, 保存済みページの下位フォルダ, 総数を数える SQL)
KINDS = {
    "training": ("調教の評価", "oikiri", "SELECT COUNT(*) FROM races"),
    "pedigree": ("血統", "ped", "SELECT COUNT(DISTINCT horse_id) FROM results WHERE horse_id IS NOT NULL"),
}


@dataclass
class Progress:
    name: str
    done: int
    total: int | None
    last_saved: dt.datetime | None  # 最後にページを保存した時刻
    per_min: float  # 直近の速さ(ページ/分)。分からなければ 0

    @property
    def fraction(self) -> float | None:
        return min(self.done / self.total, 1.0) if self.total else None

    @property
    def alive(self) -> bool:
        return self.last_saved is not None and (dt.datetime.now() - self.last_saved).total_seconds() <= ALIVE_SEC

    @property
    def eta_hours(self) -> float | None:
        if not self.total or self.per_min <= 0 or self.done >= self.total:
            return None
        return (self.total - self.done) / self.per_min / 60


def scan_folder(folder: str | Path, now: float | None = None) -> tuple[int, float | None, int]:
    """(ファイル数, 最後の更新時刻(epoch), 直近 RATE_WINDOW_SEC 秒に保存された数)。フォルダがなければ (0, None, 0)。"""
    now = now if now is not None else dt.datetime.now().timestamp()
    n = recent = 0
    last = None
    try:
        with os.scandir(folder) as it:
            for e in it:
                if not e.name.endswith(".html"):
                    continue
                m = e.stat().st_mtime
                n += 1
                last = m if last is None or m > last else last
                recent += now - m <= RATE_WINDOW_SEC
    except FileNotFoundError:
        return 0, None, 0
    return n, last, recent


def read_total(db_path: str | Path, sql: str) -> int | None:
    """総数を読み取り専用で1回だけ読む。DB が使えなければ None(画面は件数だけ出す)。"""
    try:
        conn = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True, timeout=1)
        try:
            return conn.execute(sql).fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def progress_of(kind: str, cache_dir: str | Path, total: int | None, now: float | None = None) -> Progress:
    name, sub, _ = KINDS[kind]
    now = now if now is not None else dt.datetime.now().timestamp()
    n, last, recent = scan_folder(Path(cache_dir) / sub, now)
    # 直近30分に満たない新しい取得でも速さが出るよう、窓の長さを実際の経過に合わせる
    per_min = recent / (RATE_WINDOW_SEC / 60) if recent else 0.0
    return Progress(name, n, total, dt.datetime.fromtimestamp(last) if last else None, per_min)


def tail(path: str | Path, lines: int = 8) -> str:
    """ログの最後の数行。ファイルがなければ空。"""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])
