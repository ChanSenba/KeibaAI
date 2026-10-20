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
    "training": ("調教の評価(netkeiba)", "oikiri", "SELECT COUNT(*) FROM races"),
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


@dataclass
class DataStatus:
    """DB にあるデータの揃い具合(JRA-VAN から入れたもの)。"""
    races: int
    last_race_date: str | None
    horses: int  # 成績に出てくる馬
    horses_with_pedigree: int
    training_rows: int
    last_training_date: str | None

    @property
    def pedigree_fraction(self) -> float | None:
        return self.horses_with_pedigree / self.horses if self.horses else None


def read_data_status(db_path: str | Path) -> DataStatus | None:
    """揃い具合を読み取り専用で1回だけ読む。DB が使えなければ None。"""
    try:
        conn = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True, timeout=1)
        try:
            one = lambda sql: conn.execute(sql).fetchone()
            races, last_race = one("SELECT COUNT(*), MAX(date) FROM races")
            horses, with_ped = one(
                "SELECT COUNT(*), COUNT(p.horse_id) FROM (SELECT DISTINCT horse_id FROM results WHERE horse_id IS NOT NULL) h "
                "LEFT JOIN pedigree p USING (horse_id)")
            try:  # 調教タイムの表がない古い DB もある
                rows = one("SELECT COALESCE(MAX(rowid), 0) FROM jv_training")[0]  # 件数の目安(数えるより速い)
                last_tr = one("SELECT MAX(date) FROM jv_training")[0]
            except sqlite3.Error:
                rows, last_tr = 0, None
            return DataStatus(races, last_race, horses, with_ped, rows, last_tr)
        finally:
            conn.close()
    except sqlite3.Error:
        return None


ODDS_STALE_SEC = 3600  # 開催中に、最後のオッズ記録からこの秒数を超えたら「止まっている」(発走前の記録は各レースの30分前と5分前なので、ふつうは30分も空かない)
ODDS_WINDOW_BEFORE_MIN = 60  # 最初のレースの発走の何分前から見張るか(朝9:00の全レースの記録がある)


@dataclass
class OddsStatus:
    """発走前オッズの記録の状態。state は running(開催中で記録が続いている)/ stalled(開催中なのに記録が途切れている)/ idle(開催中ではない)。"""
    state: str
    last_snapped: dt.datetime | None  # 最後に記録した時刻(日付を問わず)
    races_today: int  # 今日の出馬表のレース数(0なら非開催日)
    recorded_today: int  # 今日、1回以上記録できたレース数


def _clock(text: str | None) -> dt.time | None:
    try:
        h, m = (text or "").split(":")
        return dt.time(int(h), int(m))
    except ValueError:
        return None


def odds_status_from(rows_today: list[str | None], recorded_today: int, last_snapped: dt.datetime | None,
                     now: dt.datetime) -> OddsStatus:
    """今日の発走時刻の一覧と最後の記録時刻から、状態を決める。発走時刻が分からないときは 9:00〜16:45 とみなす。"""
    n = len(rows_today)
    times = [t for t in map(_clock, rows_today) if t]
    first = dt.datetime.combine(now.date(), min(times)) if times else dt.datetime.combine(now.date(), dt.time(9, 0))
    last = dt.datetime.combine(now.date(), max(times)) if times else dt.datetime.combine(now.date(), dt.time(16, 45))
    first -= dt.timedelta(minutes=ODDS_WINDOW_BEFORE_MIN)
    if not n or not first <= now <= last:
        return OddsStatus("idle", last_snapped, n, recorded_today)
    # 開催中。見張りを始めた時刻より前の記録は数えない(始まる前に前日の記録が残っていても「動いている」にしない)
    since = max(last_snapped, first) if last_snapped else first
    stalled = (now - since).total_seconds() > ODDS_STALE_SEC
    return OddsStatus("stalled" if stalled else "running", last_snapped, n, recorded_today)


def read_odds_status(db_path: str | Path, now: dt.datetime | None = None) -> OddsStatus | None:
    """オッズの記録(odds_snapshots)の状態を読み取り専用で1回だけ読む。DB が使えなければ None。"""
    now = now or dt.datetime.now()
    today = now.date().isoformat()
    try:
        conn = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True, timeout=1)
        try:
            starts = [r[0] for r in conn.execute("SELECT start_time FROM entry_races WHERE date = ?", (today,))]
            try:
                last = conn.execute("SELECT MAX(snapped_at) FROM odds_snapshots").fetchone()[0]
                rec = conn.execute("SELECT COUNT(DISTINCT race_id) FROM odds_snapshots WHERE snapped_at >= ?",
                                   (today,)).fetchone()[0]
            except sqlite3.Error:  # 記録の表がない古い DB
                last, rec = None, 0
            return odds_status_from(starts, rec, dt.datetime.fromisoformat(last) if last else None, now)
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
