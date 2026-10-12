"""SQLite への保存。"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, fields
from pathlib import Path

from .parse import ParsedRace, Race, Result

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY,
    name TEXT, date TEXT, venue TEXT, round INTEGER,
    course_type TEXT, direction TEXT, distance INTEGER,
    weather TEXT, ground TEXT, start_time TEXT,
    lap_times TEXT, pace_first3f REAL, pace_last3f REAL
);
CREATE TABLE IF NOT EXISTS results (
    race_id TEXT NOT NULL REFERENCES races(race_id),
    number INTEGER NOT NULL,
    finish INTEGER, status TEXT, bracket INTEGER,
    horse_id TEXT, horse_name TEXT, sex_age TEXT, weight_carried REAL,
    jockey TEXT, jockey_id TEXT, time_sec REAL, margin TEXT, passing TEXT,
    last3f REAL, win_odds REAL, popularity INTEGER,
    body_weight INTEGER, body_weight_diff INTEGER,
    trainer TEXT, trainer_id TEXT, owner TEXT, prize REAL,
    PRIMARY KEY (race_id, number)
);
-- これから走るレース(出馬表)。結果の races とは別にしておく(同じ表だと結果の取得が飛ばされてしまう)
CREATE TABLE IF NOT EXISTS entry_races (
    race_id TEXT PRIMARY KEY,
    name TEXT, date TEXT, venue TEXT, round INTEGER,
    course_type TEXT, direction TEXT, distance INTEGER,
    weather TEXT, ground TEXT, start_time TEXT, fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS entries (
    race_id TEXT NOT NULL REFERENCES entry_races(race_id),
    number INTEGER NOT NULL,
    bracket INTEGER, horse_id TEXT, horse_name TEXT, sex_age TEXT, weight_carried REAL,
    jockey TEXT, jockey_id TEXT, trainer TEXT, trainer_id TEXT,
    body_weight INTEGER, body_weight_diff INTEGER, win_odds REAL, popularity INTEGER,
    PRIMARY KEY (race_id, number)
);
-- 発走前に出した予想をそのまま残す(あとから結果と照合する)。レースが終わってからは書き換えない
CREATE TABLE IF NOT EXISTS predictions (
    race_id TEXT NOT NULL, number INTEGER NOT NULL, saved_at TEXT,
    date TEXT, venue TEXT, round INTEGER, race_name TEXT, horse_name TEXT, jockey TEXT,
    win_odds REAL, popularity INTEGER, mark TEXT,
    p_win_ability REAL, p_win_market REAL, ev_ability REAL, ev_market REAL,
    PRIMARY KEY (race_id, number)
);
-- 血統(父・母・母の父)。馬ごとに1回取得する
CREATE TABLE IF NOT EXISTS pedigree (
    horse_id TEXT PRIMARY KEY,
    sire_id TEXT, sire TEXT, dam_id TEXT, dam TEXT, damsire_id TEXT, damsire TEXT,
    sire_sire_id TEXT, sire_sire TEXT, sire_line TEXT
);
-- 調教(追い切り)。1頭に調教が何本もある(seq=0 が最終追い切り)。評価だけのページでは seq=0 の1行で、タイムは空
CREATE TABLE IF NOT EXISTS training (
    race_id TEXT NOT NULL, number INTEGER NOT NULL, seq INTEGER NOT NULL,
    horse_id TEXT, date TEXT, course TEXT, slope INTEGER, ground TEXT, rider TEXT,
    f6 REAL, f5 REAL, f4 REAL, f3 REAL, f2 REAL, f1 REAL, position INTEGER, load TEXT, partner TEXT,
    critic TEXT, rank TEXT, review TEXT,
    PRIMARY KEY (race_id, number, seq)
);
-- 調教ページを読んだレース(表がなかったレースも入れて、取り直さないようにする)
CREATE TABLE IF NOT EXISTS training_fetched (race_id TEXT PRIMARY KEY, n_horses INTEGER, fetched_at TEXT);
CREATE INDEX IF NOT EXISTS idx_results_horse ON results(horse_id);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
    return conn


# 後から増えた列。作成済みの DB にも足す
_ADDED_COLUMNS = {"races": [("lap_times", "TEXT"), ("pace_first3f", "REAL"), ("pace_last3f", "REAL")]}


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, cols in _ADDED_COLUMNS.items():
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, typ in cols:
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")
    conn.commit()


def _upsert(conn: sqlite3.Connection, table: str, cls, obj) -> None:
    cols = [f.name for f in fields(cls)]
    values = asdict(obj)
    conn.execute(
        f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        [values[c] for c in cols],
    )


def save_race(conn: sqlite3.Connection, parsed: ParsedRace) -> None:
    with conn:
        _upsert(conn, "races", Race, parsed.race)
        conn.execute("DELETE FROM results WHERE race_id = ?", (parsed.race.race_id,))
        for r in parsed.results:
            _upsert(conn, "results", Result, r)


def has_race(conn: sqlite3.Connection, race_id: str) -> bool:
    return conn.execute("SELECT 1 FROM races WHERE race_id = ?", (race_id,)).fetchone() is not None


def save_entries(conn: sqlite3.Connection, parsed, fetched_at: str) -> None:
    """出馬表を保存する。同じレースを取り直したら置き換える(オッズの更新など)。"""
    from .entries import Entry, EntryRace

    with conn:
        _upsert(conn, "entry_races", EntryRace, parsed.race)
        conn.execute("UPDATE entry_races SET fetched_at = ? WHERE race_id = ?", (fetched_at, parsed.race.race_id))
        conn.execute("DELETE FROM entries WHERE race_id = ?", (parsed.race.race_id,))
        for e in parsed.entries:
            _upsert(conn, "entries", Entry, e)
