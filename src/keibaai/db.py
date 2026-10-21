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
-- 発走前のオッズを時刻つきで溜める(確定オッズしかないと、オッズの動きは後から取れない)。snapped_at は取得した時刻、official_at はオッズの発表時刻。
-- source は、空ならその場で取った記録、'ts' なら JRA-VAN の時系列オッズ(レース後にまとめて取ったもので、snapped_at も発表時刻)
CREATE TABLE IF NOT EXISTS odds_snapshots (
    race_id TEXT NOT NULL, number INTEGER NOT NULL, snapped_at TEXT NOT NULL, official_at TEXT,
    win_odds REAL, popularity INTEGER, source TEXT,
    PRIMARY KEY (race_id, number, snapped_at)
);
-- 発走前に出した予想をそのまま残す(あとから結果と照合する)。レースが終わってからは書き換えない
CREATE TABLE IF NOT EXISTS predictions (
    race_id TEXT NOT NULL, number INTEGER NOT NULL, saved_at TEXT,
    date TEXT, venue TEXT, round INTEGER, race_name TEXT, horse_name TEXT, jockey TEXT,
    win_odds REAL, popularity INTEGER, mark TEXT,
    p_win_ability REAL, p_win_market REAL, ev_ability REAL, ev_market REAL,
    PRIMARY KEY (race_id, number)
);
-- 発走前の展開予想(レースごと)。回顧で、結果から出した実際の展開と比べる
CREATE TABLE IF NOT EXISTS flow_forecast (
    race_id TEXT PRIMARY KEY, n_front INTEGER, front_share REAL, course_edge REAL, course_pace_diff REAL,
    pace TEXT, advantage TEXT, basis TEXT, saved_at TEXT
);
-- 買い方ページでユーザーが付けた印(レースの馬ごと)。印を外した馬は行を消す
CREATE TABLE IF NOT EXISTS user_marks (
    race_id TEXT NOT NULL, number INTEGER NOT NULL, mark TEXT NOT NULL, updated_at TEXT,
    PRIMARY KEY (race_id, number)
);
-- 血統(父・母・母の父)。馬ごとに1回取得する
CREATE TABLE IF NOT EXISTS pedigree (
    horse_id TEXT PRIMARY KEY,
    sire_id TEXT, sire TEXT, dam_id TEXT, dam TEXT, damsire_id TEXT, damsire TEXT,
    sire_sire_id TEXT, sire_sire TEXT, sire_line TEXT
);
-- 5代血統(JRA-VAN の産駒マスタ・繁殖馬マスタから)。pos は 1=父, 2=母, 3=父の父, 4=父の母, 5=母の父, …(親は 2k+1 と 2k+2)で 1〜62。
-- たどれない先祖は ancestor_id が空。ancestor_id は JRA-VAN の繁殖登録番号
CREATE TABLE IF NOT EXISTS pedigree5 (
    horse_id TEXT NOT NULL, pos INTEGER NOT NULL, ancestor_id TEXT, name TEXT,
    PRIMARY KEY (horse_id, pos)
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
-- JRA-VAN Data Lab の調教タイム(坂路 slope・ウッド wood)。馬の番号は netkeiba の馬 ID と同じ。秒単位
CREATE TABLE IF NOT EXISTS jv_training (
    horse_id TEXT NOT NULL, date TEXT NOT NULL, time TEXT NOT NULL, kind TEXT NOT NULL,
    center TEXT, course TEXT, turn TEXT, f4 REAL, f3 REAL, f2 REAL, l1 REAL,
    PRIMARY KEY (horse_id, date, time, kind)
);
CREATE INDEX IF NOT EXISTS idx_results_horse ON results(horse_id);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
"""


LOCK_TIMEOUT = 60  # 他の接続が書き込み中のとき、書き込みの順番を待つ秒数(標準の5秒だと、長い取得と重なって落ちる)


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=LOCK_TIMEOUT)
    try:
        conn.executescript(SCHEMA)
        _add_missing_columns(conn)
        _install_model_revision(conn)
    except Exception:
        # commit失敗時の接続を残すと、画面の例外表示中も書き込みロックが残る。
        conn.close()
        raise
    return conn


def _install_model_revision(conn: sqlite3.Connection) -> None:
    """直接SQLの訂正も含め、学習に使う表の更新を保存済み予測へ反映する。"""
    tables = ("races", "results", "pedigree", "training", "jv_training")
    events = ("INSERT", "UPDATE", "DELETE")
    required = {f"model_revision_{table}_{event.lower()}" for table in tables for event in events}
    installed = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    if required <= installed and model_revision(conn) is not None:
        # INSERT OR IGNOREでも書き込みロックを取る。通常の画面表示ではSELECTだけで済ませる。
        return
    conn.execute("CREATE TABLE IF NOT EXISTS model_revision "
                 "(id INTEGER PRIMARY KEY CHECK (id = 1), generation TEXT NOT NULL, revision INTEGER NOT NULL)")
    conn.execute("INSERT OR IGNORE INTO model_revision VALUES (1, lower(hex(randomblob(16))), 0)")
    for table in tables:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        changed = " OR ".join(f'OLD."{c}" IS NOT NEW."{c}"' for c in cols)
        for event in events:
            when = f" WHEN {changed}" if event == "UPDATE" else ""
            conn.execute(f"CREATE TRIGGER IF NOT EXISTS model_revision_{table}_{event.lower()} "
                         f"AFTER {event} ON {table}{when} BEGIN "
                         "UPDATE model_revision SET revision = revision + 1 WHERE id = 1; END")
    conn.commit()


def model_revision(conn: sqlite3.Connection) -> tuple | None:
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_revision'").fetchone():
        return None  # 更新検知のない旧DB・読み取り専用接続では予測をキャッシュしない
    row = conn.execute("SELECT generation, revision FROM model_revision WHERE id = 1").fetchone()
    return tuple(row) if row is not None else None


# 後から増えた列。作成済みの DB にも足す
_ADDED_COLUMNS = {"races": [("lap_times", "TEXT"), ("pace_first3f", "REAL"), ("pace_last3f", "REAL"), ("course", "TEXT"), ("rail", "TEXT")],
                  "entry_races": [("course", "TEXT"), ("rail", "TEXT")],
                  "odds_snapshots": [("source", "TEXT")]}


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
