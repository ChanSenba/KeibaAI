"""JRA-VAN の地方競馬(区分 A)・海外(区分 B)の成績を別の DB に入れ、中央の出走に「地方・海外を含めた前走」を足す。

中央の成績(jvrace.py)だけだと、地方や海外を走って戻ってきた馬は、本当の前走が抜けて1つ前の中央のレースが
前走になる。地方から転入した馬は、地方での経歴がまるごと抜ける。RACE.txt にはその走りも入っているので読む。
RA・SE の位置は中央と同じ(jvrace.py の説明を参照)。地方の場コードは 30〜61、海外は英字(A4 など)。
海外はタイムが 0、外国馬は馬の番号が 0000000000。古い書き出しは日本語の欄が化けていることがあるが、数字の欄は読める。

検証用。keiba.db には書かない。
    python -m keibaai.jvother --raw data/jv_raw data/jv_raw_old --db data/jv_other.db
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .jvrace import STATUS, _num, _records, _track

KUBUNS = {b"A": "地方", b"B": "海外"}
COLUMNS = ["race_id", "number", "horse_id", "date", "region", "venue_code", "course_type", "distance",
           "finish", "status", "time_sec"]
SCHEMA = """
CREATE TABLE IF NOT EXISTS other_runs (
    race_id TEXT NOT NULL, number INTEGER NOT NULL, horse_id TEXT NOT NULL, date TEXT NOT NULL, region TEXT,
    venue_code TEXT, course_type TEXT, distance INTEGER, finish INTEGER, status TEXT, time_sec REAL,
    PRIMARY KEY (race_id, number)
);
CREATE INDEX IF NOT EXISTS idx_other_horse ON other_runs(horse_id);
"""


def _rid(b: bytes) -> str:
    return (b[11:15] + b[19:27]).decode("ascii", errors="replace")


def parse_other_ra(b: bytes) -> tuple | None:
    """地方・海外の RA → (race_id, 芝/ダ, 距離)。対象外は None。"""
    if b[:2] != b"RA" or b[2:3] not in KUBUNS or len(b) < 707 or not b[11:19].isdigit():
        return None
    return _rid(b), _track(b[705:707].decode("ascii", errors="replace"))[0], _num(b[697:701])


def parse_other_se(b: bytes) -> tuple | None:
    """地方・海外の SE → COLUMNS の順(芝ダ・距離は RA から埋めるので None)。番号のない外国馬は None。"""
    if b[:2] != b"SE" or b[2:3] not in KUBUNS or len(b) < 342 or not b[11:19].isdigit() or not b[28:30].isdigit():
        return None
    horse = b[30:40].decode("ascii", errors="replace")
    if not horse.isdigit() or int(horse) == 0:
        return None
    status = STATUS.get(chr(b[331]))
    t = b[338:342].decode("ascii", errors="replace")
    time_sec = int(t[0]) * 60 + int(t[1:]) / 10 if t.isdigit() and int(t) > 0 else None
    ymd = b[11:19].decode()
    return (_rid(b), int(b[28:30]), horse, f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}", KUBUNS[b[2:3]],
            b[19:21].decode("ascii", errors="replace"), None, None,
            None if status else _num(b[334:336]), status, time_sec)


def import_other(conn: sqlite3.Connection, raw_dirs: list[str | Path]) -> int:
    """raw_dirs の RACE.txt から地方・海外の出走を other_runs に入れる(同じ出走は入れ替え)。入れた件数を返す。"""
    conn.executescript(SCHEMA)
    races, runs = {}, {}
    for d in raw_dirs:
        path = Path(d) / "RACE.txt"
        if not path.exists():
            continue
        for b in _records(path):
            if b[:2] == b"RA":
                r = parse_other_ra(b)
                if r:
                    races[r[0]] = r[1:]
            elif b[:2] == b"SE":
                r = parse_other_se(b)
                if r:
                    runs[(r[0], r[1])] = r
    rows = [(*r[:6], *races.get(r[0], (None, None)), *r[8:]) for r in runs.values()]
    with conn:
        conn.executemany(f"INSERT OR REPLACE INTO other_runs ({','.join(COLUMNS)}) "
                         f"VALUES ({','.join('?' * len(COLUMNS))})", rows)
    return len(rows)


def load_other(path: str | Path) -> pd.DataFrame:
    """other_runs を読む(読み取り専用)。頭数は同じレースの出走数。"""
    conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query("SELECT * FROM other_runs", conn)
    finally:
        conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df["field_size"] = df.groupby("race_id")["number"].transform("count")
    return df


OTHER_COLUMNS = ["days_since_any", "weeks_since_any", "is_layoff_any", "prev_is_other", "prev_other_finish", "prev_other_finish_rel", "prev_other_abroad",
                 "n_other_prev", "other_win_rate", "other_top3_rate", "n_any_prev", "runs_since_jra"]


def add_other_features(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """中央の出走(horse_id・date の列)に、地方・海外を含めた前走の列を足す。その日より前の走りだけを使う。

    days_since_any / weeks_since_any / is_layoff_any: 地方・海外を含めた前走からの日数・週数・90日以上か。prev_is_other: その前走が地方・海外か。
    prev_other_finish / _rel / _abroad: 前走が地方・海外のときの着順、着順÷頭数、海外か。
    n_other_prev / other_win_rate / other_top3_rate: それまでの地方・海外の出走数と勝率・3着内率。
    n_any_prev: 中央を含めた出走数。runs_since_jra: 前の中央のレースのあとに走った地方・海外の数。
    """
    out = {c: pd.Series(np.nan, index=df.index) for c in OTHER_COLUMNS}
    q = pd.DataFrame({"row": df.index, "horse_id": df["horse_id"], "day": pd.to_datetime(df["date"])})
    o = other[other["horse_id"].isin(set(q["horse_id"].dropna()))].sort_values(["horse_id", "date"]).copy()
    if not o.empty:
        g = o.groupby("horse_id", sort=False)
        o["k"] = g.cumcount() + 1
        o["wins"] = (o["finish"] == 1).astype(int).groupby(o["horse_id"], sort=False).cumsum()
        o["top3"] = (o["finish"] <= 3).astype(int).groupby(o["horse_id"], sort=False).cumsum()
        # 同じ日の地方・海外の走りは数えない(その日の中央のレースと前後が分からない)
        last = pd.merge_asof(q.sort_values("day"), o.sort_values("date"), left_on="day", right_on="date",
                             by="horse_id", allow_exact_matches=False).set_index("row").reindex(df.index)
        has = last["date"].notna()
        out["n_other_prev"] = last["k"].fillna(0.0)
        out["other_win_rate"] = (last["wins"] / last["k"]).where(has)
        out["other_top3_rate"] = (last["top3"] / last["k"]).where(has)
        other_last = last["date"]
    else:
        has = pd.Series(False, index=df.index)
        out["n_other_prev"] = pd.Series(0.0, index=df.index)
        other_last = pd.Series(pd.NaT, index=df.index)
    g = df.groupby("horse_id", sort=False)
    jra_last = pd.to_datetime(g["date"].shift(1))
    day = pd.to_datetime(df["date"])
    is_other = has & (jra_last.isna() | (other_last > jra_last))
    any_last = other_last.where(is_other, jra_last)
    out["days_since_any"] = (day - any_last).dt.days
    out["weeks_since_any"] = out["days_since_any"] / 7
    out["is_layoff_any"] = (out["days_since_any"] >= 90).astype(float).where(out["days_since_any"].notna())
    out["prev_is_other"] = is_other.astype(float).where(any_last.notna())
    if not o.empty:
        out["prev_other_finish"] = last["finish"].where(is_other)
        out["prev_other_finish_rel"] = (last["finish"] / last["field_size"]).where(is_other)
        out["prev_other_abroad"] = (last["region"] == "海外").astype(float).where(is_other)
        # 前の中央のレースの日より後の地方・海外の走りの数
        before = pd.merge_asof(pd.DataFrame({"row": df.index, "horse_id": df["horse_id"], "day": jra_last})
                               .dropna(subset=["day"]).sort_values("day"),
                               o[["horse_id", "date", "k"]].sort_values("date"), left_on="day", right_on="date",
                               by="horse_id").set_index("row")["k"].reindex(df.index).fillna(0.0)
        out["runs_since_jra"] = (out["n_other_prev"] - before).where(jra_last.notna(), out["n_other_prev"])
    else:
        out["runs_since_jra"] = pd.Series(0.0, index=df.index)
    out["n_any_prev"] = g.cumcount().astype(float) + out["n_other_prev"]
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="JRA-VAN の地方・海外の成績を別の DB に入れる(keiba.db には書かない)")
    ap.add_argument("--raw", nargs="+", default=["data/jv_raw"], help="RACE.txt のあるフォルダ(複数可)")
    ap.add_argument("--db", default="data/jv_other.db")
    args = ap.parse_args(argv)
    conn = sqlite3.connect(args.db)
    n = import_other(conn, args.raw)
    rows = conn.execute("SELECT region, COUNT(*), COUNT(DISTINCT horse_id), MIN(date), MAX(date) FROM other_runs "
                        "GROUP BY region").fetchall()
    conn.close()
    print(f"{n:,} 件を入れました: {args.db}")
    for r in rows:
        print(f"  {r[0]}: {r[1]:,} 件, {r[2]:,} 頭, {r[3]} 〜 {r[4]}")


if __name__ == "__main__":
    main()
