"""ブラウザで操作する GUI の起動と、画面から呼ぶ処理。

画面そのものは gui_app.py(Streamlit)。ここは CLI と同じ関数を呼ぶ薄い層にとどめる。
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import db
from .cli import collect_day, daterange

def summary(conn: sqlite3.Connection) -> dict:
    """取得済みデータの件数と期間。"""
    n_races, n_results, n_horses, first, last = conn.execute(
        "SELECT (SELECT COUNT(*) FROM races), (SELECT COUNT(*) FROM results), "
        "(SELECT COUNT(DISTINCT horse_id) FROM results), MIN(date), MAX(date) FROM races"
    ).fetchone()
    return {"races": n_races, "results": n_results, "horses": n_horses, "first": first, "last": last}


def races_by_month(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT substr(date, 1, 7) AS month, COUNT(*) AS races FROM races "
        "WHERE date IS NOT NULL GROUP BY month ORDER BY month",
        conn,
    )


def races_by_venue(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT venue, COUNT(*) AS races FROM races GROUP BY venue ORDER BY races DESC", conn
    )


def recent_races(conn: sqlite3.Connection, limit: int = 50) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT c.date, c.venue, c.round, c.name, c.course_type, c.distance, c.ground, "
        "COUNT(r.number) AS runners FROM races c LEFT JOIN results r USING (race_id) "
        "GROUP BY c.race_id ORDER BY c.date DESC, c.venue, c.round LIMIT ?",
        conn,
        params=(limit,),
    )


PRESETS = ["先週末", "1か月", "3か月", "今年"]


def preset_range(name: str, today: dt.date) -> tuple[dt.date, dt.date]:
    """取得期間のよく使う選び方。「先週末」は直近の終わった土日(月曜開催を含む)。"""
    if name == "先週末":
        sat = today - dt.timedelta(days=(today.weekday() - 5) % 7 or 7)
        return sat, min(sat + dt.timedelta(days=2), today)
    if name == "1か月":
        return today - dt.timedelta(days=30), today
    if name == "3か月":
        return today - dt.timedelta(days=91), today
    if name == "今年":
        return dt.date(today.year, 1, 1), today
    raise ValueError(name)


def race_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """CLI の collect と同じく、土日と月曜(祝日開催)だけを対象にする。"""
    return [d for d in daterange(start, end) if d.weekday() >= 5 or d.weekday() == 0]


def collect(conn, fetcher, start: dt.date, end: dt.date, on_day=None) -> int:
    """期間内を取得して保存したレース数を返す。on_day(i, n, date) で進み具合を知らせる。"""
    days = race_days(start, end)
    failures: list[str] = []
    total = 0
    for i, d in enumerate(days):
        if on_day:
            on_day(i, len(days), d)
        total += collect_day(conn, fetcher, d.strftime("%Y%m%d"), failures)
    if on_day:
        on_day(len(days), len(days), None)
    return total


def train_reports(conn: sqlite3.Connection, test_from: str) -> pd.DataFrame:
    """CLI の train と同じ3行(基準・オッズなし・オッズあり)を表にして返す。"""
    from .features import build_features, load_results
    from .model import baseline_market, evaluate

    df = build_features(load_results(conn))
    r_abil, test = evaluate(df, test_from, use_market=False)
    r_mkt, _ = evaluate(df, test_from, use_market=True)
    rows = [asdict(r) for r in (baseline_market(test), r_abil, r_mkt)]
    return pd.DataFrame(rows).rename(columns={
        "name": "モデル", "n_races": "レース数", "win_logloss": "勝LogLoss", "win_auc": "勝AUC",
        "top3_auc": "複AUC", "top1_hit_rate": "1番手的中", "top1_roi": "単勝回収率(%)",
    })


def next_race_day(today: dt.date) -> dt.date:
    """予想の初期日。今日が土日・月曜(祝日開催)ならその日、平日なら次の土曜。"""
    if today.weekday() >= 5 or today.weekday() == 0:
        return today
    return today + dt.timedelta(days=5 - today.weekday())


def entries_summary(conn: sqlite3.Connection, date: str) -> dict:
    """保存済みの出馬表の件数と取得時刻。"""
    n_races, n_horses, with_odds, fetched = conn.execute(
        "SELECT COUNT(DISTINCT c.race_id), COUNT(e.number), COUNT(e.win_odds), MAX(c.fetched_at) "
        "FROM entry_races c LEFT JOIN entries e USING (race_id) WHERE c.date = ?", (date,)
    ).fetchone()
    return {"races": n_races, "horses": n_horses, "with_odds": with_odds, "fetched_at": fetched}


def day_races(conn: sqlite3.Connection, date: str) -> pd.DataFrame:
    """出馬表を取得済みのレース一覧。競馬場は race_id の場コード順(札幌→小倉)、その中はレース番号順。"""
    return pd.read_sql_query(
        "SELECT c.race_id, c.venue, c.round, c.name, c.course_type, c.distance, c.ground, c.start_time, "
        "COUNT(e.number) AS runners FROM entry_races c LEFT JOIN entries e USING (race_id) "
        "WHERE c.date = ? GROUP BY c.race_id ORDER BY substr(c.race_id, 5, 2), c.round",
        conn, params=(date,),
    )


def race_entries(conn: sqlite3.Connection, race_id: str, prediction: pd.DataFrame | None = None) -> pd.DataFrame:
    """1レースの出馬表(馬番順)。予想があれば勝率と期待値の列を足す。"""
    df = pd.read_sql_query(
        "SELECT number, bracket, horse_name, sex_age, weight_carried, jockey, win_odds, popularity "
        "FROM entries WHERE race_id = ? ORDER BY number",
        conn, params=(race_id,),
    )
    if prediction is not None:
        cols = ["number", "p_win_ability", "p_win_market", "ev_ability", "ev_market"]
        df = df.merge(prediction.loc[prediction["race_id"] == race_id, cols], on="number", how="left")
    return df


def fetch_entries_for(conn, fetcher, date: dt.date, on_race=None) -> int:
    from .predict import fetch_entries

    return fetch_entries(conn, fetcher, date.isoformat(), on_race=on_race)


def predict_table(conn: sqlite3.Connection, date: dt.date) -> pd.DataFrame:
    from .predict import predict_day

    return predict_day(conn, date.isoformat())


def date_bounds(conn: sqlite3.Connection) -> tuple[dt.date | None, dt.date | None]:
    s = summary(conn)
    if not s["first"]:
        return None, None
    return dt.date.fromisoformat(s["first"]), dt.date.fromisoformat(s["last"])


def main() -> None:
    """`keibaai-gui` で Streamlit を起動し、ブラウザを開く。"""
    from streamlit.web import cli as stcli

    # ポートや配色は gui_app.py の隣の .streamlit/config.toml で設定している
    app = Path(__file__).with_name("gui_app.py")
    sys.argv = ["streamlit", "run", str(app)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
