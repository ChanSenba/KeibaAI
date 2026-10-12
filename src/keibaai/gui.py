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

PORT = 8501


def summary(conn: sqlite3.Connection) -> dict:
    """取得済みデータの件数と期間。"""
    n_races, n_results, first, last = conn.execute(
        "SELECT (SELECT COUNT(*) FROM races), (SELECT COUNT(*) FROM results), MIN(date), MAX(date) FROM races"
    ).fetchone()
    return {"races": n_races, "results": n_results, "first": first, "last": last}


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
    from .model import baseline_favorite, evaluate

    df = build_features(load_results(conn))
    r_abil, test = evaluate(df, test_from, use_market=False)
    r_mkt, _ = evaluate(df, test_from, use_market=True)
    rows = [asdict(r) for r in (baseline_favorite(test), r_abil, r_mkt)]
    return pd.DataFrame(rows).rename(columns={
        "name": "モデル", "n_races": "レース数", "win_logloss": "勝LogLoss", "win_auc": "勝AUC",
        "top3_auc": "複AUC", "top1_hit_rate": "1番手的中", "top1_roi": "単勝回収率(%)",
    })


def date_bounds(conn: sqlite3.Connection) -> tuple[dt.date | None, dt.date | None]:
    s = summary(conn)
    if not s["first"]:
        return None, None
    return dt.date.fromisoformat(s["first"]), dt.date.fromisoformat(s["last"])


def main() -> None:
    """`keibaai-gui` で Streamlit を起動し、ブラウザを開く。"""
    from streamlit.web import cli as stcli

    app = Path(__file__).with_name("gui_app.py")
    sys.argv = ["streamlit", "run", str(app), f"--server.port={PORT}", "--server.address=localhost",
                "--browser.gatherUsageStats=false", "--server.showEmailPrompt=false",
                "--client.toolbarMode=minimal"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
