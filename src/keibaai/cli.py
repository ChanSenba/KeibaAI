from __future__ import annotations

import argparse
import datetime as dt

from . import db
from .fetch import Fetcher
from .parse import parse_race


def collect_day(conn, fetcher: Fetcher, date: str) -> int:
    n = 0
    for race_id in fetcher.race_ids_on(date):
        if db.has_race(conn, race_id):
            continue
        db.save_race(conn, parse_race(fetcher.race_html(race_id), race_id))
        n += 1
        print(f"saved {race_id}")
    return n


def daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="keibaai")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="期間内のJRAレース結果を取得してDBに保存")
    c.add_argument("--from", dest="start", required=True, help="YYYY-MM-DD")
    c.add_argument("--to", dest="end", required=True, help="YYYY-MM-DD")
    c.add_argument("--db", default="data/keiba.db")
    c.add_argument("--cache", default="data/cache")
    args = p.parse_args(argv)

    conn = db.connect(args.db)
    fetcher = Fetcher(cache_dir=args.cache)
    total = 0
    # JRA の開催は土日(祝日・月曜開催あり)。平日も確認して、レースが無い日は空で返る。
    for d in daterange(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)):
        if d.weekday() >= 5 or d.weekday() == 0:
            total += collect_day(conn, fetcher, d.strftime("%Y%m%d"))
    print(f"done: {total} races")
