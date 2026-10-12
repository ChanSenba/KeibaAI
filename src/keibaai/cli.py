from __future__ import annotations

import argparse
import datetime as dt

from . import db
from .fetch import Fetcher, PageUnavailable
from .parse import parse_race


MAX_CONSECUTIVE_FAILURES = 5


def collect_day(conn, fetcher: Fetcher, date: str, failures: list[str] | None = None) -> int:
    """1日分を保存し、保存したレース数を返す。結果表のないレースは飛ばす。

    連続で失敗が続く場合はアクセス制限の可能性があるため、中断する。
    """
    failures = failures if failures is not None else []
    n = 0
    for race_id in fetcher.race_ids_on(date):
        if db.has_race(conn, race_id):
            continue
        try:
            db.save_race(conn, parse_race(fetcher.race_html(race_id), race_id))
        except ValueError as e:  # PageUnavailable を含む
            fetcher.discard(race_id)  # 壊れたキャッシュを残さない
            failures.append(race_id)
            print(f"skip {race_id}: {e}")
            if len(failures) >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{MAX_CONSECUTIVE_FAILURES}件連続で取得に失敗しました。"
                    "アクセス制限の可能性があるので、時間をおいて再実行してください"
                ) from e
            continue
        failures.clear()
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
    failures: list[str] = []
    # JRA の開催は土日(祝日・月曜開催あり)。平日も確認して、レースが無い日は空で返る。
    for d in daterange(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)):
        if d.weekday() >= 5 or d.weekday() == 0:
            total += collect_day(conn, fetcher, d.strftime("%Y%m%d"), failures)
    print(f"done: {total} races")
