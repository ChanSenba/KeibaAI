from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from . import db
import sys

from .fetch import Fetcher, ResultNotPublished, is_race_time
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
        except ResultNotPublished as e:  # その日はまだ掲載前。残りは飛ばし、失敗には数えない
            fetcher.discard(race_id)
            print(f"skip rest of {date}: {e}")
            break
        except ValueError as e:  # 結果表のないページなど
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


def reparse_cache(conn, cache_dir: str | Path) -> tuple[int, int]:
    """キャッシュ済みページを解析し直して DB を更新する(通信しない)。(成功, 失敗) を返す。"""
    ok = bad = 0
    for path in sorted((Path(cache_dir) / "race").glob("*.html")):
        try:
            db.save_race(conn, parse_race(path.read_text(encoding="utf-8"), path.stem))
            ok += 1
        except ValueError as e:
            bad += 1
            print(f"skip {path.stem}: {e}")
    return ok, bad


def daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def warn_if_race_time(now: dt.datetime | None = None) -> bool:
    """土日の日中は一般の利用者が多いので、夜間や平日に回すよう促す(止めはしない)。"""
    hit = is_race_time(now or dt.datetime.now())
    if hit:
        print("注意: 土日の日中は netkeiba が混み合う時間帯です。平日や夜間に実行するのがおすすめです。", file=sys.stderr)
    return hit


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="keibaai")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="期間内のJRAレース結果を取得してDBに保存")
    c.add_argument("--from", dest="start", required=True, help="YYYY-MM-DD")
    c.add_argument("--to", dest="end", required=True, help="YYYY-MM-DD")
    c.add_argument("--db", default="data/keiba.db")
    c.add_argument("--cache", default="data/cache")
    r = sub.add_parser("reparse", help="キャッシュ済みページを解析し直してDBを更新（通信なし）")
    r.add_argument("--db", default="data/keiba.db")
    r.add_argument("--cache", default="data/cache")
    t = sub.add_parser("train", help="特徴量を作ってLightGBMで学習し、時系列で評価する")
    t.add_argument("--db", default="data/keiba.db")
    t.add_argument("--test-from", required=True, help="この日付以降を評価用にする YYYY-MM-DD")
    e = sub.add_parser("entries", help="指定日の出馬表と単勝オッズを取得して保存")
    e.add_argument("--date", required=True, help="YYYY-MM-DD")
    e.add_argument("--db", default="data/keiba.db")
    e.add_argument("--cache", default="data/cache")
    pr = sub.add_parser("predict", help="指定日の各馬の勝率と期待値を表示(先に entries で出馬表を取得)")
    pr.add_argument("--date", required=True, help="YYYY-MM-DD")
    pr.add_argument("--db", default="data/keiba.db")
    args = p.parse_args(argv)

    if args.cmd in ("entries", "collect"):
        warn_if_race_time()
    if args.cmd == "entries":
        from .predict import fetch_entries

        n = fetch_entries(db.connect(args.db), Fetcher(cache_dir=args.cache), args.date)
        print(f"done: {n} races")
        return
    if args.cmd == "predict":
        from .predict import predict_day

        out = predict_day(db.connect(args.db), args.date)
        show = out.rename(columns={"venue": "場", "round": "R", "number": "番", "horse_name": "馬名",
                                   "win_odds": "オッズ", "p_win_ability": "勝率(能力)",
                                   "p_win_market": "勝率(オッズ込)", "ev_ability": "期待値(能力)",
                                   "ev_market": "期待値(オッズ込)"})
        print(show.drop(columns=["race_id", "date", "race_name", "jockey", "popularity"]).to_string(
            index=False, float_format=lambda x: f"{x:.2f}"))
        return

    if args.cmd == "train":
        from .features import build_features, load_results
        from .model import baseline_market, evaluate, format_report

        df = build_features(load_results(db.connect(args.db)))
        r_abil, test = evaluate(df, args.test_from, use_market=False)
        r_mkt, _ = evaluate(df, args.test_from, use_market=True)
        print(format_report([baseline_market(test), r_abil, r_mkt]))
        return

    if args.cmd == "reparse":
        ok, bad = reparse_cache(db.connect(args.db), args.cache)
        print(f"done: {ok} races reparsed, {bad} skipped")
        return

    conn = db.connect(args.db)
    fetcher = Fetcher(cache_dir=args.cache)
    total = 0
    failures: list[str] = []
    # JRA の開催は土日(祝日・月曜開催あり)。平日も確認して、レースが無い日は空で返る。
    for d in daterange(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)):
        if d.weekday() >= 5 or d.weekday() == 0:
            total += collect_day(conn, fetcher, d.strftime("%Y%m%d"), failures)
    print(f"done: {total} races")
