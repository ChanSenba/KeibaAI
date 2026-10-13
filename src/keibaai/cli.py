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
    pd_ = sub.add_parser("pedigree", help="馬ごとの血統(父・母の父)を取得して保存する。出馬表の馬が先。途中で止めても続きから")
    pd_.add_argument("--db", default="data/keiba.db")
    pd_.add_argument("--cache", default="data/cache")
    pd_.add_argument("--limit", type=int, default=0, help="この頭数で止める(0なら全部)。まず試すとき用")
    tr_ = sub.add_parser("training", help="調教の評価(短評とA〜D)を取得して保存する。新しいレースから。途中で止めても続きから")
    tr_.add_argument("--db", default="data/keiba.db")
    tr_.add_argument("--cache", default="data/cache")
    tr_.add_argument("--limit", type=int, default=0, help="このレース数で止める(0なら全部)。まず試すとき用")
    tr_.add_argument("--date", help="YYYY-MM-DD。出馬表を取った日のレースだけ取り直す(発走前の予想用)")
    tr_.add_argument("--reparse", action="store_true", help="保存済みページから読み直す(通信なし)")
    jv_ = sub.add_parser("jv", help="JRA-VAN から取り込む(調教タイム・過去の成績・血統)。32bit Python で JV-Link から書き出し、DB に入れる")
    jv_.add_argument("--what", nargs="+", choices=["training", "races", "pedigree"], default=["training"],
                     help="取り込むもの。training=坂路とウッドの調教タイム、races=レース成績、pedigree=全馬の血統(入れ替え)")
    jv_.add_argument("--replace", action="store_true",
                     help="races: すでに DB にあるレースも JRA-VAN の内容で入れ替える(既定は、無いレースだけ足す)")
    jv_.add_argument("--db", default="data/keiba.db")
    jv_.add_argument("--raw", default="data/jv_raw", help="書き出し先(生データ)")
    jv_.add_argument("--from", dest="from_date", default="20230901", help="取得を始める日 YYYYMMDD")
    jv_.add_argument("--option", type=int, default=4, help="JVOpen のオプション(4=セットアップ、1=通常の差分)")
    jv_.add_argument("--python32", default=r"C:\Python311-32\jvenv\Scripts\python.exe", help="pywin32 を入れた32bit Python")
    jv_.add_argument("--import-only", action="store_true", help="JV-Link は呼ばず、書き出し済みの生データだけ取り込む")
    bp = sub.add_parser("bets", help="1レースの買い方(券種と金額配分)を出す。先に entries で出馬表を取得しておく")
    bp.add_argument("--date", required=True, help="YYYY-MM-DD")
    bp.add_argument("--race-id", required=True, help="対象レースの race_id(12桁)")
    bp.add_argument("--bankroll", type=int, default=100_000, help="使える資金(円)")
    bp.add_argument("--cap", type=int, default=10_000, help="このレースに使う上限(円)")
    bp.add_argument("--kelly", type=float, default=0.5, help="ケリー基準をどれだけ抑えるか(0.5で半分)")
    bp.add_argument("--db", default="data/keiba.db")
    bp.add_argument("--cache", default="data/cache")
    fa = sub.add_parser("factors", help="要素(馬の質・騎手と厩舎・展開と馬場)ごとの予想の当たりやすさを比べる")
    fa.add_argument("--test-from", required=True, help="この日以後で評価する(YYYY-MM-DD)")
    fa.add_argument("--db", default="data/keiba.db")
    wk = sub.add_parser("weekly", help="毎週の処理: 前の週の予想を結果と照合して回顧を書き、今週末の予想を保存する")
    wk.add_argument("--db", default="data/keiba.db")
    wk.add_argument("--cache", default="data/cache")
    wk.add_argument("--reviews", default="data/reviews", help="回顧のファイルを置くフォルダ")
    wk.add_argument("--today", help="今日の日付(YYYY-MM-DD)。試すとき用")
    bt = sub.add_parser("backtest", help="過去のレースで単勝の買い方ルールを再現し、回収率などを表示")
    bt.add_argument("--test-from", required=True, help="この日以後を買う期間にする(YYYY-MM-DD)")
    bt.add_argument("--db", default="data/keiba.db")
    bt.add_argument("--market", action="store_true", help="オッズを入力に含めたモデルの勝率を使う")
    bt.add_argument("--ev", type=float, nargs="*", help="期待値の基準(複数可)。省略すると 1.0 / 1.2 / 1.5")
    bt.add_argument("--blend", action="store_true",
                    help="モデルの勝率とオッズの確率を混ぜた勝率を使う(混ぜ方は評価開始日の前の1年で決める)")
    bt.add_argument("--max-odds", type=float, help="これを超えるオッズの馬は買わない")
    bt.add_argument("--min-p", type=float, default=0.0, help="これ未満の勝率の馬は買わない(例: 0.03)")
    bt.add_argument("--grid", action="store_true", help="期待値 × 勝率の下限 × オッズの上限の組み合わせを一度に試す")
    bt.add_argument("--stake", type=int, default=100, help="1点の金額(円)")
    pr = sub.add_parser("predict", help="指定日の各馬の勝率と期待値を表示(先に entries で出馬表を取得)")
    pr.add_argument("--date", required=True, help="YYYY-MM-DD")
    pr.add_argument("--db", default="data/keiba.db")
    args = p.parse_args(argv)

    if args.cmd in ("entries", "collect", "weekly", "pedigree", "bets", "training"):
        warn_if_race_time()
    if args.cmd == "entries":
        from .predict import fetch_entries

        n = fetch_entries(db.connect(args.db), Fetcher(cache_dir=args.cache), args.date)
        print(f"done: {n} races")
        return
    if args.cmd == "bets":
        from . import gui
        from .predict import predict_day, blend_for_day

        conn = db.connect(args.db)
        try:
            pred = predict_day(conn, args.date)
        except ValueError as e:
            raise SystemExit(str(e))
        if args.race_id not in set(pred["race_id"]):
            raise SystemExit(f"{args.race_id} の出馬表がありません。先に entries で取得してください")
        print("モデルとオッズの混ぜ方を決めています(数分かかります)")
        blend = blend_for_day(conn, args.date)
        odds = gui.fetch_bet_odds(Fetcher(cache_dir=args.cache), args.race_id)
        plan, _ = gui.bet_plan_for(pred, args.race_id, odds, blend, args.bankroll, args.cap, args.kelly)
        print(f"混ぜ方: モデル {blend.a:.2f} / オッズ {blend.b:.2f}")
        if not plan.bets:
            print(plan.note)
            for b in plan.top_ev:
                print(f"  参考 {b.kind} {b.label} オッズ{b.odds:.1f} 的中{b.prob:.1%} 期待値{b.ev:.2f}")
            return
        print(gui.plan_table(plan.bets).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print(f"合計 {plan.total:,}円 / どれか的中 {plan.hit_prob:.0%} / 期待回収率 {plan.expected_return:.0%} / "
              f"期待収支 {plan.expected_profit:+,.0f}円 / 全部外れ {plan.worst:,}円")
        return
    if args.cmd == "pedigree":
        from .pedigree import collect_pedigrees, pending_horse_ids

        conn = db.connect(args.db)
        total = len(pending_horse_ids(conn))
        todo = min(total, args.limit) if args.limit else total
        print(f"血統が未取得の馬: 全部で{total}頭。今回は{todo}頭(1頭 3〜5秒かかるので、およそ {todo * 4 / 3600:.1f} 時間)")
        n = collect_pedigrees(conn, Fetcher(cache_dir=args.cache), args.limit)
        print(f"done: {n} horses")
        return
    if args.cmd == "jv":
        import os
        import subprocess
        from pathlib import Path

        from .jvrace import import_pedigree, import_races
        from .jvtraining import import_raw

        specs = {"training": ["SLOP", "WOOD"], "races": ["RACE"], "pedigree": ["DIFN"]}
        if not args.import_only:
            src = str(Path(__file__).resolve().parent.parent)
            env = {**os.environ, "PYTHONPATH": src}
            cmd = [args.python32, "-m", "keibaai.jvexport", "--from", args.from_date, "--out", args.raw,
                   "--option", str(args.option), "--specs", *[s for w in args.what for s in specs[w]]]
            if subprocess.run(cmd, env=env).returncode:
                raise SystemExit("JV-Link からの書き出しに失敗しました(上のメッセージを確認)")
        conn = db.connect(args.db)
        if "training" in args.what:
            print(f"done: 調教 {import_raw(conn, args.raw)} 件を取り込みました")
        if "races" in args.what:
            d = args.from_date
            n_race, n_res = import_races(conn, args.raw, f"{d[:4]}-{d[4:6]}-{d[6:]}", args.replace)
            print(f"done: 成績 {n_race} レース・{n_res} 頭を取り込みました")
        if "pedigree" in args.what:
            print(f"done: 血統 {import_pedigree(conn, args.raw)} 頭を入れ替えました")
        return
    if args.cmd == "training":
        from .training import collect_training, fetch_entry_training, pending_race_ids, reparse_training

        conn = db.connect(args.db)
        if args.reparse:
            ok, bad = reparse_training(conn, args.cache)
            print(f"done: {ok} races reparsed, {bad} skipped")
            return
        if args.date:
            n = fetch_entry_training(conn, Fetcher(cache_dir=args.cache), args.date)
            print(f"done: {n} races")
            return
        total = len(pending_race_ids(conn))
        todo = min(total, args.limit) if args.limit else total
        print(f"調教が未取得のレース: 全部で{total}。今回は{todo}(1レース 3〜5秒かかるので、およそ {todo * 4 / 3600:.1f} 時間)")
        n = collect_training(conn, Fetcher(cache_dir=args.cache), args.limit)
        print(f"done: {n} races")
        return
    if args.cmd == "weekly":
        from .weekly import run_weekly

        today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
        out = run_weekly(db.connect(args.db), Fetcher(cache_dir=args.cache), today, args.reviews)
        print(f"reviewed: {', '.join(out['reviewed']) or 'なし'}")
        for d, n in out["forecast"].items():
            print(f"forecast {d}: {n}頭" if n else f"forecast {d}: 出馬表がまだありません")
        return
    if args.cmd == "factors":
        from .factors import evaluate_factors, format_factor_report
        from .features import build_features, load_results

        reports, uniform = evaluate_factors(build_features(load_results(db.connect(args.db))), args.test_from)
        print(format_factor_report(reports, uniform))
        return
    if args.cmd == "backtest":
        from .backtest import DEFAULT_RULES, format_summary, predict_test, run_rules
        from .features import build_features, load_results

        feats = build_features(load_results(db.connect(args.db)))
        if args.blend:
            from .backtest import blend_summary
            from .blend import predict_test_blend

            test, blend = predict_test_blend(feats, args.test_from)
            print(blend_summary(test, blend))
        else:
            test = predict_test(feats, args.test_from, args.market)
        if args.grid:
            from .backtest import GRID_RULES
            rules = GRID_RULES
        else:
            rules = DEFAULT_RULES if not args.ev else [("favorite", {}), ("top1", {})] + [
                ("ev", {"ev_min": e, "max_odds": args.max_odds, "min_p": args.min_p}) for e in args.ev] + [("all", {})]
        table, _ = run_rules(test, rules, args.stake)
        print(f"{args.test_from} 以後 {test['race_id'].nunique()} レース / 単勝を毎回 {args.stake} 円")
        print(format_summary(table))
        return
    if args.cmd == "predict":
        from .predict import predict_day

        out = predict_day(db.connect(args.db), args.date)
        show = out.rename(columns={"venue": "場", "round": "R", "number": "番", "horse_name": "馬名",
                                   "win_odds": "オッズ", "p_win_ability": "勝率(能力)",
                                   "p_win_market": "勝率(オッズ込)", "ev_ability": "期待値(能力)",
                                   "ev_market": "期待値(オッズ込)", "p_win_quality": "馬の質", "p_win_fit": "適性", "p_win_rotation": "ローテ",
                                   "p_win_human": "騎手厩舎", "p_win_track": "展開馬場", "p_win_pedigree": "血統",
                                   "p_win_training": "調教"})
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
