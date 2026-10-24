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
from .parse import VENUES

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
        "SELECT c.race_id, c.venue, c.round, c.name, c.course_type, c.direction, c.course, c.rail, c.distance, "
        "c.ground, c.start_time, "
        "COUNT(e.number) AS runners FROM entry_races c LEFT JOIN entries e USING (race_id) "
        "WHERE c.date = ? GROUP BY c.race_id ORDER BY substr(c.race_id, 5, 2), c.round",
        conn, params=(date,),
    )


def day_courses(conn: sqlite3.Connection, date: str) -> dict[str, str]:
    """その日の各レースのコースの表記(race_id → 例: 芝・右回り 1600m)。"""
    return {rid: course_text(r) for rid, r in day_races(conn, date).set_index("race_id").iterrows()}


def race_entries(conn: sqlite3.Connection, race_id: str, prediction: pd.DataFrame | None = None) -> pd.DataFrame:
    """1レースの出馬表(馬番順)。予想があれば勝率と期待値の列を足す。"""
    df = pd.read_sql_query(
        "SELECT number, bracket, horse_name, sex_age, weight_carried, jockey, win_odds, popularity "
        "FROM entries WHERE race_id = ? ORDER BY number",
        conn, params=(race_id,),
    )
    if prediction is not None:
        cols = ["number", "p_win_ability", "p_win_market", "ev_ability", "ev_market"]
        cols += [c for c in ("p_win_quality", "p_win_fit", "p_win_rotation", "p_win_human", "p_win_track", "p_win_pedigree",
                                     "p_win_training") if c in prediction]
        cols += [c for c in ("mark", "star") if c in prediction]
        df = df.merge(prediction.loc[prediction["race_id"] == race_id, cols], on="number", how="left")
    return df


def race_choices(conn: sqlite3.Connection, date: str, listed_ids=()) -> pd.DataFrame:
    """1レースだけ取得・予想するときの選択肢。取得済みのレースと、レース一覧で知ったレースを合わせる。

    出馬表を取る前は名前がわからないので、競馬場とレース番号は race_id(年・場・回・日・R)から読む。
    """
    fetched = set(day_races(conn, date)["race_id"])
    ids = sorted(fetched | set(listed_ids), key=lambda r: (r[4:6], r[10:12]))
    return pd.DataFrame(
        [{"race_id": r, "venue": VENUES.get(r[4:6], r[4:6]), "round": int(r[10:12]), "fetched": r in fetched}
         for r in ids],
        columns=["race_id", "venue", "round", "fetched"],
    )


def fetch_entries_for(conn, fetcher, date: dt.date, on_race=None, race_ids: list[str] | None = None) -> int:
    from .predict import fetch_entries

    return fetch_entries(conn, fetcher, date.isoformat(), on_race=on_race, race_ids=race_ids)


def review_dates(conn: sqlite3.Connection) -> list[str]:
    """予想を保存してある日(新しい順)。"""
    return [r[0] for r in conn.execute("SELECT DISTINCT date FROM predictions ORDER BY date DESC")]


def backtest_run(conn: sqlite3.Connection, test_from: str, use_market: bool, ev_min: float,
                 max_odds: float | None, stake: int, blend: bool = False):
    """(レース数, まとめの表, ルールごとの精算済み買い目)。学習は test_from より前だけを使う。"""
    from . import backtest as bt
    from .features import build_features, load_results

    feats = build_features(load_results(conn))
    if blend:
        from .blend import predict_test_blend

        test, _ = predict_test_blend(feats, test_from, use_market=use_market)
    else:
        test = bt.predict_test(feats, test_from, use_market)
    rules = [("favorite", {}), ("top1", {}), ("ev", {"ev_min": ev_min, "max_odds": max_odds}), ("all", {})]
    table, settled = bt.run_rules(test, rules, stake)
    return int(test["race_id"].nunique()), table, settled


def text_or_blank(v) -> str:
    """表示用の文字。None や NaN(出馬表の欠けた項目)は空にする。"""
    return "" if v is None or v != v else str(v)


def course_text(race) -> str:
    """コースの表記。芝・ダート(障害)、回り、内回り・外回り、A〜E コース、距離。分からない項目は省く。

    例: 芝・右回り・外回り・Aコース 1600m、芝・直線 1000m。
    """
    distance = pd.to_numeric(race.get("distance"), errors="coerce")
    kind = text_or_blank(race.get("course_type"))
    way = text_or_blank(race.get("direction"))
    way = way if way == "直線" else (f"{way}回り" if way else "")
    inout = text_or_blank(race.get("course"))  # 外 → 外回り、外-内 → 外回り→内回り、内2周 → 内回り2周
    inout = "→".join(f"{c[0]}回り{c[1:]}" for c in inout.split("-")) if inout else ""
    rail = text_or_blank(race.get("rail"))
    rail = f"{rail}コース" if rail else ""
    head = "・".join(p for p in (kind, way, inout, rail) if p)
    far = f"{int(distance)}m" if distance == distance else ""
    return f"{head} {far}".strip()


def race_head_html(race) -> str:
    """競馬場・レース番号・名前・コースの見出しの HTML。出馬表に欠けた項目(レース名など)があっても落ちない。"""
    import html

    runners = pd.to_numeric(race.get("runners"), errors="coerce")
    start = text_or_blank(race.get("start_time"))
    meta = [course_text(race), text_or_blank(race.get("ground")), f"{start} 発走" if start else None,
            f"{int(runners)}頭" if runners == runners else None]
    place = f"{text_or_blank(race.get('venue'))}{int(race['round'])}R"
    return (f'<div class="kb-race-head"><span class="kb-race-no">{html.escape(place)}</span>'
            f'<span class="kb-race-name">{html.escape(text_or_blank(race.get("name")))}</span></div>'
            f'<div class="kb-race-meta">{html.escape(" ・ ".join(m for m in meta if m))}</div>')


MARKS = ["◎", "○", "▲"]
# ★の基準: オッズ込みモデルの期待値、勝率、オッズ。過去8年(2019年以降)の検証では、この基準の馬を単勝で買った回収率は
# 90〜97%(能力のみのモデルの期待値で選ぶと79.5%で、無作為に買うのと変わらなかった)。100%には届かない
STAR_EV, STAR_P, STAR_ODDS = 1.1, 0.03, 30.0


def add_marks(table: pd.DataFrame) -> pd.DataFrame:
    """予想印(mark 列)を付ける。予想の中身には手を入れず、出た勝率と期待値から決める。

    レースごとに、勝率の高い順に ◎ ○ ▲、続く 2 頭に △(9 頭立て以下は 1 頭)。
    勝率は印用の勝率(p_win_mix。ニューラルネットの能力のみの勝率と、予想を出した時点のオッズを混ぜたもの)を使い、
    そのレースにオッズがなければ能力のみを使う。印用の勝率のない古い予想では、オッズ込みのモデルの勝率を使う。
    当日9時のオッズでの検証(2026年)で、◎の的中は オッズ込みのモデル 31.5% → 32.4%(predict.mix_with_odds)。
    印のない馬のうち、期待値(オッズ込み。なければ能力のみ)がいちばん高く 1 以上の馬に ☆(穴)。
    能力のみの期待値は人気薄を高く見すぎる(100倍超の馬の勝率を実際の約4倍に見る)ので、オッズ込みを先に使う。
    2021〜2026年の検証で、☆の単勝の勝率は 1.0% → 2.1%、回収率は 69.5% → 75.8% になった(それでも得な印ではない)。
    別の star 列に、オッズ込みのモデルが、オッズより高く評価している馬(その期待値・勝率・オッズが基準以上)へ ★。
    オッズがない馬には付けない。
    """
    out = table.copy()
    out["mark"] = ""
    out["ev_star"] = out["ev_market"].fillna(out["ev_ability"]) if "ev_market" in out else out["ev_ability"]
    out["star"] = ((out["ev_market"] >= STAR_EV) & (out["p_win_market"] >= STAR_P)
                   & (out["win_odds"] <= STAR_ODDS)).map({True: "★", False: ""})
    for _, g in out.groupby("race_id", sort=False):
        p = next((c for c in ("p_win_mix", "p_win_market") if c in g and g[c].notna().all()), "p_win_ability")
        ranked = g[p].dropna().sort_values(ascending=False, kind="stable").index
        n_marked = len(MARKS) + (2 if len(g) >= 10 else 1)
        top = list(ranked[:n_marked])
        out.loc[top, "mark"] = (MARKS + ["△", "△"])[:len(top)]
        rest = g.drop(top)["ev_star"].dropna()
        if not rest.empty and rest.max() >= 1:
            out.loc[rest.idxmax(), "mark"] = "☆"
    return out.drop(columns="ev_star")


def predict_table(conn: sqlite3.Connection, date: dt.date) -> pd.DataFrame:
    from .predict import predict_day

    table = predict_day(conn, date.isoformat())
    table.attrs = {}  # 展開予想(表)が attrs に入っていると、concat や astype が attrs を比べて落ちる。画面では使わない
    return table


def fetch_bet_odds(fetcher, race_id: str, on_type=None, conn: sqlite3.Connection | None = None) -> dict:
    """1レースの全券種のオッズを取得して {券種: {馬番の組: オッズ}} にする。

    JV-Link を呼べる PC で conn(出馬表の入った DB)を渡せば、JRA-VAN の速報オッズを使う。
    そうでなければ netkeiba に6回アクセスする(3〜5秒おき)。
    """
    from . import jvlink
    from .entries import BET_ODDS_TYPES, parse_bet_odds

    if conn is not None and jvlink.available():
        try:
            return jvlink.fetch_bet_odds(conn, race_id)
        except jvlink.JVError as e:
            raise RuntimeError(str(e)) from e

    out: dict = {}
    for i, t in enumerate(BET_ODDS_TYPES):
        if on_type:
            on_type(i, len(BET_ODDS_TYPES), t)
        out.update(parse_bet_odds(fetcher.odds_json(race_id, t), t))
    return out


def mixed_probs(prediction: pd.DataFrame, race_id: str, odds: dict, blend) -> pd.DataFrame:
    """そのレースの各馬の勝率(p_win)。オッズ込みのモデルの勝率と、いま取ったオッズの確率を blend の重みで混ぜたもの。

    過去7年の検証で、能力のモデルと混ぜるより当たりやすかった(LogLoss 0.2053 → 0.2048。同じ買い方の回収率は
    馬連 86.7% → 90.4%、ワイド 86.8% → 90.0%)。能力のモデルは、混ぜても重みがほぼ 0 になり、オッズに何も足せていなかった。
    出馬表を取ったときにオッズがなくてオッズ込みの勝率が出せていない場合は、いまのオッズの確率だけを使う。
    """
    from .blend import Blend, apply_blend

    cols = ["race_id", "number", "p_win_ability"] + (["p_win_market"] if "p_win_market" in prediction.columns else [])
    one = prediction.loc[prediction["race_id"] == race_id, cols].copy()
    win = odds.get("単勝", {})
    one["win_odds"] = one["number"].map(lambda n: win.get((int(n),)))
    if "p_win_market" in one.columns and one["p_win_market"].notna().all():
        one["p_win"] = one["p_win_market"]
    else:  # オッズ込みの勝率がない。重みはオッズ込みの勝率に合わせて決めてあるので、能力の勝率には当てはめない
        one["p_win"] = one["p_win_ability"]
        blend = Blend(0.0, 1.0, blend.loglik, blend.baseline_loglik)
    if one["win_odds"].notna().sum() < 3:
        raise ValueError("単勝オッズのある馬が3頭に満たないため、買い方を出せません(発売前の可能性があります)")
    return apply_blend(one, blend)


def market_probs(odds: dict) -> pd.DataFrame:
    """単勝オッズだけから出した各馬の勝率(オッズの逆数を合計1にしたもの)。予想がないときに使う。"""
    win = {k[0]: v for k, v in odds.get("単勝", {}).items() if v and v > 0}
    if len(win) < 3:
        raise ValueError("単勝オッズのある馬が3頭に満たないため、買い方を出せません(発売前の可能性があります)")
    t = pd.DataFrame({"number": list(win), "win_odds": list(win.values())}).sort_values("number")
    t["p_win"] = (1 / t["win_odds"]) / (1 / t["win_odds"]).sum()
    return t.reset_index(drop=True)


def bet_plan_for(prediction: pd.DataFrame, race_id: str, odds: dict, blend, bankroll: int = 100_000,
                 cap: int = 10_000, kelly_scale: float = 0.5, equal_stake: int = 0):
    """予想(predict_table の結果)と全券種のオッズから、そのレースの買い方を出す。(買い方, 混ぜた勝率の表) を返す。

    勝率は mixed_probs のもの。
    """
    from .betting import make_plan

    mixed = mixed_probs(prediction, race_id, odds, blend)
    plan = make_plan([int(n) for n in mixed["number"]], mixed["p_win"].to_numpy(), odds, bankroll, cap, kelly_scale,
                     equal_stake=equal_stake)
    return plan, mixed


def load_user_marks(conn: sqlite3.Connection, race_id: str) -> dict[int, str]:
    """買い方ページで保存した、そのレースのユーザーの印 {馬番: 印}。"""
    return {int(n): m for n, m in conn.execute("SELECT number, mark FROM user_marks WHERE race_id = ?", (race_id,))}


def save_user_marks(conn: sqlite3.Connection, race_id: str, marks: dict[int, str], now: dt.datetime | None = None) -> None:
    """そのレースのユーザーの印を置き換える。空の印の馬は保存しない。"""
    stamp = (now or dt.datetime.now()).isoformat(timespec="seconds")
    with conn:
        conn.execute("DELETE FROM user_marks WHERE race_id = ?", (race_id,))
        conn.executemany("INSERT INTO user_marks (race_id, number, mark, updated_at) VALUES (?,?,?,?)",
                         [(race_id, int(n), m, stamp) for n, m in marks.items() if m])


def blend_for(conn: sqlite3.Connection, date: dt.date):
    """その日の予想に使う、モデルとオッズの混ぜ方(直前12か月の結果で決める。数分かかる)。"""
    from .predict import blend_for_day

    return blend_for_day(conn, date.isoformat())


def plan_table(bets) -> pd.DataFrame:
    return pd.DataFrame([{"券種": b.kind, "買い目": b.label, "オッズ": b.odds, "的中確率": b.prob, "期待値": b.ev,
                          "金額": b.stake, "的中時の払戻": int(b.stake * b.odds)} for b in bets])


def date_bounds(conn: sqlite3.Connection) -> tuple[dt.date | None, dt.date | None]:
    s = summary(conn)
    if not s["first"]:
        return None, None
    return dt.date.fromisoformat(s["first"]), dt.date.fromisoformat(s["last"])


def eval_start_bounds(first: dt.date, last: dt.date) -> tuple[dt.date, dt.date]:
    """評価開始日の (選べる最も早い日, 初期値)。学習には評価開始日より前のレースが要るので、最初の1年(短い期間なら前半)は選ばせない。
    初期値は、最後の日の1年前(それが最も早い日より前なら、最も早い日)。"""
    earliest = first + min(dt.timedelta(days=365), (last - first) / 2)
    return earliest, max(earliest, last - dt.timedelta(days=365))


def main() -> None:
    """`keibaai-gui` で Streamlit を起動し、ブラウザを開く。"""
    from streamlit.web import cli as stcli

    # ポートや配色は gui_app.py の隣の .streamlit/config.toml で設定している
    app = Path(__file__).with_name("gui_app.py")
    sys.argv = ["streamlit", "run", str(app)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
