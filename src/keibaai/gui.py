"""ブラウザで操作する GUI の起動と、画面から呼ぶ処理。

画面そのものは gui_app.py(Streamlit)。ここは CLI と同じ関数を呼ぶ薄い層にとどめる。
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from .cli import collect_day, daterange
from .marks import top_marks
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


REPORT_COLUMNS = {"name": "モデル", "n_races": "レース数", "win_logloss": "勝LogLoss", "win_auc": "勝AUC",
                  "top3_auc": "複AUC", "top1_hit_rate": "1番手的中", "top1_roi": "単勝回収率(%)"}


def load_evals(evals_dir: str | Path) -> list[dict]:
    """日曜の評価(weekly.write_evaluation の <日付>.json)を新しい順に。各要素は date・test_from・report(表)。"""
    import json

    out = []
    for f in sorted(Path(evals_dir).glob("*.json"), reverse=True):
        try:
            e = json.loads(f.read_text(encoding="utf-8"))
            out.append({"date": e["date"], "test_from": e["test_from"],
                        "report": pd.DataFrame(e["reports"]).rename(columns=REPORT_COLUMNS)})
        except (ValueError, KeyError):  # 書きかけ・壊れたファイルは飛ばす
            continue
    return out


def eval_history(evals: list[dict]) -> pd.DataFrame:
    """日曜の評価の週ごとの推移(古い順): 日付と、モデルごとの勝LogLoss・勝AUC・単勝回収率。"""
    rows = []
    for e in evals:
        for _, r in e["report"].iterrows():
            rows.append({"日付": e["date"], "モデル": r["モデル"], "勝LogLoss": r["勝LogLoss"], "勝AUC": r["勝AUC"],
                         "単勝回収率(%)": r["単勝回収率(%)"]})
    return pd.DataFrame(rows).sort_values("日付").reset_index(drop=True) if rows else pd.DataFrame()


def live_totals(trend: pd.DataFrame) -> dict:
    """実運用の推移(weekly.live_trend)の通算。回収率は、日ごとの回収率を買った数(◎はレース数)で重み付けて合わせる。"""
    if trend.empty:
        return {"days": 0, "races": 0, "honmei_wins": 0, "honmei_roi": float("nan"), "ev_bets": 0, "ev_roi": float("nan"),
                "star_bets": 0, "star_roi": float("nan")}

    def wavg(col: str, w: pd.Series) -> float:
        ok = trend[col].notna() & (w > 0)
        return float((trend.loc[ok, col] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")

    star_n = trend["★の頭数"] if "★の頭数" in trend else pd.Series(0, index=trend.index)  # ★を測り始める前の推移には列がない
    out = {"days": len(trend), "races": int(trend["レース数"].sum()), "honmei_wins": int(trend["◎1着"].sum()),
           "honmei_roi": wavg("◎単勝回収率(%)", trend["レース数"]), "ev_bets": int(trend["期待値≧1の頭数"].sum()),
           "ev_roi": wavg("期待値≧1の回収率(%)", trend["期待値≧1の頭数"]), "star_bets": int(star_n.sum())}
    out["star_roi"] = wavg("★単勝回収率(%)", star_n) if "★単勝回収率(%)" in trend else float("nan")
    return out


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


def review_dates(conn: sqlite3.Connection) -> list[str]:
    """予想を保存してある日(新しい順)。"""
    return [r[0] for r in conn.execute("SELECT DISTINCT date FROM predictions ORDER BY date DESC")]


def backtest_run(conn: sqlite3.Connection, test_from: str, use_market: bool, ev_min: float,
                 max_odds: float | None, stake: int, blend: bool = False):
    """(レース数, まとめの表, ルールごとの精算済み買い目)。学習は test_from より前だけを使う。"""
    from . import backtest as bt
    from .features import features_from_db

    from .nn_year import use_db

    use_db(conn)  # オッズ込みのモデルが使う年ごとのネットを、DB の隣に保存して使い回す
    feats = features_from_db(conn)
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


# ★の基準: オッズ込みモデルの期待値、勝率、オッズ。過去8年(2019年以降)の検証では、この基準の馬を単勝で買った回収率は
# 90〜97%(能力のみのモデルの期待値で選ぶと79.5%で、無作為に買うのと変わらなかった)。100%には届かない
STAR_EV, STAR_P, STAR_ODDS = 1.1, 0.03, 30.0


def add_marks(table: pd.DataFrame) -> pd.DataFrame:
    """予想印(mark 列)を付ける。予想の中身には手を入れず、出た勝率と期待値から決める。

    ◎ ○ ▲ △ は、能力のみの勝率(p_win_ability。オッズを見ない)の高い順。続く 2 頭に △(9 頭立て以下は 1 頭)。
    印は「AI が能力をどう見たか」を示すもので、人気(オッズ)は混ぜない。2026-11 まではオッズを混ぜた勝率で付けていて、
    保存した139レースで◎が確定の1番人気だった割合が76%(オッズを取ったあとは79%)と、ほぼ人気順になっていた。
    オッズは、買うかどうかの判断(☆・★・期待値)にだけ使う。
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
        ranked = g["p_win_ability"].dropna().sort_values(ascending=False, kind="stable").index
        marks = top_marks(len(g))
        top = list(ranked[:len(marks)])
        out.loc[top, "mark"] = marks[:len(top)]
        rest = g.drop(top)["ev_star"].dropna()
        if not rest.empty and rest.max() >= 1:
            out.loc[rest.idxmax(), "mark"] = "☆"
    return out.drop(columns="ev_star")


def predict_with_flow(conn: sqlite3.Connection, date: dt.date) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """予想の表と、展開予想(attrs["flow"])を分けて返す。表の attrs は空にする。"""
    from .predict import predict_day

    table = predict_day(conn, date.isoformat())
    flow = table.attrs.get("flow")
    table.attrs = {}  # 展開予想(表)が attrs に入っていると、concat や astype が attrs を比べて落ちる。画面では使わない
    return table, flow


def predict_table(conn: sqlite3.Connection, date: dt.date) -> pd.DataFrame:
    return predict_with_flow(conn, date)[0]


def saved_prediction(conn: sqlite3.Connection, date: dt.date) -> pd.DataFrame | None:
    """その日の予想がすでに計算・保存されていれば(自動の予想 weekly など)、印を付けた表を返す。なければ None。"""
    from .predict import saved_predict_day

    raw = saved_predict_day(conn, date.isoformat())
    if raw is None:
        return None
    raw.attrs = {}
    return add_marks(raw)


def predict_and_save(conn: sqlite3.Connection, date: dt.date) -> pd.DataFrame:
    """予想して、回顧用に(展開予想も)保存し、印を付けた表を返す。"""
    raw, flow = predict_with_flow(conn, date)
    save_prediction(conn, raw, flow)
    return add_marks(raw)


def save_prediction(conn: sqlite3.Connection, table: pd.DataFrame, flow: pd.DataFrame | None = None,
                    now: dt.datetime | None = None) -> int:
    """出した予想を、回顧のために保存する(keibaai weekly と同じ。結果が出たレースと、当日朝以降の予想は書き換えない)。

    ただし予想を1件も保存していない直近の日は、結果が出たあとでも保存する(画面で予想したのに回顧に出ない、を防ぐ)。
    """
    from . import flow_review
    from .weekly import save_predictions

    now = now or dt.datetime.now()
    n = save_predictions(conn, table, now, late_ok=True)
    if n and flow is not None:
        saved = {r[0] for r in conn.execute("SELECT race_id FROM predictions WHERE saved_at = ?",
                                            (now.isoformat(timespec="seconds"),))}
        flow_review.save_forecast(conn, flow, saved, now)
    return n


def fetch_bet_odds(fetcher, race_id: str, on_type=None, conn: sqlite3.Connection | None = None) -> dict:
    """1レースの全券種のオッズを取得して {券種: {馬番の組: オッズ}} にする。

    JV-Link を呼べる PC で conn(出馬表の入った DB)を渡せば、JRA-VAN の速報オッズを使う。
    そうでなければ netkeiba に6回アクセスする(3〜5秒おき)。
    conn を渡したときは、取ったオッズを取得時刻つきで bet_odds_snapshots にも残す
    (全券種のオッズは後から取り直せないので、買う時点のオッズでの検証に使う)。
    保存に失敗しても、オッズの取得そのものは止めない。
    """
    from . import db, jvlink
    from .entries import BET_ODDS_TYPES, parse_bet_odds

    if conn is not None and jvlink.available():
        try:
            out = jvlink.fetch_bet_odds(conn, race_id)
        except jvlink.JVError as e:
            raise RuntimeError(str(e)) from e
    else:
        out = {}
        for i, t in enumerate(BET_ODDS_TYPES):
            if on_type:
                on_type(i, len(BET_ODDS_TYPES), t)
            out.update(parse_bet_odds(fetcher.odds_json(race_id, t), t))
    if conn is not None and out:
        try:
            db.save_bet_odds_snapshot(conn, race_id, out, dt.datetime.now().isoformat(timespec="seconds"))
        except Exception as e:  # 保存の失敗で買い方の表示を止めない
            print(f"bet odds snapshot {race_id}: {e}")
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


def train_start(first: dt.date, last: dt.date) -> dt.date:
    """学習に使う最初の日。モデルは TRAIN_FROM(model.py)より前のレースを学習に使わないので、取得済みの最初の日と遅いほう。
    TRAIN_FROM が最後の日より後なら、学習できる日がないので取得済みの最初の日を返す(実行時に足りないと伝える)。"""
    from .model import TRAIN_FROM

    start = max(first, dt.date.fromisoformat(TRAIN_FROM))
    return start if start < last else first


def eval_start_bounds(first: dt.date, last: dt.date) -> tuple[dt.date, dt.date]:
    """評価開始日の (選べる最も早い日, 初期値)。学習には評価開始日より前のレースが要るので、学習に使う最初の日
    (train_start)から1年(短い期間なら前半)は選ばせない。初期値は、最後の日の1年前(それが最も早い日より前なら、最も早い日)。"""
    start = train_start(first, last)
    earliest = start + min(dt.timedelta(days=365), (last - start) / 2)
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
