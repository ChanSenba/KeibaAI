"""毎週の流れ: 出走前に予想を残し、結果が出たら照合して回顧をまとめる。

予想は発走前の内容のまま predictions に保存し、結果が出たあとは書き換えない。
回顧は事実(予想と結果の差)だけをまとめる。理由の推測や講評は付けない。
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .gui import add_marks

PRED_COLS = ["race_id", "number", "saved_at", "date", "venue", "round", "race_name", "horse_name", "jockey",
             "win_odds", "popularity", "mark", "p_win_ability", "p_win_market", "ev_ability", "ev_market"]


FREEZE_HOUR = 9  # 開催日のこの時刻以降は、保存済みの予想を書き換えない(レースが始まってから変えないため)


def save_predictions(conn: sqlite3.Connection, table: pd.DataFrame, now: dt.datetime) -> int:
    """印を付けて保存する。結果が出たレースと、凍結した予想は保存しない。保存した頭数を返す。

    予想は開催日の前日までと、当日の朝(FREEZE_HOUR 時前)なら、取り直したオッズで更新できる。
    """
    t = add_marks(table)
    t["saved_at"] = now.isoformat(timespec="seconds")
    t["date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
    done = {r[0] for r in conn.execute("SELECT race_id FROM races")}
    if now.hour >= FREEZE_HOUR:
        today = now.date().isoformat()
        done |= {r[0] for r in conn.execute("SELECT race_id FROM predictions WHERE date <= ?", (today,))}
    done |= {r[0] for r in conn.execute("SELECT race_id FROM predictions WHERE date < ?", (now.date().isoformat(),))}
    t = t[~t["race_id"].isin(done)][PRED_COLS]
    t = t.astype(object).where(t.notna(), None)
    with conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO predictions ({','.join(PRED_COLS)}) VALUES ({','.join('?' * len(PRED_COLS))})",
            t.itertuples(index=False, name=None))
    return len(t)


def forecast_day(conn: sqlite3.Connection, fetcher, date: dt.date, now: dt.datetime | None = None,
                 jv: tuple[str, str] | None = None) -> int:
    """出馬表を取り直して予想を作り、保存する。出馬表がまだなら 0 を返す。jv=(書き出し先, 32bit Python) なら JRA-VAN から取る。"""
    from . import jvlink
    from .predict import fetch_entries, predict_day

    now = now or dt.datetime.now()
    n = (jvlink.fetch_entries(conn, date.isoformat(), True, jv[0], jv[1], now=now) if jv
         else fetch_entries(conn, fetcher, date.isoformat(), now=now))
    if n == 0:
        return 0
    return save_predictions(conn, predict_day(conn, date.isoformat()), now)


def unreviewed_dates(conn: sqlite3.Connection, today: dt.date, reviews_dir: Path,
                     include_today: bool = False) -> list[str]:
    """予想があり、今日より前(include_today なら今日も含む)で、回顧のファイルがまだない日。"""
    last = today + dt.timedelta(days=1) if include_today else today
    days = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM predictions WHERE date < ? ORDER BY date", (last.isoformat(),))]
    return [d for d in days if not (reviews_dir / f"{d}.md").exists()]


@dataclass
class Review:
    date: str
    horses: pd.DataFrame  # 予想した馬ごと。結果(finish)付き
    races: pd.DataFrame  # レースごとのまとめ
    summary: dict
    marks: pd.DataFrame | None = None  # 印ごとの成績(頭数・勝率・連対率・馬券内率・単勝回収率)


MARK_ORDER = ["◎", "○", "▲", "△", "☆"]


def mark_stats(h: pd.DataFrame) -> pd.DataFrame:
    """印ごとの成績。取消・除外(着順のない馬)は数えない。馬券内率は3着以内に入った割合。"""
    rows = []
    for m in MARK_ORDER:
        g = h[(h["mark"] == m) & h["finish"].notna()]
        if g.empty:
            continue
        n = len(g)
        rows.append({"印": m, "頭数": n, "1着": int((g["finish"] == 1).sum()),
                     "勝率": float((g["finish"] == 1).mean()), "連対率": float((g["finish"] <= 2).mean()),
                     "馬券内率": float((g["finish"] <= 3).mean()), "単勝回収率": _roi(g, 100) / 100})
    return pd.DataFrame(rows, columns=["印", "頭数", "1着", "勝率", "連対率", "馬券内率", "単勝回収率"])


def review_day(conn: sqlite3.Connection, date: str, stake: int = 100) -> Review:
    pred = pd.read_sql_query("SELECT * FROM predictions WHERE date = ?", conn, params=(date,))
    res = pd.read_sql_query(
        "SELECT race_id, number, finish, status, passing, last3f, time_sec, popularity AS final_pop, win_odds AS final_odds "
        "FROM results WHERE race_id IN (SELECT race_id FROM predictions WHERE date = ?)", conn, params=(date,))
    h = pred.merge(res, on=["race_id", "number"], how="left")
    h = h[h.groupby("race_id")["finish"].transform(lambda s: s.notna().any())].copy()  # 結果が出たレースだけ
    h["win"] = (h["finish"] == 1).astype(int)
    h["odds"] = h["final_odds"].fillna(h["win_odds"])  # 精算は確定オッズ。なければ予想時のもの
    h["rank"] = h.groupby("race_id")["p_win_ability"].rank(ascending=False, method="first")

    rows = []
    for rid, g in h.groupby("race_id", sort=False):
        top = g[g["mark"] == "◎"]
        top = top.iloc[0] if len(top) else g.sort_values("rank").iloc[0]
        win = g[g["win"] == 1]
        w = win.iloc[0] if len(win) else None
        top3 = set(g[g["finish"].between(1, 3)]["number"])
        marked = set(g[g["mark"].isin(["◎", "○", "▲"])]["number"])
        rows.append({
            "race_id": rid, "venue": g["venue"].iloc[0], "round": g["round"].iloc[0], "race_name": g["race_name"].iloc[0],
            "honmei": top["horse_name"], "honmei_finish": top["finish"], "honmei_win": int(top["win"]),
            "winner": None if w is None else w["horse_name"],
            "winner_rank": None if w is None else int(w["rank"]),
            "winner_p": None if w is None else w["p_win_ability"],
            "winner_odds": None if w is None else w["odds"],
            "top3_hit": len(top3 & marked),
        })
    races = pd.DataFrame(rows)
    ev = h[h["ev_ability"] >= 1]
    summary = {
        "races": len(races),
        "honmei_wins": int(races["honmei_win"].sum()) if len(races) else 0,
        "top3_hits": int(races["top3_hit"].sum()) if len(races) else 0,
        "honmei_roi": _roi(h[h["mark"] == "◎"], stake), "ev_bets": len(ev), "ev_wins": int(ev["win"].sum()),
        "ev_roi": _roi(ev, stake),
    }
    return Review(date, h, races, summary, mark_stats(h))


def _roi(bets: pd.DataFrame, stake: int) -> float:
    return float(bets["win"].mul(bets["odds"].fillna(0)).sum() / len(bets) * 100) if len(bets) else float("nan")


def _pct(x: float) -> str:
    return "—" if x != x else f"{x:.0f}%"


def finish_text(finish, status=None) -> str:
    """着順の表示。着順がなければ取消・除外などの理由(なければ —)。"""
    if finish == finish and finish is not None:
        return f"{int(finish)}着"
    return status if isinstance(status, str) and status else "—"


def marked_finishes(horses: pd.DataFrame, race_id: str) -> list[tuple[str, int, str, str]]:
    """そのレースの印のある馬の (印, 馬番, 馬名, 着順の表示)。印の順(◎○▲△☆)。"""
    g = horses[(horses["race_id"] == race_id) & horses["mark"].isin(MARK_ORDER)]
    g = g.assign(_o=g["mark"].map(MARK_ORDER.index)).sort_values(["_o", "number"])
    status = g["status"] if "status" in g else [None] * len(g)
    return [(m, int(n), name, finish_text(f, s))
            for m, n, name, f, s in zip(g["mark"], g["number"], g["horse_name"], g["finish"], status)]


def render_markdown(r: Review) -> str:
    s = r.summary
    out = [f"# {r.date} 回顧", "",
           f"- 予想したレース: {s['races']}",
           f"- ◎が1着: {s['honmei_wins']}レース" + (f"({s['honmei_wins'] / s['races']:.0%})" if s["races"] else "")
           + f" / ◎の単勝回収率 {_pct(s['honmei_roi'])}",
           f"- ◎○▲のうち3着以内に入った頭数(のべ): {s['top3_hits']}(最大 {3 * s['races']})",
           f"- 期待値≧1 の馬: {s['ev_bets']}頭を買って{s['ev_wins']}頭が勝ち、単勝回収率 {_pct(s['ev_roi'])}", "",
           "回収率は確定オッズで計算。予想は発走前に保存したものです。", ""]
    if r.marks is not None and not r.marks.empty:
        out += ["印ごとの成績:", ""] + [
            f"- {m['印']}: {m['頭数']}頭・勝率 {m['勝率']:.0%}・連対率 {m['連対率']:.0%}・馬券内率 {m['馬券内率']:.0%}"
            f"・単勝回収率 {_pct(m['単勝回収率'] * 100)}" for _, m in r.marks.iterrows()] + [""]
    for _, x in r.races.sort_values(["venue", "round"]).iterrows():
        fin = "—" if x["honmei_finish"] != x["honmei_finish"] else f"{int(x['honmei_finish'])}着"
        line = f"- **{x['venue']}{int(x['round'])}R {x['race_name']}**: ◎{x['honmei']}は{fin}。"
        if pd.notna(x["winner_rank"]):
            line += (f"勝ち馬は{x['winner']}(予想{int(x['winner_rank'])}位・勝率{x['winner_p']:.0%}"
                     f"・単勝{x['winner_odds']:.1f}倍)。")
            if x["winner_rank"] >= 5:
                line += " ← 予想の下位から勝った。"
        marked = marked_finishes(r.horses, x["race_id"])
        if marked:
            line += " 印の馬: " + " / ".join(f"{m}{name} {fin}" for m, _, name, fin in marked)
        out.append(line)
    return "\n".join(out) + "\n"


def run_weekly(conn: sqlite3.Connection, fetcher, today: dt.date, reviews_dir: str | Path,
               now: dt.datetime | None = None, jv: tuple[str, str] | None = None,
               after_races: bool = False, evals_dir: str | Path | None = None) -> dict:
    """毎週の処理。何度実行しても、済んだ分は飛ばす(木〜金と月曜の両方に同じ命令を登録できる)。

    jv=(書き出し先, 32bit Python) なら、成績と出馬表を JRA-VAN から取る(なければ netkeiba)。
    after_races=True は最終レース後の実行用(日曜の夜): 今日の分も回顧し、今週末の予想は作らず、
    evals_dir があれば学習・評価の結果も書く。
    """
    from . import jvlink
    from .cli import collect_day

    now = now or dt.datetime.now()
    reviews_dir = Path(reviews_dir)
    reviews_dir.mkdir(parents=True, exist_ok=True)
    out = {"reviewed": [], "forecast": {}, "evaluated": None}
    for d in unreviewed_dates(conn, today, reviews_dir, include_today=after_races):
        if jv:
            day = dt.date.fromisoformat(d)
            jvlink.fetch_results(conn, day, day, jv[0], jv[1], today=today)
        else:
            collect_day(conn, fetcher, d.replace("-", ""))
        r = review_day(conn, d)
        if r.summary["races"]:  # 結果がまだなら、次回に回す
            (reviews_dir / f"{d}.md").write_text(render_markdown(r), encoding="utf-8")
            out["reviewed"].append(d)
    if after_races:
        if evals_dir is not None:
            out["evaluated"] = str(write_evaluation(conn, today, evals_dir))
        return out
    for d in _upcoming_weekend(today):
        out["forecast"][d.isoformat()] = forecast_day(conn, fetcher, d, now, jv)
    return out


EVAL_YEARS = 1  # 評価期間: 今日からさかのぼる年数(それ以前で学習する)


def live_trend(conn: sqlite3.Connection) -> pd.DataFrame:
    """回顧した日ごとの、実運用の成績(発走前に保存した予想と確定結果の照合)。"""
    rows = []
    for (d,) in conn.execute("SELECT DISTINCT date FROM predictions ORDER BY date").fetchall():
        s = review_day(conn, d).summary
        if s["races"]:
            rows.append({"日付": d, "レース数": s["races"], "◎1着": s["honmei_wins"],
                         "◎単勝回収率(%)": s["honmei_roi"], "期待値≧1の頭数": s["ev_bets"],
                         "期待値≧1の回収率(%)": s["ev_roi"]})
    return pd.DataFrame(rows)


def write_evaluation(conn: sqlite3.Connection, today: dt.date, evals_dir: str | Path) -> Path:
    """学習・評価(直近 EVAL_YEARS 年を評価期間にした keibaai train と同じ3行)と、実運用の推移を書く。"""
    from .features import build_features, load_results
    from .model import baseline_market, evaluate, format_report

    evals_dir = Path(evals_dir)
    evals_dir.mkdir(parents=True, exist_ok=True)
    test_from = (today - dt.timedelta(days=365 * EVAL_YEARS)).isoformat()
    df = build_features(load_results(conn))
    r_abil, test = evaluate(df, test_from, use_market=False)
    r_mkt, _ = evaluate(df, test_from, use_market=True)
    trend = live_trend(conn)
    out = [f"# {today.isoformat()} 学習・評価", "", f"評価期間: {test_from} 以降(それ以前で学習)", "",
           "```", format_report([baseline_market(test), r_abil, r_mkt]), "```", "",
           "## 実運用の推移(発走前に保存した予想と確定結果)", ""]
    if trend.empty:
        out.append("回顧できた日はまだありません。")
    else:
        out.append(trend.to_markdown(index=False, floatfmt=".0f") if _has_tabulate() else trend.to_string(index=False))
    path = evals_dir / f"{today.isoformat()}.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


def _upcoming_weekend(today: dt.date) -> list[dt.date]:
    """今日以後の直近の土曜・日曜。平日なら次の土日、土曜なら土日、日曜なら今日だけ。"""
    if today.weekday() == 6:
        return [today]
    sat = today + dt.timedelta(days=(5 - today.weekday()) % 7)
    return [sat, sat + dt.timedelta(days=1)]
