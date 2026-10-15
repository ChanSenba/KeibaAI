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


def unreviewed_dates(conn: sqlite3.Connection, today: dt.date, reviews_dir: Path) -> list[str]:
    """予想があり、今日より前で、回顧のファイルがまだない日。"""
    days = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM predictions WHERE date < ? ORDER BY date", (today.isoformat(),))]
    return [d for d in days if not (reviews_dir / f"{d}.md").exists()]


@dataclass
class Review:
    date: str
    horses: pd.DataFrame  # 予想した馬ごと。結果(finish)付き
    races: pd.DataFrame  # レースごとのまとめ
    summary: dict


def review_day(conn: sqlite3.Connection, date: str, stake: int = 100) -> Review:
    pred = pd.read_sql_query("SELECT * FROM predictions WHERE date = ?", conn, params=(date,))
    res = pd.read_sql_query(
        "SELECT race_id, number, finish, passing, last3f, time_sec, popularity AS final_pop, win_odds AS final_odds "
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
    return Review(date, h, races, summary)


def _roi(bets: pd.DataFrame, stake: int) -> float:
    return float(bets["win"].mul(bets["odds"].fillna(0)).sum() / len(bets) * 100) if len(bets) else float("nan")


def _pct(x: float) -> str:
    return "—" if x != x else f"{x:.0f}%"


def render_markdown(r: Review) -> str:
    s = r.summary
    out = [f"# {r.date} 回顧", "",
           f"- 予想したレース: {s['races']}",
           f"- ◎が1着: {s['honmei_wins']}レース" + (f"({s['honmei_wins'] / s['races']:.0%})" if s["races"] else "")
           + f" / ◎の単勝回収率 {_pct(s['honmei_roi'])}",
           f"- ◎○▲のうち3着以内に入った頭数(のべ): {s['top3_hits']}(最大 {3 * s['races']})",
           f"- 期待値≧1 の馬: {s['ev_bets']}頭を買って{s['ev_wins']}頭が勝ち、単勝回収率 {_pct(s['ev_roi'])}", "",
           "回収率は確定オッズで計算。予想は発走前に保存したものです。", ""]
    for _, x in r.races.sort_values(["venue", "round"]).iterrows():
        fin = "—" if x["honmei_finish"] != x["honmei_finish"] else f"{int(x['honmei_finish'])}着"
        line = f"- **{x['venue']}{int(x['round'])}R {x['race_name']}**: ◎{x['honmei']}は{fin}。"
        if pd.notna(x["winner_rank"]):
            line += (f"勝ち馬は{x['winner']}(予想{int(x['winner_rank'])}位・勝率{x['winner_p']:.0%}"
                     f"・単勝{x['winner_odds']:.1f}倍)。")
            if x["winner_rank"] >= 5:
                line += " ← 予想の下位から勝った。"
        out.append(line)
    return "\n".join(out) + "\n"


def run_weekly(conn: sqlite3.Connection, fetcher, today: dt.date, reviews_dir: str | Path,
               now: dt.datetime | None = None, jv: tuple[str, str] | None = None) -> dict:
    """毎週の処理。何度実行しても、済んだ分は飛ばす(木〜金と月曜の両方に同じ命令を登録できる)。

    jv=(書き出し先, 32bit Python) なら、成績と出馬表を JRA-VAN から取る(なければ netkeiba)。
    """
    from . import jvlink
    from .cli import collect_day

    now = now or dt.datetime.now()
    reviews_dir = Path(reviews_dir)
    reviews_dir.mkdir(parents=True, exist_ok=True)
    out = {"reviewed": [], "forecast": {}}
    for d in unreviewed_dates(conn, today, reviews_dir):
        if jv:
            day = dt.date.fromisoformat(d)
            jvlink.fetch_results(conn, day, day, jv[0], jv[1], today=today)
        else:
            collect_day(conn, fetcher, d.replace("-", ""))
        r = review_day(conn, d)
        if r.summary["races"]:  # 結果がまだなら、次回に回す
            (reviews_dir / f"{d}.md").write_text(render_markdown(r), encoding="utf-8")
            out["reviewed"].append(d)
    for d in _upcoming_weekend(today):
        out["forecast"][d.isoformat()] = forecast_day(conn, fetcher, d, now, jv)
    return out


def _upcoming_weekend(today: dt.date) -> list[dt.date]:
    """今日以後の直近の土曜・日曜。平日なら次の土日、土曜なら土日、日曜なら今日だけ。"""
    if today.weekday() == 6:
        return [today]
    sat = today + dt.timedelta(days=(5 - today.weekday()) % 7)
    return [sat, sat + dt.timedelta(days=1)]
