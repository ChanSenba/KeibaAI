"""★が付かない理由の切り分け(保存済みの予想だけを読む。DB は読み取り専用で開く)。

使い方:
    python -m keibaai.star_diag --db data/keiba.db

★(star)は、オッズ込みの勝率 p_win_market がないレースには付かない。p_win_market は、
そのレースの全馬に単勝オッズがあるときだけ出る値なので、予想を保存した時点でオッズが
そろっていなければ、そのレースには最初から付かない。逆に、オッズがそろったレースでも
ev_market(オッズ込みの期待値)の最大値が基準(1.1)に届く馬がいなければ、モデルが市場に
寄っていて基準を満たす馬が出ていないことになる。

このレポートは、保存時刻の帯ごとの p_win_market の NaN 率(オッズがなかったレースの割合)と、
オッズがそろったレースの ev_market 最大値の分布を並べ、どちらの問題かを切り分ける材料を出す。
出力は Markdown。
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
from pathlib import Path

import pandas as pd

from .gui import STAR_EV, STAR_ODDS, STAR_P

BANDS = ["前日以前", "当日・9時前", "当日・9時以降", "翌日以降", "不明"]
FREEZE_HOUR = 9  # weekly.FREEZE_HOUR と同じ(開催日のこの時刻以降、保存済みの予想は書き換えない)


def save_band(saved_at, date) -> str:
    """予想を保存した時刻の帯。レースの日付との差で分け、当日なら FREEZE_HOUR の前後で分ける。"""
    try:
        s = dt.datetime.fromisoformat(str(saved_at))
        d = dt.date.fromisoformat(str(date)[:10])
    except (TypeError, ValueError):
        return "不明"
    off = (s.date() - d).days
    if off < 0:
        return "前日以前"
    if off > 0:
        return "翌日以降"
    return "当日・9時前" if s.hour < FREEZE_HOUR else "当日・9時以降"


def load_predictions(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT race_id, date, saved_at, number, win_odds, p_win_market, ev_market FROM predictions", conn)
    for c in ("win_odds", "p_win_market", "ev_market"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["band"] = [save_band(s, d) for s, d in zip(df["saved_at"], df["date"])]
    return df


def band_stats(df: pd.DataFrame) -> pd.DataFrame:
    """保存時刻の帯ごと: 頭数・レース数と、p_win_market が全頭 NaN のレース(オッズなし)の数と割合。"""
    per = (df.groupby(["band", "race_id"])["p_win_market"].apply(lambda s: bool(s.isna().all()))
             .rename("no_odds").reset_index())
    rows = []
    for band in BANDS:
        g = df[df["band"] == band]
        if g.empty:
            continue
        p = per[per["band"] == band]
        rows.append({"帯": band, "頭数": len(g), "レース数": len(p),
                     "オッズなしレース": int(p["no_odds"].sum()), "オッズなし率": float(p["no_odds"].mean())})
    return pd.DataFrame(rows, columns=["帯", "頭数", "レース数", "オッズなしレース", "オッズなし率"])


def full_odds_races(df: pd.DataFrame) -> pd.DataFrame:
    """p_win_market が全頭そろったレースだけ(★が付きうるレース)。1頭でも欠けているレースは除く。"""
    lacking = df.loc[df["p_win_market"].isna(), "race_id"].unique()
    return df[~df["race_id"].isin(lacking)]


def criteria_stats(df: pd.DataFrame) -> dict:
    """オッズがそろったレースでの、レースごとの ev_market 最大値の分布と、現行の★の基準を満たす割合。"""
    full = full_odds_races(df)
    out = {"races": int(full["race_id"].nunique()), "horses": len(full),
           "quantiles": {}, "hit_races": 0, "hit_race_rate": 0.0, "hit_horses": 0, "hit_horse_rate": 0.0}
    if full.empty:
        return out
    max_ev = full.groupby("race_id")["ev_market"].max()
    out["quantiles"] = {q: float(max_ev.quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)}
    hit = (full["ev_market"] >= STAR_EV) & (full["p_win_market"] >= STAR_P) & (full["win_odds"] <= STAR_ODDS)
    out["hit_horses"] = int(hit.sum())
    out["hit_horse_rate"] = float(hit.mean())
    out["hit_races"] = int(full.loc[hit, "race_id"].nunique())
    out["hit_race_rate"] = out["hit_races"] / out["races"]
    return out


def markdown_report(df: pd.DataFrame) -> str:
    bands = band_stats(df)
    crit = criteria_stats(df)
    out = ["# ★の切り分け診断", "",
           f"対象: 保存済みの予想 {len(df):,} 頭 / {df['race_id'].nunique():,} レース"
           f"({df['date'].min()} 〜 {df['date'].max()})", "",
           "## 読み方", "",
           "- 保存時刻の帯ごとに「オッズなし率」が高い帯がある → その帯の保存では、オッズがそろう前に"
           "予想を保存している(保存タイミングの問題)。★はオッズ込みの勝率がないと付かないので、"
           "その帯のレースには最初から付いていない",
           "- オッズがそろったレースでも、レースごとの ev_market 最大値が基準(1.1)に届くレースが少ない → "
           "モデルが市場に寄っていて、基準を満たす馬が出ていない(モデル・基準側の問題)",
           "- 両方ありうる。まず帯ごとのオッズなし率、次に最大値の分布の順に見る", "",
           "## 保存時刻の帯ごとのオッズのそろい方", "",
           "| 帯 | 頭数 | レース数 | オッズなしレース | オッズなし率 |", "|---|---|---|---|---|"]
    out += [f"| {x['帯']} | {x['頭数']:,} | {x['レース数']:,} | {x['オッズなしレース']:,} | {x['オッズなし率']:.0%} |"
            for _, x in bands.iterrows()]
    out += ["", "## オッズがそろったレースでの ev_market 最大値", "",
            f"対象: {crit['races']:,} レース / {crit['horses']:,} 頭", ""]
    if crit["quantiles"]:
        labels = [(0.0, "最小"), (0.25, "25%"), (0.5, "中央"), (0.75, "75%"), (0.9, "90%"), (0.95, "95%"), (1.0, "最大")]
        out += ["| 分位点 | 最大 ev_market |", "|---|---|"]
        out += [f"| {name} | {crit['quantiles'][q]:.2f} |" for q, name in labels]
        out += ["", f"現行の基準(期待値 {STAR_EV} 以上・勝率 {STAR_P:.0%} 以上・単勝 {STAR_ODDS:.0f} 倍以内)を満たす馬: "
                f"{crit['hit_horses']:,} 頭({crit['hit_horse_rate']:.1%})、"
                f"いるレース: {crit['hit_races']:,} レース({crit['hit_race_rate']:.1%})"]
    else:
        out += ["オッズがそろったレースがありません(全レースで p_win_market が欠けています)。"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="★が付かない理由の切り分け(保存済みの予想だけを読む。DB は読み取り専用)")
    ap.add_argument("--db", default="data/keiba.db")
    args = ap.parse_args(argv)
    conn = sqlite3.connect(f"file:{Path(args.db).as_posix()}?mode=ro", uri=True)
    df = load_predictions(conn)
    conn.close()
    if df.empty:
        raise SystemExit("保存済みの予想がありません")
    print(markdown_report(df))


if __name__ == "__main__":
    main()
