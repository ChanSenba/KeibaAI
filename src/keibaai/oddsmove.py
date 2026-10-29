"""単勝オッズの動き(JRA-VAN の時系列オッズ)から、締め切り前の人気の変化を特徴量にする(検証用)。

時系列オッズは odds_snapshots(source='ts')に、発表ごとの単勝オッズと人気が入っている(keibaai odds-history)。
発走時刻(races.start_time)の 60・30・10 分前に出ていた最後のオッズと、確定オッズ(成績の単勝オッズ)の比を取る。
負なら、そのあと人気が上がった(オッズが下がった)馬。発売開始直後の最初のオッズとの比も取る。
どれも、その時刻より前に出ていたオッズだけから作る。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

OFFSETS = (60, 30, 10)  # 発走の何分前の時点のオッズを見るか
MOVE_COLUMNS = ["om_open", "om_60", "om_30", "om_10", "om_late", "om_30_z", "pop_chg_30", "om_n_ann"]


def load_series(ts_db: str | Path, since: str) -> pd.DataFrame:
    """since(YYYY-MM-DD)以後のレースの時系列オッズ。列は race_id, number, t(発表時刻), odds, pop。読み取り専用。"""
    conn = sqlite3.connect(f"file:{Path(ts_db).resolve().as_posix()}?mode=ro", uri=True)
    try:
        s = pd.read_sql_query(
            "SELECT race_id, number, snapped_at AS t, win_odds AS odds, popularity AS pop FROM odds_snapshots "
            "WHERE source = 'ts' AND win_odds > 0 AND snapped_at >= ?", conn, params=(since,))
    finally:
        conn.close()
    s["t"] = pd.to_datetime(s["t"])
    return s.sort_values("t", kind="stable").reset_index(drop=True)


def _last_before(s: pd.DataFrame, cutoff: pd.Series, col: str) -> pd.Series:
    """レース・馬ごとに、cutoff(レースごとの時刻)以前で最後の col の値。"""
    sub = s[s["t"] <= s["race_id"].map(cutoff)]
    return sub.groupby(["race_id", "number"])[col].last()


def add_odds_move(df: pd.DataFrame, series: pd.DataFrame, start_times: pd.Series) -> pd.DataFrame:
    """df(race_id, number, date, win_odds を持つ)にオッズの動きの列を足す。

    start_times は race_id → 発走時刻('HH:MM')。時系列オッズのないレース・発走時刻のないレースは NaN。
    """
    df = df.copy()
    race_date = df.drop_duplicates("race_id").set_index("race_id")["date"]
    hhmm = start_times.reindex(race_date.index).astype("string")
    t_off = pd.to_datetime(race_date.dt.strftime("%Y-%m-%d").astype("string") + " " + hhmm, errors="coerce")
    s = series[series["race_id"].isin(t_off.dropna().index)]
    key = pd.MultiIndex.from_frame(df[["race_id", "number"]])
    final = pd.to_numeric(df["win_odds"], errors="coerce").where(lambda x: x > 0)
    out = {}
    for m in OFFSETS:
        o = _last_before(s, t_off - pd.Timedelta(minutes=m), "odds")
        out[m] = pd.Series(o.reindex(key).to_numpy(), index=df.index)
        df[f"om_{m}"] = np.log(out[m] / final)
    first = s.groupby(["race_id", "number"])["odds"].first()
    df["om_open"] = np.log(pd.Series(first.reindex(key).to_numpy(), index=df.index) / final)
    df["om_late"] = df["om_30"] - df["om_10"]  # 直前30分から10分前までの動き。正ならその間にオッズが下がった
    z = df.groupby("race_id")["om_30"]
    df["om_30_z"] = (df["om_30"] - z.transform("mean")) / z.transform("std").replace(0, np.nan)
    pop30 = _last_before(s, t_off - pd.Timedelta(minutes=30), "pop")
    pop_final = pd.to_numeric(df["popularity"], errors="coerce")
    df["pop_chg_30"] = pop_final - pd.Series(pop30.reindex(key).to_numpy(), index=df.index)  # 負なら人気が上がった
    n_ann = s.groupby(["race_id", "number"]).size()
    df["om_n_ann"] = pd.Series(n_ann.reindex(key).to_numpy(), index=df.index)
    return df


def move_variants() -> dict[str, tuple[list[str], list[str]]]:
    return {
        "base": ([], []),
        "30分前・10分前のオッズと確定の比": (["om_30", "om_10"], []),
        "60・30・10分前と最初のオッズの比": (["om_open", "om_60", "om_30", "om_10"], []),
        "レース内での30分前の動きの順位(z)": (["om_30_z"], []),
        "人気の変化(30分前→確定)": (["pop_chg_30"], []),
        "直前30分→10分前の動き": (["om_late"], []),
        "動きの候補すべて": (MOVE_COLUMNS, []),
    }
