"""結果テーブルから学習用の特徴量を作る。

過去の成績に基づく特徴量は、必ず「そのレースより前」の結果だけから計算する(未来の情報を使わない)。
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

CATEGORICAL = ["course_type", "venue", "ground", "sex"]
# 馬の過去成績・騎手/調教師の過去成績・レース条件。オッズは含めない(能力だけで予測するため)
BASE_FEATURES = [
    "number", "bracket", "weight_carried", "age", "distance", "field_size",
    "body_weight", "body_weight_diff", "days_since_last",
    "n_prev", "prev_finish", "avg_finish_3", "avg_finish_5", "win_rate", "top3_rate",
    "avg_last3f_3", "same_dist_top3_rate",
    "jockey_n", "jockey_win_rate", "jockey_top3_rate", "trainer_n", "trainer_top3_rate",
]
MARKET_FEATURES = ["win_odds", "popularity"]


def load_results(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT r.*, c.date, c.venue, c.round, c.course_type, c.distance, c.ground
        FROM results r JOIN races c USING (race_id)
        WHERE r.finish IS NOT NULL AND c.date IS NOT NULL
        """,
        conn,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def _expanding_rate(df: pd.DataFrame, key: str, flag: str) -> tuple[pd.Series, pd.Series]:
    """key ごとの『このレースより前』の出走数と flag 率。同じ日の他レースは含めない。"""
    day = df.groupby([key, "date"])[flag].agg(["sum", "count"]).reset_index().sort_values([key, "date"])
    day["cum_sum"] = day.groupby(key)["sum"].cumsum() - day["sum"]
    day["cum_n"] = day.groupby(key)["count"].cumsum() - day["count"]
    merged = df[[key, "date"]].merge(day[[key, "date", "cum_sum", "cum_n"]], on=[key, "date"], how="left")
    n = merged["cum_n"].to_numpy(dtype=float)
    rate = np.where(n > 0, merged["cum_sum"].to_numpy(dtype=float) / np.where(n > 0, n, 1), np.nan)
    return pd.Series(n, index=df.index), pd.Series(rate, index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "race_id", "number"]).reset_index(drop=True)
    df["win"] = (df["finish"] == 1).astype(int)
    df["top3"] = (df["finish"] <= 3).astype(int)
    df["sex"] = df["sex_age"].str[:1]
    df["age"] = pd.to_numeric(df["sex_age"].str[1:], errors="coerce")
    df["field_size"] = df.groupby("race_id")["number"].transform("count")
    df["dist_bucket"] = (df["distance"] // 400).astype("Int64")

    # --- 馬の過去成績(同じ馬の1つ前以前のレースだけ) ---
    g = df.groupby("horse_id", sort=False)
    df["n_prev"] = g.cumcount()
    df["prev_finish"] = g["finish"].shift(1)
    shifted = g["finish"].shift(1)
    df["avg_finish_3"] = shifted.groupby(df["horse_id"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["avg_finish_5"] = shifted.groupby(df["horse_id"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df["win_rate"] = g["win"].cumsum().sub(df["win"]) / df["n_prev"].replace(0, np.nan)
    df["top3_rate"] = g["top3"].cumsum().sub(df["top3"]) / df["n_prev"].replace(0, np.nan)
    l3 = g["last3f"].shift(1)
    df["avg_last3f_3"] = l3.groupby(df["horse_id"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["days_since_last"] = (df["date"] - g["date"].shift(1)).dt.days

    sd = df.groupby(["horse_id", "dist_bucket"], sort=False)
    n_sd = sd.cumcount()
    df["same_dist_top3_rate"] = sd["top3"].cumsum().sub(df["top3"]) / n_sd.replace(0, np.nan)

    # --- 騎手・調教師(前日までの成績) ---
    df["jockey_n"], df["jockey_win_rate"] = _expanding_rate(df, "jockey_id", "win")
    _, df["jockey_top3_rate"] = _expanding_rate(df, "jockey_id", "top3")
    df["trainer_n"], df["trainer_top3_rate"] = _expanding_rate(df, "trainer_id", "top3")

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df
