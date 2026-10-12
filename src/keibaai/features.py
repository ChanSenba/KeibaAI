"""結果テーブルから学習用の特徴量を作る。

過去の成績に基づく特徴量は、必ず「そのレースより前」の結果だけから計算する(未来の情報を使わない)。
"""
from __future__ import annotations

import re
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
    "race_class", "class_change", "distance_change", "early_pos_rel_3", "jockey_changed",
    "jockey_n", "jockey_win_rate", "jockey_top3_rate", "trainer_n", "trainer_top3_rate",
]
MARKET_FEATURES = ["win_odds", "popularity"]
BIAS_FEATURES = ["track_front_bias", "track_inner_bias"]  # トラックバイアス(下の _track_bias)
BIAS_WINDOW = "10D"


def load_results(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT r.*, c.name AS race_name, c.date, c.venue, c.round, c.course_type, c.distance, c.ground
        FROM results r JOIN races c USING (race_id)
        WHERE r.finish IS NOT NULL AND c.date IS NOT NULL
        """,
        conn,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


_CLASS_PATTERNS = [  # 上から順に判定。値が大きいほど格上
    (r"新馬", 0), (r"未勝利", 1), (r"1勝|500万", 2), (r"2勝|1000万", 3), (r"3勝|1600万", 4),
    (r"\(GIII\)|\(G3\)", 6), (r"\(GII\)|\(G2\)", 7), (r"\(GI\)|\(G1\)", 8),
    (r"\(L\)|リステッド|オープン|OP|\(OP\)", 5),
]


def parse_class(name: str | None) -> float:
    """レース名から格付けの目安(0=新馬 … 8=G1)を返す。分からなければ NaN。"""
    if not isinstance(name, str):
        return float("nan")
    for pat, rank in _CLASS_PATTERNS:
        if re.search(pat, name):
            return float(rank)
    return float("nan")


def parse_early_position(passing: str | None) -> float:
    """通過順位('13-12-12-7')の最初の数字。取れなければ NaN。"""
    m = re.match(r"\s*(\d+)", passing if isinstance(passing, str) else "")  # DB の NULL は NaN(float) で来る
    return float(m.group(1)) if m else float("nan")


def _expanding_rate(df: pd.DataFrame, key: str, flag: str) -> tuple[pd.Series, pd.Series]:
    """key ごとの『このレースより前』の出走数と flag 率。同じ日の他レースは含めない。"""
    day = df.groupby([key, "date"])[flag].agg(["sum", "count"]).reset_index().sort_values([key, "date"])
    day["cum_sum"] = day.groupby(key)["sum"].cumsum() - day["sum"]
    day["cum_n"] = day.groupby(key)["count"].cumsum() - day["count"]
    merged = df[[key, "date"]].merge(day[[key, "date", "cum_sum", "cum_n"]], on=[key, "date"], how="left")
    n = merged["cum_n"].to_numpy(dtype=float)
    rate = np.where(n > 0, merged["cum_sum"].to_numpy(dtype=float) / np.where(n > 0, n, 1), np.nan)
    return pd.Series(n, index=df.index), pd.Series(rate, index=df.index)


def _track_bias(df: pd.DataFrame) -> pd.DataFrame:
    """同じ競馬場・同じ芝/ダートの、直近(BIAS_WINDOW 以内)の前の開催日に出ていた偏りを返す。

    front: 前に付けた馬(序盤の位置が前の3分の1)の3着以内率 − そのほかの馬の率。大きいほど先行有利。
    inner: 内の馬(馬番がレース内で前の3分の1)の3着以内率 − 外の馬(後ろの3分の1)の率。大きいほど内有利。
    当日のレースは含めない(出走前には分からないため)。値は ±1 に収まり、差がない所は 0 付近。
    """
    done = df[df["finish"].notna() & df["venue"].notna()].copy()
    done["_top3"] = (done["finish"] <= 3).astype(float)
    pos = done["number"] / done["field_size"]
    done["_front"] = np.where(done["early_pos"].isna(), np.nan, (done["early_pos"] <= 1 / 3).astype(float))
    done["_inner"] = np.where(pos <= 1 / 3, 1.0, np.where(pos >= 2 / 3, 0.0, np.nan))

    def diff(g: pd.DataFrame, col: str) -> float:
        a, b = g.loc[g[col] == 1, "_top3"], g.loc[g[col] == 0, "_top3"]
        return a.mean() - b.mean() if len(a) and len(b) else np.nan

    keys = ["venue", "course_type", "date"]
    day = done.groupby(keys, observed=True).apply(
        lambda g: pd.Series({"track_front_bias": diff(g, "_front"), "track_inner_bias": diff(g, "_inner")}),
        include_groups=False).reset_index()
    out = []
    for _, g in day.sort_values("date").groupby(["venue", "course_type"], observed=True):
        r = g.set_index("date")[BIAS_FEATURES].rolling(BIAS_WINDOW, closed="left", min_periods=1).mean()
        out.append(r.reset_index().assign(venue=g["venue"].iloc[0], course_type=g["course_type"].iloc[0]))
    if not out:
        return pd.DataFrame(columns=keys + BIAS_FEATURES)
    return pd.concat(out, ignore_index=True)


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

    # --- クラス・距離・脚質・乗り替わり(いずれも前走との比較は shift で過去だけを使う) ---
    df["race_class"] = df["race_name"].map(parse_class)
    df["class_change"] = df["race_class"] - g["race_class"].shift(1)
    df["distance_change"] = df["distance"] - g["distance"].shift(1)
    df["early_pos"] = df["passing"].map(parse_early_position) / df["field_size"]
    ep = g["early_pos"].shift(1)
    df["early_pos_rel_3"] = ep.groupby(df["horse_id"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    prev_jockey = g["jockey_id"].shift(1)
    df["jockey_changed"] = np.where(prev_jockey.isna(), np.nan, (prev_jockey != df["jockey_id"]).astype(float))

    bias = _track_bias(df)
    df = df.merge(bias, on=["venue", "course_type", "date"], how="left")
    for c in BIAS_FEATURES:
        if c not in df:
            df[c] = np.nan

    # --- 騎手・調教師(前日までの成績) ---
    df["jockey_n"], df["jockey_win_rate"] = _expanding_rate(df, "jockey_id", "win")
    _, df["jockey_top3_rate"] = _expanding_rate(df, "jockey_id", "top3")
    df["trainer_n"], df["trainer_top3_rate"] = _expanding_rate(df, "trainer_id", "top3")

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df
