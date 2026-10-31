"""結果テーブルから学習用の特徴量を作る。

過去の成績に基づく特徴量は、必ず「そのレースより前」の結果だけから計算する(未来の情報を使わない)。
"""
from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd

from .jvrace import INNER_OUTER_VENUES
from .jvtraining import JV_FEATURES, add_jv_features

CATEGORICAL = ["course_type", "venue", "ground", "sex", "train_critic"]
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
# 記事(予想ファクター)に沿った特徴量。すべて「そのレースより前」の情報だけで作る
APTITUDE_FEATURES = ["horse_ct_top3_rate", "horse_venue_top3_rate", "horse_dir_top3_rate", "horse_ground_top3_rate",
                     "prev_same_ct_finish",
                     # 組み合わせの条件: 競馬場×芝ダ×距離帯(同コースではない。距離は400m刻み)、芝ダ×馬場状態、同じレース名。件数も付ける
                     "horse_course_top3_rate", "horse_course_n", "horse_ctground_top3_rate",
                     "horse_race_top3_rate", "horse_race_n"]
FORM_FEATURES = ["l3f_rank_3", "time_diff_5", "pos_gain_3",
                 # 直近1〜2走の中身(平均ではなく前走そのもの)
                 "prev_l3f_rank", "prev_early_pos", "prev_pos_gain", "prev_class", "prev_field_size",
                 "prev_l3f_vs_lap", "prev2_finish", "prev2_time_diff", "bw_vs_avg", "weight_ratio", "weight_change",
                 "prev_pop_gap", "dist_shorten300", "dist_extend300"]
ROTATION_FEATURES = ["weeks_since_last", "is_layoff", "runs_since_layoff", "is_renso",
                     # 前走の内容(タイム差・前半のペース・距離)と、今回までの間隔の組み合わせ
                     "prev_time_diff", "prev_pace_z", "prev_distance", "hard_prev", "load_x_short", "load_per_day",
                     "hard_prev_short", "dist_x_short"]
HUMAN_EXTRA = ["jockey_ct_top3_rate", "jockey_dist_top3_rate", "jockey_venue_top3_rate", "jockey_roi",
               "pair_n", "pair_top3_rate", "tj_n", "tj_top3_rate",
               "jockey_course_top3_rate", "jockey_course_n", "jockey_ctground_top3_rate", "jockey_class_top3_rate"]
PACE_FEATURES = ["style_front_rate", "style_closer_rate", "n_front_runners", "front_share", "field_style_mean",
                 "style_vs_field", "inner_pos"]
# ラップ: 過去のレースのペース(前半3F − 後半3F。正なら後傾=スロー)と、自分の上がりが場のラップよりどれだけ速かったか
LAP_FEATURES = ["prev_pace_diff", "pace_diff_3", "l3f_vs_lap_3"]
# 同コース(同じ競馬場の、芝ダ・内回り外回り・距離が同じコース)の直近5開催日の展開: ペース差(前半3F−後半3F。正なら後傾=スロー)の平均と、
# 前に付けた馬(序盤の位置が前の3分の1)の3着内率 − そのほかの馬の率の平均。そのレースの顔ぶれの脚質との組み合わせで効く
COURSE_FLOW_FEATURES = ["course_pace_diff_5", "course_front_edge_5", "front_x_course_edge"]
COURSE_FLOW_DAYS = 5
COURSE_KEYS = ["venue", "course_type", "inout", "distance"]  # 同コースの条件(inout は inner_outer の値)


def inner_outer(venue, course_type, course) -> str:
    """同コースを見分けるための内回り・外回りの別(外 / 内 / 外-内 など)。別のないコースは空文字。

    netkeiba は内回りを書かない(外回りだけ「外」)が、JRA-VAN は「内」と書くので、内外の別がある競馬場の芝で
    空のものは内回りとみなしてそろえる。
    """
    if isinstance(course, str) and course:
        return course
    return "内" if course_type == "芝" and venue in INNER_OUTER_VENUES else ""


def same_course(a, b) -> bool:
    """同コースか。同じ競馬場で、芝ダ・内回り外回り・距離が同じ(回りは競馬場と芝ダで決まる)。距離だけ同じなら違う。"""
    if not isinstance(a.get("venue"), str) or a.get("venue") != b.get("venue") or a.get("course_type") != b.get("course_type"):
        return False
    da, db_ = pd.to_numeric(a.get("distance"), errors="coerce"), pd.to_numeric(b.get("distance"), errors="coerce")
    if da != da or da != db_:
        return False
    return inner_outer(a.get("venue"), a.get("course_type"), a.get("course")) == \
        inner_outer(b.get("venue"), b.get("course_type"), b.get("course"))
# 血統: 父・母の父の産駒が、同じ条件でこれまでどれだけ3着以内に来たか(結果は「そのレースより前」だけ)
PEDIGREE_FEATURES = ["sire_n", "sire_top3_rate", "sire_ctdist_top3_rate", "sire_ground_top3_rate",
                     "damsire_n", "damsire_top3_rate", "damsire_ctdist_top3_rate"]
# 調教: 最終追い切りの評価(netkeiba の短評とA〜D)。レース前日に公開されるので、当日の予想にも使える
TRAINING_FEATURES = ["train_rank_score", "train_rank_vs_field", "train_critic"] + JV_FEATURES
TRAIN_RANK_SCORE = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "E": 0.0}
ARTICLE_FEATURES = (APTITUDE_FEATURES + FORM_FEATURES + ROTATION_FEATURES + HUMAN_EXTRA + PACE_FEATURES
                    + LAP_FEATURES + COURSE_FLOW_FEATURES + PEDIGREE_FEATURES + TRAINING_FEATURES)
BIAS_FEATURES = ["track_front_bias", "track_inner_bias"]  # トラックバイアス(下の _track_bias)
# レースの条件。要素別のモデルには入っているが、総合には入っていなかった
CONDITION_FEATURES = ["course_type", "ground", "venue", "sex"]
# 予想に使う組(train・predict・backtest・bets)。全要素を使う。
# 2026年の2,730レースで: 能力のみ AUC 0.794(総合の27個)→ 0.803、オッズあり LogLoss 0.2050 → 0.2044(オッズだけは 0.2047)。
# 条件の列は能力のみのモデルでは少し効き(AUC +0.0006〜0.0013)、オッズありでは効かなかったので、能力のみにだけ足す
_ALL_FEATURES = list(dict.fromkeys(BASE_FEATURES + BIAS_FEATURES + ARTICLE_FEATURES))
# 厩舎×騎手の3着内率は、能力のみのモデルでは回数の少なさを割り引いた版(tj_top3_shr)に置き換える。
# 評価 2025年から・2026年から(乱数3通り)で AUC 0.8055 → 0.8063、0.8044 → 0.8053。LogLoss の差の95%区間は
# 両方とも0をまたがなかった(k=5・10・20 はまたいだ)。オッズありのモデルでは良くならなかったので、元の率のまま
TJ_SHRINK_K = 50
ABILITY_MODEL_FEATURES = [("tj_top3_shr" if c == "tj_top3_rate" else c) for c in _ALL_FEATURES] + CONDITION_FEATURES
MARKET_MODEL_FEATURES = _ALL_FEATURES + MARKET_FEATURES
# 走破タイムの指数(add_speed_figures)。前走・直近3走の最大と平均、最大のレース内の平均との差
FIGURE_FEATURES = ["prev_fig", "best_fig_3", "avg_fig_3", "best_fig_vs_field"]
# レース内の相対値(add_relative)。木のモデルは行どうしを比べられないので、同じレースの他馬との比較を列にする
RELATIVE_BASE = ["avg_finish_3", "top3_rate", "win_rate", "jockey_top3_rate", "trainer_top3_rate", "avg_last3f_3",
                 "l3f_rank_3", "time_diff_5", "train_rank_score", "horse_course_top3_rate", "prev_finish",
                 "weight_carried", "early_pos_rel_3", "prev_time_diff", "sire_ctdist_top3_rate", "best_fig_3", "avg_fig_3"]
RELATIVE_FEATURES = [f"{c}_{k}" for c in RELATIVE_BASE for k in ("z", "rk")]
# オッズ込みのモデル(model.fit_residual)。オッズは列ではなく、市場の勝率を出発点(init_score)にして、
# モデルは市場の見立てからのずれだけを学ぶ。2025-01〜2026-10 の 6,153 レース(2024年までで学習)で、
# 勝ち馬の対数尤度は市場だけより 1レースあたり +0.0060(95%区間 +0.0034〜+0.0085)、いままでのオッズ込みモデルより +0.0030(+0.0009〜+0.0050)
RESIDUAL_MODEL_FEATURES = ABILITY_MODEL_FEATURES + FIGURE_FEATURES + RELATIVE_FEATURES
BIAS_WINDOW = "10D"


def load_results(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT r.*, c.name AS race_name, c.date, c.venue, c.round, c.course_type, c.direction, c.course, c.distance,
               c.ground, c.pace_first3f, c.pace_last3f,
               p.sire_id, p.damsire_id, t.rank AS train_rank, t.critic AS train_critic
        FROM results r JOIN races c USING (race_id) LEFT JOIN pedigree p USING (horse_id)
        LEFT JOIN training t ON t.race_id = r.race_id AND t.number = r.number AND t.seq = 0
        WHERE (r.finish IS NOT NULL OR r.status IN ('中', '失')
               OR r.status GLOB '[0-9]*(降)') AND c.date IS NOT NULL
        """,
        conn,
    )
    # 旧netkeiba取込の「3(降)」等は確定順位を持つ。DBを書き換えず読み込み時に復元。
    demoted = pd.to_numeric(df["status"].astype("string").str.extract(r"^(\d+)\(降\)$", expand=False), errors="coerce")
    df["finish"] = pd.to_numeric(df["finish"].fillna(demoted), errors="coerce").astype(float)
    df = df[result_mask(df)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["is_entry"] = False
    return add_jv_features(conn, df)


def result_mask(df: pd.DataFrame) -> pd.Series:
    """確定した出走。着順のない中止・失格も含め、出馬表とは区別する。"""
    status = df.get("status", pd.Series(None, index=df.index, dtype=object))
    known = df["finish"].notna() | status.isin(["中", "失"])
    if "is_entry" in df:
        known &= ~df["is_entry"].fillna(False).astype(bool)
    return known


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


def parse_late_position(passing: str | None) -> float:
    """通過順位('13-12-12-7')の最後の数字(最後のコーナー)。取れなければ NaN。"""
    m = re.findall(r"\d+", passing if isinstance(passing, str) else "")
    return float(m[-1]) if m else float("nan")


def parse_early_position(passing: str | None) -> float:
    """通過順位('13-12-12-7')の最初の数字。取れなければ NaN。"""
    m = re.match(r"\s*(\d+)", passing if isinstance(passing, str) else "")  # DB の NULL は NaN(float) で来る
    return float(m.group(1)) if m else float("nan")


def _expanding_rate(df: pd.DataFrame, key: str | list[str], flag: str) -> tuple[pd.Series, pd.Series]:
    """key ごとの『このレースより前』の出走数と flag 率。同じ日の他レースは含めない。

    key は列名のリストでもよい(騎手×コース など)。結果のない行(出馬表)は数えない。
    """
    keys = [key] if isinstance(key, str) else list(key)
    work = df[keys + ["date"]].copy()
    work["_f"] = df[flag].astype(float).where(result_mask(df))
    day = work.groupby(keys + ["date"], observed=True)["_f"].agg(["sum", "count"]).reset_index().sort_values(keys + ["date"])
    day["cum_sum"] = day.groupby(keys, observed=True)["sum"].cumsum() - day["sum"]
    day["cum_n"] = day.groupby(keys, observed=True)["count"].cumsum() - day["count"]
    merged = df[keys + ["date"]].merge(day[keys + ["date", "cum_sum", "cum_n"]], on=keys + ["date"], how="left")
    n = merged["cum_n"].to_numpy(dtype=float)
    rate = np.where(n > 0, merged["cum_sum"].to_numpy(dtype=float) / np.where(n > 0, n, 1), np.nan)
    return pd.Series(n, index=df.index), pd.Series(rate, index=df.index)


def _track_bias(df: pd.DataFrame) -> pd.DataFrame:
    """同じ競馬場・同じ芝/ダートの、直近(BIAS_WINDOW 以内)の前の開催日に出ていた偏りを返す。

    front: 前に付けた馬(序盤の位置が前の3分の1)の3着以内率 − そのほかの馬の率。大きいほど先行有利。
    inner: 内の馬(馬番がレース内で前の3分の1)の3着以内率 − 外の馬(後ろの3分の1)の率。大きいほど内有利。
    当日のレースは含めない(出走前には分からないため)。値は ±1 に収まり、差がない所は 0 付近。
    """
    done = df[result_mask(df) & df["venue"].notna()].copy()
    done["_top3"] = (done["finish"] <= 3).astype(float)
    pos = done["number"] / done["field_size"]
    done["_front"] = np.where(done["early_pos"].isna(), np.nan, (done["early_pos"] <= 1 / 3).astype(float))
    done["_inner"] = np.where(pos <= 1 / 3, 1.0, np.where(pos >= 2 / 3, 0.0, np.nan))

    def diff(g: pd.DataFrame, col: str) -> float:
        a, b = g.loc[g[col] == 1, "_top3"], g.loc[g[col] == 0, "_top3"]
        return a.mean() - b.mean() if len(a) and len(b) else np.nan

    keys = ["venue", "course_type", "date"]
    if done.empty:
        return df[keys].drop_duplicates().assign(**{c: np.nan for c in BIAS_FEATURES})
    day = done.groupby(keys, observed=True).apply(
        lambda g: pd.Series({"track_front_bias": diff(g, "_front"), "track_inner_bias": diff(g, "_inner")}),
        include_groups=False).reset_index()
    # まだ結果のない予測日も窓の評価対象にする。当日値は closed="left" で使わない。
    day = df[keys].drop_duplicates().merge(day, on=keys, how="left")
    out = []
    for _, g in day.sort_values("date").groupby(["venue", "course_type"], observed=True):
        r = g.set_index("date")[BIAS_FEATURES].rolling(BIAS_WINDOW, closed="left", min_periods=1).mean()
        out.append(r.reset_index().assign(venue=g["venue"].iloc[0], course_type=g["course_type"].iloc[0]))
    if not out:
        return pd.DataFrame(columns=keys + BIAS_FEATURES)
    return pd.concat(out, ignore_index=True)


def _course_flow(df: pd.DataFrame) -> pd.DataFrame:
    """同コースの直近 COURSE_FLOW_DAYS 開催日の展開(ペース差・先行有利度)を、各行に付ける。当日のレースは含めない。"""
    keys = COURSE_KEYS
    if "course" not in df:
        df["course"] = None
    df["inout"] = [inner_outer(v, c, x) for v, c, x in zip(df["venue"], df["course_type"], df["course"])]
    done = df[df["finish"].notna() & df["early_pos"].notna() & df["venue"].notna()].copy()
    done["_front"] = (done["early_pos"] <= 1 / 3)
    done["_top3"] = (done["finish"] <= 3).astype(float)
    race = done.groupby(["race_id", *keys, "date"], observed=True).apply(
        lambda g: pd.Series({
            "pace_diff": pd.to_numeric(g["pace_first3f"], errors="coerce").iloc[0]
            - pd.to_numeric(g["pace_last3f"], errors="coerce").iloc[0],
            "front_edge": (g.loc[g["_front"], "_top3"].mean() - g.loc[~g["_front"], "_top3"].mean())
            if g["_front"].any() and (~g["_front"]).any() else np.nan}),
        include_groups=False).reset_index()
    cols = ["course_pace_diff_5", "course_front_edge_5"]
    if race.empty:
        for c in cols:
            df[c] = np.nan
    else:
        day = race.groupby([*keys, "date"], observed=True)[["pace_diff", "front_edge"]].mean().reset_index()
        day = day.sort_values([*keys, "date"])
        grp = day.groupby(keys, observed=True)
        for src, dst in (("pace_diff", cols[0]), ("front_edge", cols[1])):
            day[dst] = grp[src].transform(lambda x: x.rolling(COURSE_FLOW_DAYS, min_periods=1).mean())
        # 各行の日より前(当日は含めない)で、いちばん新しい開催日の「直近5開催日の平均」を取る
        left = df[[*keys, "date"]].assign(_i=np.arange(len(df))).dropna(subset=keys).sort_values("date")
        got = pd.merge_asof(left, day[[*keys, "date", *cols]].sort_values("date"), on="date", by=keys,
                            allow_exact_matches=False)
        for c in cols:
            df[c] = np.nan
            df.loc[df.index[got["_i"].to_numpy()], c] = got[c].to_numpy()
    df["front_x_course_edge"] = (0.5 - df["early_pos_rel_3"]) * df["course_front_edge_5"]  # 前に付ける馬ほど大きい
    return df


def _prior_mean(series: pd.Series, by: pd.Series, window: int) -> pd.Series:
    """by ごとに、1つ前までの window 件の平均(今回の値は含めない)。"""
    shifted = series.groupby(by).shift(1)
    return shifted.groupby(by).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)


def _past_pace_z(df: pd.DataFrame) -> pd.Series:
    """1レース1票で、前日までの同条件の平均・標本標準偏差を基準にする。"""
    keys = ["course_type", "dist_bucket"]
    races = df[["race_id", "date", *keys, "pace_first3f"]].drop_duplicates("race_id").copy()
    races["x"] = pd.to_numeric(races["pace_first3f"], errors="coerce")
    races["x2"] = races["x"] ** 2
    day = races.groupby(keys + ["date"], observed=True).agg(
        n=("x", "count"), s=("x", "sum"), s2=("x2", "sum")).reset_index().sort_values("date")
    for col in ("n", "s", "s2"):
        day["prior_" + col] = day.groupby(keys, observed=True)[col].cumsum() - day[col]
    n = day["prior_n"].replace(0, np.nan)
    day["mean"] = day["prior_s"] / n
    variance = (day["prior_s2"] - day["prior_s"] ** 2 / n) / (n - 1).where(n > 1)
    # タイムは0.1秒刻み。一定値の二乗和の丸め誤差を分散とみなさない。
    day["std"] = np.sqrt(variance.where(variance > 1e-10))
    races = races.merge(day[keys + ["date", "mean", "std"]], on=keys + ["date"], how="left")
    races["z"] = (races["x"] - races["mean"]) / races["std"]
    return df["race_id"].map(races.set_index("race_id")["z"])


def _article_features(df: pd.DataFrame) -> pd.DataFrame:
    """記事にあるファクターのうち、結果表だけから作れるもの(適性・内容・ローテーション・騎手・ペース)。"""
    h = df["horse_id"]
    g = df.groupby("horse_id", sort=False)
    for c in ("direction", "pace_first3f", "pace_last3f", "sire_id", "damsire_id"):
        if c not in df:
            df[c] = np.nan

    # --- 適性: 同じ条件での過去の3着以内率(馬) ---
    for col, key in (("horse_ct_top3_rate", "course_type"), ("horse_venue_top3_rate", "venue"),
                     ("horse_dir_top3_rate", "direction"), ("horse_ground_top3_rate", "ground")):
        df[col] = _expanding_rate(df, ["horse_id", key], "top3")[1]
    df["dist_bucket_s"] = df["dist_bucket"].astype("float")
    course = ["venue", "course_type", "dist_bucket_s"]
    df["horse_course_n"], df["horse_course_top3_rate"] = _expanding_rate(df, ["horse_id", *course], "top3")
    df["horse_ctground_top3_rate"] = _expanding_rate(df, ["horse_id", "course_type", "ground"], "top3")[1]
    df["horse_race_n"], df["horse_race_top3_rate"] = _expanding_rate(df, ["horse_id", "race_name"], "top3")
    same_ct = df.groupby(["horse_id", "course_type"], sort=False)["finish"]
    df["prev_same_ct_finish"] = same_ct.shift(1)

    # --- 内容: 上がり順位・タイム差・位置取りの上がり(直近の平均) ---
    df["l3f_rank_rel"] = df.groupby("race_id")["last3f"].rank(method="min") / df["field_size"]
    df["time_diff"] = df["time_sec"] - df.groupby("race_id")["time_sec"].transform("min")
    df["pos_gain"] = (df["passing"].map(parse_early_position) - df["passing"].map(parse_late_position)) / df["field_size"]
    df["l3f_rank_3"] = _prior_mean(df["l3f_rank_rel"], h, 3)
    df["time_diff_5"] = _prior_mean(df["time_diff"], h, 5)
    df["pos_gain_3"] = _prior_mean(df["pos_gain"], h, 3)
    prior_bw = g["body_weight"].shift(1).groupby(h).expanding().mean().reset_index(level=0, drop=True)
    df["bw_vs_avg"] = df["body_weight"] - prior_bw
    df["weight_ratio"] = df["weight_carried"] / df["body_weight"]
    df["weight_change"] = df["weight_carried"] - g["weight_carried"].shift(1)
    df["prev_pop_gap"] = g["popularity"].shift(1) - g["finish"].shift(1)  # 前走の人気 − 着順(正なら人気以上に走った)
    df["dist_shorten300"] = (df["distance_change"] <= -300).astype(float).where(df["distance_change"].notna())
    df["dist_extend300"] = (df["distance_change"] >= 300).astype(float).where(df["distance_change"].notna())

    # --- ローテーション ---
    df["weeks_since_last"] = df["days_since_last"] / 7
    df["is_layoff"] = (df["days_since_last"] >= 90).astype(float).where(df["days_since_last"].notna())
    new_seg = (df["days_since_last"].isna() | (df["days_since_last"] >= 90)).astype(int)
    seg = new_seg.groupby(h).cumsum()
    df["runs_since_layoff"] = df.groupby([h, seg]).cumcount() + 1  # 休み明け(または初戦)を1とする
    df["is_renso"] = (df["days_since_last"] <= 8).astype(float).where(df["days_since_last"].notna())
    # 前走の内容。タイム差は勝ち馬との差(秒)、前半3Fは同じ芝ダ・距離帯の中でのz値(小さいほど速いペース)
    pace_z = _past_pace_z(df)
    df["prev_time_diff"] = df.groupby("horse_id", sort=False)["time_diff"].shift(1).astype(float)
    df["prev_pace_z"] = pace_z.groupby(df["horse_id"], sort=False).shift(1)
    df["prev_distance"] = g["distance"].shift(1)
    df["prev2_time_diff"] = df.groupby("horse_id", sort=False)["time_diff"].shift(2).astype(float)
    df["prev_l3f_rank"] = g["l3f_rank_rel"].shift(1)
    df["prev_early_pos"] = g["early_pos"].shift(1)
    df["prev_pos_gain"] = g["pos_gain"].shift(1)
    df["prev_class"] = g["race_class"].shift(1)
    df["prev_field_size"] = g["field_size"].shift(1)
    df["prev2_finish"] = g["finish"].shift(2)
    short = (df["days_since_last"] <= 21).astype(float).where(df["days_since_last"].notna())  # 3週以内
    hard = ((df["prev_time_diff"] <= 0.3) | (df["prev_pace_z"] <= -1)).astype(float).where(
        df["prev_time_diff"].notna() | df["prev_pace_z"].notna())
    df["hard_prev"] = hard
    df["load_x_short"] = df["prev_pace_z"] * short
    df["load_per_day"] = df["prev_pace_z"] / df["days_since_last"].replace(0, np.nan)
    df["hard_prev_short"] = hard * short
    df["dist_x_short"] = df["prev_distance"] * short

    # --- 騎手・厩舎(同じ日の他レースは含めない) ---
    df["jockey_ct_top3_rate"] = _expanding_rate(df, ["jockey_id", "course_type"], "top3")[1]
    df["jockey_dist_top3_rate"] = _expanding_rate(df, ["jockey_id", "dist_bucket_s"], "top3")[1]
    df["jockey_venue_top3_rate"] = _expanding_rate(df, ["jockey_id", "venue"], "top3")[1]
    df["jockey_course_n"], df["jockey_course_top3_rate"] = _expanding_rate(df, ["jockey_id", *course], "top3")
    df["jockey_ctground_top3_rate"] = _expanding_rate(df, ["jockey_id", "course_type", "ground"], "top3")[1]
    df["jockey_class_top3_rate"] = _expanding_rate(df, ["jockey_id", "race_class"], "top3")[1]
    df["_ret"] = df["win"] * df["win_odds"]  # 単勝を買い続けたときの払戻(倍)。オッズがなければ NaN
    df["jockey_roi"] = _expanding_rate(df, "jockey_id", "_ret")[1]
    df["pair_n"], df["pair_top3_rate"] = _expanding_rate(df, ["horse_id", "jockey_id"], "top3")
    df["tj_n"], df["tj_top3_rate"] = _expanding_rate(df, ["trainer_id", "jockey_id"], "top3")
    # 回数が少ない組み合わせは、全体の3着内率(前日まで)に寄せる: (3着内の数 + k×全体の率) ÷ (回数 + k)
    df["_all"] = 0
    overall = _expanding_rate(df, "_all", "top3")[1]
    hits = (df["tj_top3_rate"] * df["tj_n"]).fillna(0.0)
    df["tj_top3_shr"] = (hits + TJ_SHRINK_K * overall) / (df["tj_n"] + TJ_SHRINK_K)

    # --- ペース: 出走馬の脚質の組み合わせ(各馬の過去の位置取りだけを使う) ---
    ep = df["early_pos"]
    df["style_front_rate"] = _prior_mean((ep <= 0.25).astype(float).where(ep.notna()), h, 5)
    df["style_closer_rate"] = _prior_mean((ep >= 0.7).astype(float).where(ep.notna()), h, 5)
    front = (df["early_pos_rel_3"] <= 0.3).astype(float).where(df["early_pos_rel_3"].notna())
    df["n_front_runners"] = front.groupby(df["race_id"]).transform("sum")
    df["front_share"] = df["n_front_runners"] / df["field_size"]
    df["field_style_mean"] = df.groupby("race_id")["early_pos_rel_3"].transform("mean")
    df["style_vs_field"] = df["early_pos_rel_3"] - df["field_style_mean"]
    df["inner_pos"] = df["number"] / df["field_size"]

    # --- ラップ: 過去のレースのペースと、自分の上がりが場の後半3Fよりどれだけ速かったか ---
    for c in ("pace_first3f", "pace_last3f"):
        df[c] = pd.to_numeric(df[c], errors="coerce")  # DB の NULL は object で来る
    df["pace_diff"] = df["pace_first3f"] - df["pace_last3f"]
    df["l3f_vs_lap"] = df["last3f"] - df["pace_last3f"]
    df["prev_pace_diff"] = g["pace_diff"].shift(1)
    df["pace_diff_3"] = _prior_mean(df["pace_diff"], h, 3)
    df["l3f_vs_lap_3"] = _prior_mean(df["l3f_vs_lap"], h, 3)
    df["prev_l3f_vs_lap"] = g["l3f_vs_lap"].shift(1)

    # --- 血統: 父・母の父の産駒の過去成績(同じ日の他レースは含めない) ---
    df["sire_n"], df["sire_top3_rate"] = _expanding_rate(df, "sire_id", "top3")
    df["sire_ctdist_top3_rate"] = _expanding_rate(df, ["sire_id", "course_type", "dist_bucket_s"], "top3")[1]
    df["sire_ground_top3_rate"] = _expanding_rate(df, ["sire_id", "course_type", "ground"], "top3")[1]
    df["damsire_n"], df["damsire_top3_rate"] = _expanding_rate(df, "damsire_id", "top3")
    df["damsire_ctdist_top3_rate"] = _expanding_rate(df, ["damsire_id", "course_type", "dist_bucket_s"], "top3")[1]

    # --- 調教: そのレースの最終追い切りの評価と、メンバー内での差 ---
    for c in ("train_rank", "train_critic"):
        if c not in df:
            df[c] = np.nan
    df["train_rank_score"] = df["train_rank"].map(TRAIN_RANK_SCORE).astype(float)
    df["train_rank_vs_field"] = df["train_rank_score"] - df.groupby("race_id")["train_rank_score"].transform("mean")
    return df.drop(columns=["dist_bucket_s", "_ret", "_all", "l3f_rank_rel", "time_diff", "pos_gain", "pace_diff", "l3f_vs_lap"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce").astype(float)
    # 取消・除外は出走数にも履歴にも入れない。未知の結果を予測対象に変換しない。
    if "status" in df:
        df = df[~df["status"].isin(["取", "除"])].copy()
    if "is_entry" not in df:
        df = df.assign(is_entry=~result_mask(df))
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

    df = _article_features(df)
    df = _course_flow(df)
    bias = _track_bias(df)
    df = df.merge(bias, on=["venue", "course_type", "date"], how="left")
    for c in BIAS_FEATURES:
        if c not in df:
            df[c] = np.nan

    # --- 騎手・調教師(前日までの成績) ---
    df["jockey_n"], df["jockey_win_rate"] = _expanding_rate(df, "jockey_id", "win")
    _, df["jockey_top3_rate"] = _expanding_rate(df, "jockey_id", "top3")
    df["trainer_n"], df["trainer_top3_rate"] = _expanding_rate(df, "trainer_id", "top3")

    df = add_speed_figures(df)
    df = add_relative(df)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


FIGURE_MIN_RACES = 5  # 基準タイムに使う過去の勝ちタイムがこれ未満のコースは、指数を出さない


def add_speed_figures(df: pd.DataFrame) -> pd.DataFrame:
    """走破タイムの指数: 1000mあたり、基準タイムより何秒速かったか(大きいほど速い)。当日の馬場差を引く。

    基準タイム = 同じ競馬場・芝ダ・距離・馬場(良とそれ以外)の、その日より前の勝ちタイムの平均。
    当日の馬場差 = 同じ日・競馬場・芝ダの勝ち馬の、基準からのずれの平均(2レース以上あるとき)。
    指数はその日の結果から出すので、特徴量には前走以前の指数だけを使う。
    """
    t = pd.to_numeric(df["time_sec"], errors="coerce") if "time_sec" in df else pd.Series(np.nan, index=df.index)
    ok = df["finish"].notna() & (t > 0)
    good = df["ground"].astype("string").fillna("").str.startswith("良")
    key = (df["venue"].astype(str) + "|" + df["course_type"].astype(str) + "|" + df["distance"].astype(str) + "|"
           + good.map({True: "良", False: "道悪"}))
    win = ok & (df["finish"] == 1) & ~df["race_id"].duplicated()
    daily = (pd.DataFrame({"key": key[win], "date": df.loc[win, "date"], "t": t[win]})
             .groupby(["key", "date"])["t"].agg(["sum", "count"]).reset_index().sort_values(["key", "date"]))
    g = daily.groupby("key")
    n_before = g["count"].cumsum() - daily["count"]
    daily["std"] = ((g["sum"].cumsum() - daily["sum"]) / n_before).where(n_before >= FIGURE_MIN_RACES)
    std = pd.DataFrame({"key": key, "date": df["date"]}).merge(
        daily[["key", "date", "std"]], on=["key", "date"], how="left")["std"].to_numpy()
    dev = pd.Series(np.where(ok, (t.to_numpy() - std) / df["distance"].astype(float).to_numpy() * 1000, np.nan),
                    index=df.index)
    day = pd.DataFrame({"date": df["date"], "venue": df["venue"].astype(str), "course_type": df["course_type"].astype(str)})
    var = (day[df["finish"] == 1].assign(dev=dev).groupby(["date", "venue", "course_type"])["dev"]
           .agg(["mean", "count"]).reset_index())
    var["var"] = var["mean"].where(var["count"] >= 2)
    day_var = day.merge(var[["date", "venue", "course_type", "var"]], on=["date", "venue", "course_type"],
                        how="left")["var"].fillna(0).to_numpy()
    prev = (-(dev - day_var)).groupby(df["horse_id"], sort=False).shift(1)
    pg = prev.groupby(df["horse_id"], sort=False)
    out = pd.DataFrame({"prev_fig": prev,
                        "best_fig_3": pg.rolling(3, min_periods=1).max().reset_index(level=0, drop=True),
                        "avg_fig_3": pg.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)}, index=df.index)
    out["best_fig_vs_field"] = out["best_fig_3"] - out["best_fig_3"].groupby(df["race_id"]).transform("mean")
    return pd.concat([df.drop(columns=[c for c in out if c in df]), out], axis=1)


def add_relative(df: pd.DataFrame) -> pd.DataFrame:
    """RELATIVE_BASE の列の、同じレースの中での標準化(_z)と順位の割合(_rk。値が大きいほど 1 に近い)。"""
    cols = {}
    for c in RELATIVE_BASE:
        x = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
        by = x.groupby(df["race_id"])
        cols[f"{c}_z"] = (x - by.transform("mean")) / by.transform("std").replace(0, np.nan)
        cols[f"{c}_rk"] = by.rank(pct=True)
    return pd.concat([df.drop(columns=[c for c in cols if c in df]), pd.DataFrame(cols, index=df.index)], axis=1)
