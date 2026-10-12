"""要素ごとの予想。馬の質・騎手と厩舎・展開と馬場を別々のモデルで勝率にし、並べて見られるようにする。

どの要素が当たりに効いているかは、`keibaai factors --test-from <日付>` で確かめられる。
調教と血統は、まだデータを集めていないので入っていない。
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import BASE_FEATURES, BIAS_FEATURES, CATEGORICAL
from .model import PARAMS, auc, logloss, normalize_within_race

FACTORS: dict[str, list[str]] = {
    "馬の質": ["age", "weight_carried", "body_weight", "body_weight_diff", "days_since_last", "n_prev", "prev_finish",
              "avg_finish_3", "avg_finish_5", "win_rate", "top3_rate", "avg_last3f_3", "same_dist_top3_rate",
              "race_class", "class_change", "distance_change", "distance", "course_type", "ground"],
    "騎手・厩舎": ["jockey_changed", "jockey_n", "jockey_win_rate", "jockey_top3_rate", "trainer_n", "trainer_top3_rate"],
    "展開・馬場": ["number", "bracket", "field_size", "distance", "course_type", "venue", "ground", "early_pos_rel_3",
                 *BIAS_FEATURES],
}
OVERALL = {"総合": BASE_FEATURES, "総合+馬場の偏り": BASE_FEATURES + BIAS_FEATURES}
COLUMN = {"馬の質": "p_win_quality", "騎手・厩舎": "p_win_human", "展開・馬場": "p_win_track"}


def _fit(train: pd.DataFrame, features: list[str], rounds: int):
    cats = [c for c in CATEGORICAL if c in features]
    return lgb.train(PARAMS, lgb.Dataset(train[features], train["win"], categorical_feature=cats), rounds)


def predict_factors(train: pd.DataFrame, target: pd.DataFrame, factors: dict[str, list[str]] | None = None,
                    rounds: int = 300) -> pd.DataFrame:
    """target の各馬に、要素ごとの勝率(レース内で合計 1)を付けた表を返す。列名は要素名。"""
    out = pd.DataFrame(index=target.index)
    for name, feats in (factors or FACTORS).items():
        out[name] = normalize_within_race(target, _fit(train, feats, rounds).predict(target[feats]))
    return out


@dataclass
class FactorReport:
    name: str
    n_features: int
    win_logloss: float
    win_auc: float
    top1_hit_rate: float  # その要素だけで1番手にした馬が勝った割合


def evaluate_factors(df: pd.DataFrame, test_from: str, rounds: int = 300) -> tuple[list[FactorReport], float]:
    """評価開始日より前で学習し、以後で要素ごとの成績を返す。あわせて「全馬が同じ勝率」の LogLoss も返す。"""
    cut = pd.Timestamp(test_from)
    train, test = df[df["date"] < cut], df[df["date"] >= cut].copy()
    if train.empty or test.empty:
        raise ValueError("学習期間または評価期間にデータがありません")
    sets = {**FACTORS, **OVERALL}
    probs = predict_factors(train, test, sets, rounds)
    y = test["win"].to_numpy()
    reports = []
    for name, feats in sets.items():
        p = probs[name]
        top = test.loc[p.groupby(test["race_id"]).idxmax()]
        reports.append(FactorReport(name, len(feats), logloss(y, p.to_numpy()), auc(y, p.to_numpy()),
                                    float(top["win"].mean())))
    uniform = logloss(y, (1.0 / test["field_size"]).to_numpy())
    return reports, uniform


def format_factor_report(reports: list[FactorReport], uniform: float) -> str:
    lines = [f"{'要素':<16}{'特徴量':>6}{'勝LogLoss':>11}{'勝AUC':>8}{'1番手的中':>10}"]
    for r in reports:
        lines.append(f"{r.name:<{16 - sum(ord(c) > 255 for c in r.name)}}{r.n_features:>6}{r.win_logloss:>11.4f}"
                     f"{r.win_auc:>8.3f}{r.top1_hit_rate:>10.1%}")
    lines.append(f"(全馬が同じ勝率なら LogLoss は {uniform:.4f}。これより小さいほど情報がある)")
    return "\n".join(lines)
