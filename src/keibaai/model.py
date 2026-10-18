"""LightGBM で勝率・3着以内率を予測し、時系列で評価する。"""
from __future__ import annotations

import os
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import ABILITY_MODEL_FEATURES, CATEGORICAL, MARKET_MODEL_FEATURES

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
# 学習に使うレースの開始日。これより前の成績も DB にあり、過去走の特徴量(通算成績など)には使うが、学習の行には入れない。
# 環境変数 KEIBAAI_TRAIN_FROM(YYYY-MM-DD)で変えられる
TRAIN_FROM = os.environ.get("KEIBAAI_TRAIN_FROM", "2019-01-01")


def train_rows(train: pd.DataFrame) -> pd.DataFrame:
    """学習の行を TRAIN_FROM 以後のレースに絞る。"""
    return train[train["date"] >= pd.Timestamp(TRAIN_FROM)]


def auc(y: np.ndarray, p: np.ndarray) -> float:
    order = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((order[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def normalize_within_race(df: pd.DataFrame, p: np.ndarray, total: float = 1.0) -> np.ndarray:
    s = pd.Series(p, index=df.index).groupby(df["race_id"]).transform("sum")
    return (p / s.to_numpy()) * total


@dataclass
class Report:
    name: str
    n_races: int
    win_logloss: float
    win_auc: float
    top3_auc: float
    top1_hit_rate: float  # モデルが1番手にした馬が勝った割合
    top1_roi: float  # その馬の単勝を毎回100円買った回収率(%)


def _fit(train: pd.DataFrame, features: list[str], target: str, rounds: int):
    train = train_rows(train)
    cats = [c for c in CATEGORICAL if c in features]
    ds = lgb.Dataset(train[features], train[target], categorical_feature=cats, free_raw_data=False)
    return lgb.train(PARAMS, ds, num_boost_round=rounds)


def evaluate(df: pd.DataFrame, test_from: str, use_market: bool, rounds: int = 300) -> tuple[Report, pd.DataFrame]:
    features = MARKET_MODEL_FEATURES if use_market else ABILITY_MODEL_FEATURES
    cut = pd.Timestamp(test_from)
    train, test = df[df["date"] < cut], df[df["date"] >= cut].copy()
    if train.empty or test.empty:
        raise ValueError("学習期間または評価期間にデータがありません")
    win_m = _fit(train, features, "win", rounds)
    top3_m = _fit(train, features, "top3", rounds)
    test["p_win_raw"] = win_m.predict(test[features])
    test["p_win"] = normalize_within_race(test, test["p_win_raw"].to_numpy())
    test["p_top3"] = top3_m.predict(test[features])

    top = test.loc[test.groupby("race_id")["p_win"].idxmax()]
    hit = float(top["win"].mean())
    payout = (top["win"] * top["win_odds"].fillna(0) * 100).sum()
    roi = float(payout / (100 * len(top)) * 100) if len(top) else float("nan")
    name = "オッズあり" if use_market else "オッズなし(能力のみ)"
    return Report(name, int(test["race_id"].nunique()), logloss(test["win"].to_numpy(), test["p_win"].to_numpy()),
                  auc(test["win"].to_numpy(), test["p_win"].to_numpy()),
                  auc(test["top3"].to_numpy(), test["p_top3"].to_numpy()), hit, roi), test


def baseline_market(test: pd.DataFrame) -> Report:
    """人気・オッズだけの基準。確率は 1/単勝オッズ をレース内で正規化したもの。"""
    t = test.dropna(subset=["popularity", "win_odds"]).copy()
    p = normalize_within_race(t, (1.0 / t["win_odds"]).to_numpy())
    top = t.loc[t.groupby("race_id")["popularity"].idxmin()]
    roi = float((top["win"] * top["win_odds"] * 100).sum() / (100 * len(top)) * 100)
    return Report("基準:オッズ/1番人気", int(t["race_id"].nunique()), logloss(t["win"].to_numpy(), p),
                  auc(t["win"].to_numpy(), p), float("nan"), float(top["win"].mean()), roi)


def format_report(reports: list[Report]) -> str:
    lines = [f"{'モデル':<18}{'レース数':>8}{'勝LogLoss':>11}{'勝AUC':>8}{'複AUC':>8}{'1番手的中':>10}{'単勝回収率':>10}"]
    for r in reports:
        lines.append(f"{r.name:<18}{r.n_races:>8}{r.win_logloss:>11.4f}{r.win_auc:>8.3f}{r.top3_auc:>8.3f}"
                     f"{r.top1_hit_rate:>10.1%}{r.top1_roi:>9.1f}%")
    return "\n".join(lines)
