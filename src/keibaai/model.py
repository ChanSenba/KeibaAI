"""LightGBM で勝率・3着以内率を予測し、時系列で評価する。"""
from __future__ import annotations

import os
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import ABILITY_MODEL_FEATURES, CATEGORICAL, MARKET_MODEL_FEATURES, RESIDUAL_MODEL_FEATURES

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
# 学習に使うレースの開始日。これより前の成績も DB にあり、過去走の特徴量(通算成績など)には使うが、学習の行には入れない。
# 環境変数 KEIBAAI_TRAIN_FROM(YYYY-MM-DD)で変えられる
TRAIN_FROM = os.environ.get("KEIBAAI_TRAIN_FROM", "2019-01-01")
# 馬体重は発走の約1時間前に出るので、前日や朝に保存する予想では空になる。学習の行には必ず入っているため、
# 空のまま予想すると精度が落ちる。学習の行の一部で馬体重の列を空にして、空のときの見方も学ばせる。
# 2021〜2026年を1年ずつ評価(その年より前で学習)して、オッズ込みの LogLoss は馬体重が空のとき
# 0.20613 → 0.20517(6年とも改善)、あるときも 0.20535 → 0.20521。能力のみも両方で改善した
BODY_WEIGHT_FEATURES = ["body_weight", "body_weight_diff", "bw_vs_avg", "weight_ratio"]
BODY_WEIGHT_MASK_RATE = 0.3


def train_rows(train: pd.DataFrame) -> pd.DataFrame:
    """学習の行を TRAIN_FROM 以後のレースに絞り、BODY_WEIGHT_MASK_RATE の割合の行で馬体重の列を空にする。"""
    train = train[train["date"] >= pd.Timestamp(TRAIN_FROM)]
    cols = [c for c in BODY_WEIGHT_FEATURES if c in train]
    if not cols or train.empty:
        return train
    hide = np.random.default_rng(0).random(len(train)) < BODY_WEIGHT_MASK_RATE
    train = train.copy()
    train.loc[hide, cols] = np.nan
    return train


# オッズ込みの勝率のモデル。市場の勝率(単勝オッズの逆数をレース内で合計1にしたもの)の対数オッズを出発点にして、
# そこからのずれだけを学ぶ。木を小さく、学習率を下げて、市場から離れすぎないようにする
# (300回・葉31のままだと、モデル単独では市場より悪く、混ぜて初めて少し足せる程度だった)
RESIDUAL_PARAMS = dict(PARAMS, learning_rate=0.02, num_leaves=15, min_data_in_leaf=200)
RESIDUAL_ROUNDS = 1000


def market_logit(df: pd.DataFrame) -> np.ndarray:
    """市場の勝率の対数オッズ。オッズのない馬は NaN。"""
    odds = pd.to_numeric(df["win_odds"], errors="coerce")
    inv = (1.0 / odds).where(odds > 0)
    p = (inv / inv.groupby(df["race_id"]).transform("sum")).clip(1e-6, 1 - 1e-6).to_numpy()
    return np.log(p / (1 - p))


def has_market(df: pd.DataFrame) -> pd.Series:
    """レースの全馬に単勝オッズがある行。"""
    odds = pd.to_numeric(df["win_odds"], errors="coerce")
    return (odds > 0).groupby(df["race_id"]).transform("all")


def fit_residual(train: pd.DataFrame, rounds: int = RESIDUAL_ROUNDS) -> lgb.Booster:
    """オッズ込みの勝率のモデル(市場の見立てからのずれを学ぶ)。オッズのそろったレースだけで学習する。"""
    train = train_rows(train[has_market(train)])
    feats = RESIDUAL_MODEL_FEATURES
    cats = [c for c in CATEGORICAL if c in feats]
    ds = lgb.Dataset(train[feats], train["win"], categorical_feature=cats, init_score=market_logit(train),
                     free_raw_data=False)
    return lgb.train(RESIDUAL_PARAMS, ds, num_boost_round=rounds)


def predict_residual(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    """fit_residual のモデルの勝率(レース内で合計1にする前)。オッズのない馬は NaN。"""
    z = model.predict(df[RESIDUAL_MODEL_FEATURES], raw_score=True) + market_logit(df)
    return 1.0 / (1.0 + np.exp(-z))


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
    top3_m = _fit(train, features, "top3", rounds)
    if use_market:  # 勝率は市場の見立てからのずれを学ぶモデル。オッズのそろわないレースは評価から外す
        test = test[has_market(test)].copy()
        test["p_win_raw"] = predict_residual(fit_residual(train), test)
    else:
        test["p_win_raw"] = _fit(train, features, "win", rounds).predict(test[features])
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
