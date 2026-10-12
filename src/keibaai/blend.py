"""モデルの勝率と、オッズから出した市場の勝率を、レース単位の対数オッズで混ぜる(ベンター方式)。

p_i ∝ exp(a * log(モデル勝率_i) + b * log(市場勝率_i)) をレース内で正規化し、a と b は
「混ぜ方を決める期間」の結果から、勝ち馬の確率が最大になるように決める。評価期間の結果は使わない。
a=0, b=1 なら市場そのまま、a>0 ならモデルの見立てを足した形になる。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import evaluate, normalize_within_race


@dataclass
class Blend:
    a: float  # モデルの重み
    b: float  # 市場の重み
    loglik: float  # 決めた期間での、1レースあたりの勝ち馬の対数尤度
    baseline_loglik: float  # 同じ期間で市場だけ(a=0, b=1)にした場合


def market_prob(df: pd.DataFrame) -> pd.Series:
    """単勝オッズの逆数をレース内で合計1にした確率。"""
    return pd.Series(normalize_within_race(df, (1.0 / df["win_odds"]).to_numpy()), index=df.index)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """オッズとモデル勝率がそろい、勝ち馬がちょうど1頭いるレースだけにする。"""
    t = df.dropna(subset=["win_odds", "p_win"]).copy()
    t = t[(t["win_odds"] > 0) & (t["p_win"] > 0)]
    t["p_mkt"] = market_prob(t)
    ok = t.groupby("race_id")["win"].transform("sum") == 1
    return t[ok].sort_values(["race_id", "number"]).reset_index(drop=True)


def _blend_probs(t: pd.DataFrame, a: float, b: float) -> np.ndarray:
    z = a * np.log(t["p_win"].to_numpy()) + b * np.log(t["p_mkt"].to_numpy())
    race = pd.factorize(t["race_id"])[0]
    zmax = np.zeros(race.max() + 1)
    np.maximum.at(zmax, race, z)  # オーバーフロー防止
    e = np.exp(z - zmax[race])
    return e / np.bincount(race, weights=e)[race]


def _loglik(t: pd.DataFrame, a: float, b: float) -> float:
    p = _blend_probs(t, a, b)
    return float(np.log(np.clip(p[t["win"].to_numpy() == 1], 1e-12, 1)).mean())


def fit_blend(valid: pd.DataFrame) -> Blend:
    """valid は p_win(モデル)・win_odds・win・race_id・number を持つ、混ぜ方を決める期間の表。"""
    t = _prepare(valid)
    if t.empty:
        raise ValueError("混ぜ方を決める期間に、オッズとモデル勝率がそろったレースがありません")
    best = (-np.inf, 0.0, 1.0)
    for a_grid, b_grid in ((np.arange(0, 1.51, 0.1), np.arange(0.4, 1.61, 0.1)), (None, None)):
        if a_grid is None:  # 粗い探索の近くを細かく見直す
            a0, b0 = best[1], best[2]
            a_grid, b_grid = np.arange(a0 - 0.1, a0 + 0.101, 0.02), np.arange(b0 - 0.1, b0 + 0.101, 0.02)
        for a in a_grid:
            for b in b_grid:
                ll = _loglik(t, float(a), float(b))
                if ll > best[0]:
                    best = (ll, float(a), float(b))
    return Blend(round(best[1], 3), round(best[2], 3), best[0], _loglik(t, 0.0, 1.0))


def apply_blend(df: pd.DataFrame, blend: Blend) -> pd.DataFrame:
    """df の p_win を混ぜた確率に置き換える(元の確率は p_model に残す)。混ぜられないレースは除く。"""
    d = df.dropna(subset=["win_odds", "p_win"]).copy()
    d = d[(d["win_odds"] > 0) & (d["p_win"] > 0)]
    d["p_mkt"] = market_prob(d)
    d = d.sort_values(["race_id", "number"]).reset_index(drop=True)
    d["p_model"] = d["p_win"]
    d["p_win"] = _blend_probs(d, blend.a, blend.b)
    return d


def predict_test_blend(df: pd.DataFrame, test_from: str, valid_months: int = 12) -> tuple[pd.DataFrame, Blend]:
    """評価期間の各馬に、混ぜた勝率(p_win)を付ける。

    混ぜ方は test_from の valid_months か月前から test_from まで(学習はその前だけ)の結果で決め、
    評価期間にはそこで決めた a, b と、test_from より前で学習したモデルを使う。
    """
    cut = pd.Timestamp(test_from)
    valid_from = cut - pd.DateOffset(months=valid_months)
    _, valid = evaluate(df[df["date"] < cut], str(valid_from.date()), use_market=False)
    blend = fit_blend(valid)
    _, test = evaluate(df, test_from, use_market=False)
    return apply_blend(test, blend), blend
