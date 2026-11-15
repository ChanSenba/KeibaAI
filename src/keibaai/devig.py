"""単勝オッズから市場の勝率を作る方法(控除の剥がし方)。

現行は比例方式(オッズの逆数をレース内で合計1に正規化)で、控除を人気薄にも比例して
割り振るため、favourite-longshot bias をそのまま市場確率に持ち込む。
Shin (1991, 1993) は、インサイダー(情報を持つ賭け手)の割合 z がいる市場では
ブックメーカーが人気薄ほどオッズを厚く削る、という均衡から z を推定し、
人気薄側に厚く乗った控除を剥がす。z はレースごとに「Shin 確率の合計が1」を満たすよう二分探索で解く。

ここは方法の部品だけを置き、どちらが JRA で当たるかは devig_trials で比べる(本番の経路はまだ変えない)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def proportional(odds: np.ndarray) -> np.ndarray:
    """比例方式。odds は 1 レース分の単勝オッズ(正の値だけ)。"""
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum()


def _shin_at_z(inv: np.ndarray, booksum: float, z: float) -> np.ndarray:
    """z を固定したときの Shin 確率。inv はオッズの逆数、booksum はその合計。"""
    if z <= 0.0:
        return inv / np.sqrt(booksum)
    return (np.sqrt(z * z + 4.0 * (1.0 - z) * inv * inv / booksum) - z) / (2.0 * (1.0 - z))


def shin(odds: np.ndarray) -> tuple[np.ndarray, float]:
    """Shin 方式の (確率, 推定した z)。odds は 1 レース分の単勝オッズ(正の値だけ)。

    控除がない(booksum <= 1)レースでは比例方式と同じものを z=0 で返す。
    z の二分探索は上限を広げながら行い、収束しない場合は比例方式に倒す(呼び出し側が分かるよう z は NaN)。
    """
    odds = np.asarray(odds, dtype=float)
    inv = 1.0 / odds
    booksum = float(inv.sum())
    if booksum <= 1.0 or len(odds) < 2:
        return proportional(odds), 0.0
    lo, hi = 0.0, 0.5
    while _shin_at_z(inv, booksum, hi).sum() > 1.0 and hi < 0.99:
        hi = 1.0 - (1.0 - hi) / 2.0
    if _shin_at_z(inv, booksum, hi).sum() > 1.0:
        return proportional(odds), float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _shin_at_z(inv, booksum, mid).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2.0
    p = _shin_at_z(inv, booksum, z)
    return p / p.sum(), z


def market_probs(df: pd.DataFrame, method: str = "proportional") -> pd.Series:
    """df(win_odds, race_id)の各行の市場確率。オッズのない馬・そろわないレースは NaN。

    method は 'proportional'(現行)か 'shin'。レース単位で作るので、同じレースの馬は同じ方法・同じ z で扱う。
    行の対応は位置で持つので、df の index が重複していても壊れない。
    """
    if method not in ("proportional", "shin"):
        raise ValueError(f"method は proportional か shin: {method}")
    odds = pd.to_numeric(df["win_odds"], errors="coerce").to_numpy(dtype=float)
    race = df["race_id"].to_numpy()
    out = np.full(len(df), np.nan)
    pos = pd.Series(np.arange(len(df)))
    for _, idx in pos.groupby(race).groups.items():
        o = odds[idx]
        if not (o > 0).all():
            continue
        out[idx] = proportional(o) if method == "proportional" else shin(o)[0]
    return pd.Series(out, index=df.index)


def market_z(df: pd.DataFrame) -> pd.Series:
    """レースごとの Shin の z(推定できなかったレースは NaN)。market_probs と同じ対象のレースだけ返す。"""
    odds = pd.to_numeric(df["win_odds"], errors="coerce").to_numpy(dtype=float)
    race = df["race_id"].to_numpy()
    pos = pd.Series(np.arange(len(df)))
    rows = {}
    for race_id, idx in pos.groupby(race).groups.items():
        o = odds[idx]
        if (o > 0).all():
            rows[race_id] = shin(o)[1]
    return pd.Series(rows, dtype=float)
