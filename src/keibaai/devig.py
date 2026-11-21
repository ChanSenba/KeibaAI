"""単勝オッズから市場の勝率を作る方法(控除の剥がし方)。

現行は比例方式(オッズの逆数をレース内で合計1に正規化)で、控除を人気薄にも比例して
割り振るため、favourite-longshot bias をそのまま市場確率に持ち込む。
Shin (1991, 1993) は、インサイダー(情報を持つ賭け手)の割合 z がいる市場では
ブックメーカーが人気薄ほどオッズを厚く削る、という均衡から z を推定し、
人気薄側に厚く乗った控除を剥がす。z はレースごとに「Shin 確率の合計が1」を満たすよう二分探索で解く。
Power 法は、暗黙確率を指数 γ で累乗して正規化する1パラメータの方法で、γ はレースごとには
推定せず、評価より前の期間で1つ決めて使う。

なお、JRA の単勝はパリミュチュエル(控除はプールから一律に差し引かれる)なので、比例方式は
「売上のシェアの復元」として正確であり、Shin や Power は控除の剥がし方(devig)というより、
favourite-longshot bias の補正として位置づけるのが正しい。海外のブックメーカー市場を対象にした
比較 (Clarke, Kovalchik & Ingram 2017) では、Power が比例方式を上回り、Shin と同等以上だったと
報告されている(パリミュチュエルで同じかは、ここではなく devig_trials の比較で確かめる)。

ここは方法の部品だけを置き、どちらが JRA で当たるかは devig_trials で比べる(本番の経路はまだ変えない)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def proportional(odds: np.ndarray) -> np.ndarray:
    """比例方式。odds は 1 レース分の単勝オッズ(正の値だけ)。"""
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum()


def power(odds: np.ndarray, gamma: float) -> np.ndarray:
    """Power 法。odds は 1 レース分の単勝オッズ(正の値だけ)、gamma は期間で1つ決める指数。

    暗黙確率 q_i = 1/オッズ に対して p_i ∝ q_i^gamma とし、レース内で合計1に正規化する。
    gamma = 1 で比例方式と一致する。gamma > 1 では人気馬が比例より厚く・人気薄が薄くなり
    (favourite-longshot bias の補正方向)、gamma < 1 ではその逆に、人気薄が持ち上がる。
    JRA ではこれは devig ではなく FLB の補正で、gamma はレースごとには推定せず、
    評価より前の期間で勝ち馬の対数尤度が最大になるよう1つ決める前提(決め方は devig_trials)。
    """
    if gamma <= 0:
        raise ValueError(f"gamma は正の値: {gamma}")
    inv = 1.0 / np.asarray(odds, dtype=float)
    w = inv ** gamma
    return w / w.sum()


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


def market_probs(df: pd.DataFrame, method: str = "proportional", gamma: float | None = None) -> pd.Series:
    """df(win_odds, race_id)の各行の市場確率。オッズのない馬・そろわないレースは NaN。

    method は 'proportional'(現行)か 'shin' か 'power'。レース単位で作るので、同じレースの馬は
    同じ方法・同じパラメータで扱う。'power' のときは gamma(期間で決めた指数)が要る。
    行の対応は位置で持つので、df の index が重複していても壊れない。
    """
    if method not in ("proportional", "shin", "power"):
        raise ValueError(f"method は proportional か shin か power: {method}")
    if method == "power":
        if gamma is None:
            raise ValueError("method が power のときは gamma(指数)が要ります")
        if gamma <= 0:
            raise ValueError(f"gamma は正の値: {gamma}")
    odds = pd.to_numeric(df["win_odds"], errors="coerce").to_numpy(dtype=float)
    race = df["race_id"].to_numpy()
    out = np.full(len(df), np.nan)
    pos = pd.Series(np.arange(len(df)))
    for _, idx in pos.groupby(race).groups.items():
        o = odds[idx]
        if not (o > 0).all():
            continue
        if method == "proportional":
            out[idx] = proportional(o)
        elif method == "shin":
            out[idx] = shin(o)[0]
        else:
            out[idx] = power(o, gamma)
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
