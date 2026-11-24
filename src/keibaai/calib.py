"""市場の勝率(比例方式で作った確率 q)を、過去データで学んだ補正で校正する部品。

比例方式(devig.proportional)は売上のシェアの復元としては正確だが、favourite-longshot bias
(人気馬は勝率の割に勝ち、人気薄はその逆)をそのまま確率に持ち込む。ここでは

    p_i ∝ exp(a・log q_i + c_band(i))     (レース内で合計1に正規化)

という少パラメータの形で校正する。a は温度(傾き)で、c_band は単勝オッズの帯ごとのオフセット。
帯オフセットには 0 への L2 正則化をかけ、勝ち数の少ない帯(大穴側)が暴れないようにする。
レース内で正規化するので、確率の合計1の制約は自然に満たされる。切片にあたる全馬共通の
定数は正規化で消えるため、パラメータは a と帯オフセット(5個)だけになる。

帯オフセットが全部 0 なら p_i ∝ q_i^a で、devig.power(オッズ, a) と一致する。つまり温度だけの
校正は Power 法そのもので、Power は仕様 8-27 の比較(devig_trials)で「比例と比べて採らない」と
判定済み。したがってこの校正で新しく効く可能性があるのは帯オフセットの側で、a はその土台として
一緒に推定する。校正が本当に効くかは calib_trials で年ウォークフォワードに確かめる(本番の経路は
まだ変えない)。

校正済み確率の使い道は、市場だけから確率を作る場面(ペーパートレードの確率、買い方での
市場とのブレンド、★の土台)に限る想定。残差モデルは init が市場の logit で、市場からのずれを
自分で学ぶので、生の市場確率のまま渡す(校正済みを渡すと二重補正になる)。今回は部品の提供までで、
適用先への配線はしない。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .devig import market_probs

BAND_EDGES = np.array([2.0, 5.0, 10.0, 30.0])  # 単勝オッズの帯の境目(この値以上が次の帯)
BAND_LABELS = ("〜2倍", "2〜5倍", "5〜10倍", "10〜30倍", "30倍〜")
DEFAULT_LAM = 10.0  # 帯オフセットの L2 正則化の強さ(0 への引き戻し)。勝ち馬が約1万頭の規模では弱めの引き戻し


def odds_band(odds: np.ndarray) -> np.ndarray:
    """単勝オッズ → 帯の番号(0〜4)。境目ちょうどのオッズは上の帯に入る。"""
    return np.searchsorted(BAND_EDGES, np.asarray(odds, dtype=float), side="right")


@dataclass
class Calibrator:
    """市場の勝率への校正(温度 a と帯オフセット)。fit_calibrator で作る。"""

    a: float
    band_offsets: np.ndarray = field(default_factory=lambda: np.zeros(len(BAND_LABELS)))
    lam: float = DEFAULT_LAM  # フィットに使った正則化の強さ(記録用)
    n_races: int = 0  # フィットに使ったレース数(記録用)

    def apply_race(self, q: np.ndarray, odds: np.ndarray) -> np.ndarray:
        """1レース分の市場確率 q を校正する。q は合計1、odds はその単勝オッズ。"""
        q = np.asarray(q, dtype=float)
        s = self.a * np.log(np.clip(q, 1e-300, None)) + self.band_offsets[odds_band(odds)]
        w = np.exp(s - s.max())
        return w / w.sum()


def fit_calibrator(df: pd.DataFrame, lam: float = DEFAULT_LAM) -> Calibrator:
    """df(race_id, win, win_odds。各レースの勝ち馬は1頭、オッズは全頭そろっていること)で校正を学ぶ。

    物差しは勝ち馬の対数尤度(レース単位の合計)から、帯オフセットの L2 罰則 lam・Σc² を引いたもの。
    罰則があることで、帯オフセットの共通成分(正規化で消える方向)が 0 に寄り、パラメータが定まる。
    ここは渡された範囲だけで学ぶので、評価期間の df を混ぜると漏れる。期間の切り分けは呼び出し側
    (calib_trials の年ウォークフォワード)の責任。
    """
    if df.empty:
        raise ValueError("校正を学ぶためのレースがありません")
    odds = pd.to_numeric(df["win_odds"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(odds).all() or (odds <= 0).any():
        raise ValueError("オッズに有限でない値・0以下が混ざっています")
    win_sum = df.groupby("race_id")["win"].transform("sum").to_numpy(dtype=float)
    if (win_sum != 1).any():
        raise ValueError("勝ち馬が1頭でないレースが混ざっています(load_market の形の df を渡してください)")
    q = market_probs(df, "proportional").to_numpy(dtype=float)
    if not np.isfinite(q).all():
        raise ValueError("市場確率を作れないレースが混ざっています")
    codes_np, _ = pd.factorize(df["race_id"].to_numpy())
    n_races = int(codes_np.max()) + 1
    logq = torch.tensor(np.log(np.clip(q, 1e-300, None)), dtype=torch.float64)
    band = torch.tensor(odds_band(odds), dtype=torch.long)
    codes = torch.tensor(codes_np, dtype=torch.long)
    win = torch.tensor(df["win"].to_numpy(dtype=float), dtype=torch.float64)
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    c = torch.zeros(len(BAND_LABELS), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, c], lr=0.5, max_iter=200,
                            tolerance_grad=1e-12, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        s = a * logq + c[band]
        top = s.max()  # exp の桁あふれ防止(レースごとの正規化には影響しない共通のずらし)
        denom = torch.zeros(n_races, dtype=torch.float64).index_add(0, codes, torch.exp(s - top))
        logp = s - top - torch.log(denom)[codes]
        loss = -(win * logp).sum() + lam * (c * c).sum()
        loss.backward()
        return loss

    opt.step(closure)
    a_val = float(a.detach())
    offsets = c.detach().numpy().copy()
    if not (np.isfinite(a_val) and np.isfinite(offsets).all()):
        raise ValueError("校正のフィットが収束しませんでした(パラメータが有限でない)")
    return Calibrator(a=a_val, band_offsets=offsets, lam=float(lam), n_races=n_races)


def calibrate_probs(df: pd.DataFrame, cal: Calibrator) -> pd.Series:
    """df(win_odds, race_id)の各行の校正済み市場確率。オッズのそろわないレースは NaN(market_probs と同じ扱い)。

    行の対応は位置で持つので、df の index が重複していても壊れない。
    """
    q = market_probs(df, "proportional")
    odds = pd.to_numeric(df["win_odds"], errors="coerce").to_numpy(dtype=float)
    race = df["race_id"].to_numpy()
    qv = q.to_numpy(dtype=float)
    out = np.full(len(df), np.nan)
    pos = pd.Series(np.arange(len(df)))
    for _, idx in pos.groupby(race).groups.items():
        if np.isfinite(qv[idx]).all():
            out[idx] = cal.apply_race(qv[idx], odds[idx])
    return pd.Series(out, index=df.index)
