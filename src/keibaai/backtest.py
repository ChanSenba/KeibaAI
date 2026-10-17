"""過去のレースに買い方のルールを当てはめて、単勝を買い続けた場合の成績を再現する。

学習は評価開始日より前だけ、買うのは評価開始日以後のレースだけ。1 点 stake 円の単勝で数える。
オッズには確定オッズを使うので、実際に買う時点のオッズとはずれる(人気薄ほど買った後に下がりやすい)。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .model import evaluate

RULES = {
    "ev": "期待値が基準以上の馬を全部買う",
    "top1": "モデルの勝率1位(◎)を買う",
    "favorite": "1番人気を買う(比較用)",
    "all": "全頭を買う(比較用。控除率ぶん負ける目安)",
}


SHORT = {"favorite": "1番人気", "top1": "◎(勝率1位)", "all": "全頭(比較用)", "ev": "期待値"}


@dataclass
class Summary:
    rule: str
    n_bets: int
    n_hits: int
    hit_rate: float
    stake: int
    payout: int
    profit: int
    roi: float  # 回収率(%)
    roi_low: float  # 回収率の95%区間(買い目の再抽出による)
    roi_high: float
    max_drawdown: int  # 収支の最大の落ち込み(円)


def with_ev(test: pd.DataFrame) -> pd.DataFrame:
    """期待値(勝率×単勝オッズ)の列を足す。オッズのない馬は買えないので除く。"""
    t = test.dropna(subset=["win_odds"]).copy()
    t["ev"] = t["p_win"] * t["win_odds"]
    return t


def select_bets(test: pd.DataFrame, rule: str, ev_min: float = 1.0, max_odds: float | None = None,
                min_p: float = 0.0) -> pd.DataFrame:
    t = with_ev(test)
    if rule == "ev":
        bets = t[t["ev"] >= ev_min]
    elif rule == "top1":
        bets = t.loc[t.groupby("race_id")["p_win"].idxmax()]
    elif rule == "favorite":
        bets = t[t["popularity"] == 1]
    elif rule == "all":
        bets = t
    else:
        raise ValueError(f"知らないルールです: {rule}")
    if max_odds is not None:
        bets = bets[bets["win_odds"] <= max_odds]
    if min_p > 0:
        bets = bets[bets["p_win"] >= min_p]
    return bets.sort_values(["date", "race_id", "number"]).reset_index(drop=True)


def settle(bets: pd.DataFrame, stake: int = 100) -> pd.DataFrame:
    """払戻と、買い始めからの収支(balance)を足す。"""
    b = bets.copy()
    b["stake"] = stake
    b["payout"] = (b["win"] * b["win_odds"] * stake).round().astype(int)
    b["balance"] = (b["payout"] - b["stake"]).cumsum()
    return b


def max_drawdown(balance: pd.Series) -> int:
    if balance.empty:
        return 0
    path = np.concatenate([[0], balance.to_numpy()])
    return int((np.maximum.accumulate(path) - path).max())


def roi_interval(settled: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """買い目を引き直して回収率を何度も計算し、中央の95%を返す。当たりが少ないほど幅は広い。"""
    if len(settled) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pay = settled["payout"].to_numpy(dtype=float)
    idx = rng.integers(0, len(pay), size=(n_boot, len(pay)))
    rois = pay[idx].sum(axis=1) / (settled["stake"].iloc[0] * len(pay)) * 100
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(rule: str, settled: pd.DataFrame) -> Summary:
    n = len(settled)
    stake, payout = int(settled["stake"].sum()), int(settled["payout"].sum())
    lo, hi = roi_interval(settled)
    hits = int((settled["win"] == 1).sum()) if n else 0
    return Summary(rule, n, hits, hits / n if n else float("nan"), stake, payout, payout - stake,
                   payout / stake * 100 if stake else float("nan"), lo, hi,
                   max_drawdown(settled["balance"]) if n else 0)


def balance_by_date(settled: pd.DataFrame) -> pd.Series:
    """開催日ごとの収支の推移(その日が終わった時点)。"""
    if settled.empty:
        return pd.Series(dtype=float)
    return settled.groupby("date")["balance"].last()


def predict_test(df: pd.DataFrame, test_from: str, use_market: bool = False) -> pd.DataFrame:
    """評価期間の各馬に勝率(p_win)を付ける。use_market=True はオッズを入力に含めたモデル。"""
    _, test = evaluate(df, test_from, use_market=use_market)
    return test


def run_rules(test: pd.DataFrame, rules: list[tuple[str, dict]], stake: int = 100) -> tuple[pd.DataFrame, dict]:
    """rules は (ルール名, 引数) の並び。まとめの表と、ルールごとの精算済み買い目を返す。"""
    rows, settled = [], {}
    for rule, kw in rules:
        label = f"期待値≧{kw['ev_min']:g}" if rule == "ev" else SHORT[rule]
        if kw.get("max_odds"):
            label += f"[〜{kw['max_odds']:g}倍]"
        if kw.get("min_p"):
            label += f"[勝率{kw['min_p']:.0%}〜]"
        s = settle(select_bets(test, rule, **kw), stake)
        settled[label] = s
        rows.append(asdict(summarize(label, s)))
    return pd.DataFrame(rows), settled


DEFAULT_RULES = [("favorite", {}), ("top1", {}), ("ev", {"ev_min": 1.0}), ("ev", {"ev_min": 1.2}),
                 ("ev", {"ev_min": 1.5}), ("all", {})]

# ★(期待値の印)の基準を決めるための組み合わせ: 期待値 × 勝率の下限 × オッズの上限
GRID_RULES = [("favorite", {}), ("all", {})] + [
    ("ev", {"ev_min": e, "min_p": p, "max_odds": o})
    for e in (1.0, 1.1, 1.2, 1.5) for p in (0.0, 0.03, 0.05) for o in (None, 50.0)]

COLUMNS = {
    "rule": "ルール", "n_bets": "買い目数", "n_hits": "的中", "hit_rate": "的中率", "stake": "投資(円)",
    "payout": "払戻(円)", "profit": "収支(円)", "roi": "回収率(%)", "roi_low": "回収率 下限", "roi_high": "回収率 上限",
    "max_drawdown": "最大の落ち込み(円)",
}


def format_summary(table: pd.DataFrame) -> str:
    lines = [f"{'ルール':<34}{'買い目':>7}{'的中率':>8}{'収支(円)':>11}{'回収率':>8}{'95%区間':>16}{'最大落ち込み':>12}"]
    for r in table.itertuples():
        lines.append(f"{r.rule:<34}{r.n_bets:>7}{r.hit_rate:>8.1%}{r.profit:>11,}{r.roi:>7.1f}%"
                     f"{f'{r.roi_low:.0f}〜{r.roi_high:.0f}%':>16}{r.max_drawdown:>12,}")
    return "\n".join(lines)


def blend_summary(test: pd.DataFrame, blend, use_market: bool = False) -> str:
    """混ぜた確率と、市場だけの確率の、評価期間での成績を比べる文。"""
    from .model import auc, logloss

    y = test["win"].to_numpy()
    model_name = "モデルだけ(オッズ込)" if use_market else "モデルだけ(能力)"
    rows = [("市場だけ(オッズ)", test["p_mkt"]), (model_name, test["p_model"]), ("混ぜた確率", test["p_win"])]
    lines = [f"混ぜ方: モデルの重み a={blend.a:g}, 市場の重み b={blend.b:g}"
             f"(決めた期間の対数尤度 {blend.loglik:.4f}、市場だけなら {blend.baseline_loglik:.4f})",
             f"{'評価期間の確率':<18}{'勝LogLoss':>11}{'勝AUC':>8}"]
    for name, p in rows:
        lines.append(f"{name:<{18 - sum(ord(c) > 255 for c in name)}}{logloss(y, p.to_numpy()):>11.4f}{auc(y, p.to_numpy()):>8.3f}")
    return "\n".join(lines)
