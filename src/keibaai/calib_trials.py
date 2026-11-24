"""市場の勝率の校正(calib.py)を、年ウォークフォワードで市場だけで確かめる(DB は読むだけ)。

使い方:
    python -m keibaai.calib_trials --db data/keiba.db --test-from 2024-01-01 --test-to 2026-01-01

各年の校正は、その年より前のデータ(--fit-from 〜 その年の1月1日)だけで学び、評価年の結果は
学びにも選択にも使わない。年が進むごとに学ぶ範囲は広がる(拡大ウィンドウ)。評価期間は、
判定に使い込んでいない年に固定すること(devig_trials と同じ運用)。

主指標は勝ち馬の確率のレース平均 LogLoss(-log p_勝ち馬)の校正前後。差はレースを引き直した
95% 区間(ペア・ブートストラップ)つき。補助に、オッズ帯別の reliability(予測平均と実頻度の差)と、
Henery の着順確率での2着以内・3着以内の LogLoss を出す。Henery の γ(GAMMA_2ND・GAMMA_3RD)は
現行値のままで、校正に合わせた再フィットは今回の対象外(採用するなら別途確かめる)。

出力は Markdown で、docs/spec.md の第8章に貼れる形(結果の表と、事前に固定した採否基準つき)にする。
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .betting import GAMMA_2ND, GAMMA_3RD, harville_outcomes
from .calib import BAND_LABELS, DEFAULT_LAM, Calibrator, calibrate_probs, fit_calibrator, odds_band
from .devig import market_probs
from .devig_trials import load_market

RECENT_FROM = 2024  # 「直近年」の境目(採否基準2)
DIFF_TARGET = 0.0005  # 改善の目安(1レースあたりの勝ち馬LogLoss。採否基準4)


def _topk_probs(p: np.ndarray, k: int) -> np.ndarray:
    """勝率 p から、馬ごとの「k 着以内に入る確率」(Henery 版ハーヴィル、γ は現行値)。"""
    order, prob = harville_outcomes(p, GAMMA_2ND, GAMMA_3RD)
    return np.bincount(order[:, :k].ravel(), weights=np.repeat(prob, k), minlength=len(p))


def _bin_logloss_sum(p: np.ndarray, y: np.ndarray) -> float:
    """二値 LogLoss の合計(平均にしないのは、年をまたいで合算できるようにするため)。"""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def evaluate_year(fit_df: pd.DataFrame, test_df: pd.DataFrame, lam: float = DEFAULT_LAM) -> dict:
    """fit_df(評価年より前だけ)で校正を学び、test_df で前後の指標を測る。

    fit と test でレースが重なっていたら止める(重なりは期間の重なり、つまり漏れの兆候)。
    戻り値の win_ll_raw / win_ll_cal はレース単位の勝ち馬 LogLoss の配列(プール時の bootstrap 用)。
    reliability は帯ごとの (頭数, 予測の合計, 勝ち数) の生の合計で、年をまたいで合算できる形にする。
    """
    if not set(fit_df["race_id"]).isdisjoint(test_df["race_id"]):
        raise ValueError("校正を学ぶ期間と評価期間に同じレースが含まれています")
    cal = fit_calibrator(fit_df, lam=lam)
    d = test_df.copy()
    d["p_raw"] = market_probs(d, "proportional")
    d["p_cal"] = calibrate_probs(d, cal)
    d = d.dropna(subset=["p_raw", "p_cal"]).reset_index(drop=True)
    win = d["win"].to_numpy(dtype=float)
    res: dict = {"cal": cal, "n_races": int(d["race_id"].nunique()),
                 "win_ll_raw": -np.log(d.loc[d["win"] == 1, "p_raw"].clip(1e-12)).to_numpy(),
                 "win_ll_cal": -np.log(d.loc[d["win"] == 1, "p_cal"].clip(1e-12)).to_numpy()}
    band = odds_band(pd.to_numeric(d["win_odds"], errors="coerce").to_numpy(dtype=float))
    for name, col in (("raw", "p_raw"), ("cal", "p_cal")):
        p = d[col].to_numpy(dtype=float)
        res[f"rel_{name}"] = pd.DataFrame(
            {"頭数": np.bincount(band, minlength=len(BAND_LABELS)),
             "予測の合計": np.bincount(band, weights=p, minlength=len(BAND_LABELS)),
             "勝ち数": np.bincount(band, weights=win, minlength=len(BAND_LABELS))},
            index=pd.Index(BAND_LABELS, name="帯"))
    finish = pd.to_numeric(d["finish"], errors="coerce").to_numpy(dtype=float)
    sums = {k: {"raw": 0.0, "cal": 0.0} for k in (2, 3)}
    pos = pd.Series(np.arange(len(d)))
    praw = d["p_raw"].to_numpy(dtype=float)
    pcal = d["p_cal"].to_numpy(dtype=float)
    n_hen = 0
    for _, idx in pos.groupby(d["race_id"].to_numpy()).groups.items():
        idx = np.asarray(idx)
        fin = finish[idx]
        ok = np.isfinite(fin)  # 着順のある馬だけで着順確率を作る(中止・失格は着順がなく、対象に入れると合計がずれる)
        if ok.sum() < 3:
            continue
        n_hen += int(ok.sum())
        for k in (2, 3):
            y = (fin[ok] <= k).astype(float)
            for name, p in (("raw", praw), ("cal", pcal)):
                pk = p[idx][ok]
                sums[k][name] += _bin_logloss_sum(_topk_probs(pk / pk.sum(), k), y)
    res["n_horses"] = n_hen  # Henery の LogLoss を1頭あたりに直すための頭数(着順のある馬だけ)
    for k in (2, 3):
        res[f"hen{k}_raw_sum"] = sums[k]["raw"]
        res[f"hen{k}_cal_sum"] = sums[k]["cal"]
    return res


def _boot_ci(diff: np.ndarray, seed: int, reps: int) -> tuple[float, float]:
    """差の平均の、レースを引き直した 95% 区間。"""
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(reps, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _rel_table(rel_raw: pd.DataFrame, rel_cal: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """帯ごとの reliability の表と、頭数で重みづけた絶対ずれ(校正前・校正後)。"""
    n = rel_raw["頭数"].to_numpy(dtype=float)
    rate = rel_raw["勝ち数"].to_numpy(dtype=float) / np.maximum(n, 1)
    mean_raw = rel_raw["予測の合計"].to_numpy(dtype=float) / np.maximum(n, 1)
    mean_cal = rel_cal["予測の合計"].to_numpy(dtype=float) / np.maximum(n, 1)
    table = pd.DataFrame(
        {"帯": BAND_LABELS, "頭数": rel_raw["頭数"].to_numpy(),
         "校正前の予測平均": mean_raw, "実頻度": rate, "差(校正前)": mean_raw - rate,
         "校正後の予測平均": mean_cal, "差(校正後)": mean_cal - rate})
    total = n.sum()
    wabs_raw = float((n * np.abs(mean_raw - rate)).sum() / total) if total else float("nan")
    wabs_cal = float((n * np.abs(mean_cal - rate)).sum() / total) if total else float("nan")
    return table, wabs_raw, wabs_cal


def walk_forward(df: pd.DataFrame, test_from: str, test_to: str,
                 lam: float = DEFAULT_LAM, seed: int = 0, reps: int = 1000) -> dict:
    """df(fit 開始〜評価終わりを通しで読んだ load_market の形)で年ウォークフォワードに評価する。

    年 Y の評価は、Y の1月1日より前の行だけで校正を学ぶ。評価の切り口は暦年で、
    test_from / test_to が年の途中なら、その年の切り口は期間内に丸める。
    """
    try:
        d_from, d_to = dt.date.fromisoformat(test_from), dt.date.fromisoformat(test_to)
    except ValueError as e:
        raise ValueError(f"日付は YYYY-MM-DD で指定してください: {e}") from e
    if d_from >= d_to:
        raise ValueError(f"評価期間が空です: {test_from} 〜 {test_to}")
    date = df["date"].astype(str)
    years: list[dict] = []
    rel_raw = rel_cal = None
    ll_raw_all: list[np.ndarray] = []
    ll_cal_all: list[np.ndarray] = []
    hen = {k: {"raw": 0.0, "cal": 0.0} for k in (2, 3)}
    n_horses = 0
    for year in range(d_from.year, d_to.year + 1):
        start = max(test_from, f"{year}-01-01")
        end = min(test_to, f"{year + 1}-01-01")
        if start >= end:
            continue
        test_df = df[(date >= start) & (date < end)]
        if test_df.empty:
            continue
        fit_df = df[date < f"{year}-01-01"]
        r = evaluate_year(fit_df, test_df, lam=lam)
        cal: Calibrator = r["cal"]
        years.append({"年": year, "レース数": r["n_races"], "a": cal.a, "offsets": cal.band_offsets,
                      "勝ち馬LogLoss(校正前)": float(r["win_ll_raw"].mean()),
                      "勝ち馬LogLoss(校正後)": float(r["win_ll_cal"].mean()),
                      "差": float(r["win_ll_cal"].mean() - r["win_ll_raw"].mean())})
        rel_raw = r["rel_raw"] if rel_raw is None else rel_raw + r["rel_raw"]
        rel_cal = r["rel_cal"] if rel_cal is None else rel_cal + r["rel_cal"]
        ll_raw_all.append(r["win_ll_raw"])
        ll_cal_all.append(r["win_ll_cal"])
        n_horses += r["n_horses"]
        for k in (2, 3):
            hen[k]["raw"] += r[f"hen{k}_raw_sum"]
            hen[k]["cal"] += r[f"hen{k}_cal_sum"]
    if not years:
        raise ValueError("評価できる年がありません(評価期間に確定オッズと勝ち馬がそろったレースが要ります)")
    diff = np.concatenate(ll_cal_all) - np.concatenate(ll_raw_all)
    lo, hi = _boot_ci(diff, seed, reps)
    table, wabs_raw, wabs_cal = _rel_table(rel_raw, rel_cal)
    band_gain = (table["差(校正後)"].abs() - table["差(校正前)"].abs()).to_numpy()
    return {"years": years, "pooled_diff": float(diff.mean()), "ci": (lo, hi),
            "n_races": int(sum(y["レース数"] for y in years)),
            "n_improved": int(sum(1 for y in years if y["差"] < 0)),
            "recent": [(y["年"], y["差"]) for y in years if y["年"] >= RECENT_FROM],
            "rel": table, "wabs_raw": wabs_raw, "wabs_cal": wabs_cal,
            "band_worst": (str(table["帯"].iloc[int(np.argmax(band_gain))]), float(band_gain.max())),
            "hen": {k: (hen[k]["raw"] / n_horses, hen[k]["cal"] / n_horses) for k in (2, 3)}}


def markdown_report(res: dict, *, test_from: str, test_to: str, fit_from: str, lam: float) -> str:
    """docs/spec.md の第8章に貼れる形の Markdown(結果の表と採否判定の下書き)。見出しの番号は貼るときに振る。"""
    years = res["years"]
    lo, hi = res["ci"]
    lines = [
        "### 8-XX 市場の勝率の校正(b1: 温度とオッズ帯オフセット)",
        "",
        f"評価は {test_from} 〜 {test_to} の {res['n_races']:,} レース(確定オッズ、モデルの勝率は使わない)。"
        f"校正 p_i ∝ exp(a・log q_i + c_帯) は年ウォークフォワードで、各年の分を {fit_from} 〜 その年の1月1日より前"
        "のデータだけで学んだ(評価年の結果は学びに使っていない)。帯オフセットの L2 正則化は "
        f"lam={lam:g}。Henery の γ(GAMMA_2ND・GAMMA_3RD)は現行値のままで、校正に合わせた再フィットは"
        "今回の対象外(採用するなら別途確かめる)。",
        "",
        "| 年 | レース数 | a | 勝ち馬LogLoss 校正前 | 校正後 | 差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for y in years:
        lines.append(f"| {y['年']} | {y['レース数']:,} | {y['a']:.3f} | {y['勝ち馬LogLoss(校正前)']:.5f} "
                     f"| {y['勝ち馬LogLoss(校正後)']:.5f} | {y['差']:+.5f} |")
    lines += [
        "",
        f"プールした差(校正後 − 校正前、勝ち馬LogLoss、1レースあたり): {res['pooled_diff']:+.5f}"
        f" (レース単位 bootstrap の95%区間 {lo:+.5f} 〜 {hi:+.5f})。負が改善。",
        "",
        "オッズ帯別の reliability(予測平均と実頻度の差。0に近いほどよい):",
        "",
        "| 帯 | 頭数 | 校正前の予測平均 | 実頻度 | 差(校正前) | 校正後の予測平均 | 差(校正後) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in res["rel"].iterrows():
        lines.append(f"| {r['帯']} | {int(r['頭数']):,} | {r['校正前の予測平均']:.4f} | {r['実頻度']:.4f} "
                     f"| {r['差(校正前)']:+.4f} | {r['校正後の予測平均']:.4f} | {r['差(校正後)']:+.4f} |")
    lines += [
        "",
        f"帯の頭数で重みづけた絶対ずれ: 校正前 {res['wabs_raw']:.4f} → 校正後 {res['wabs_cal']:.4f}。",
        "",
        "Henery の着順確率(市場の勝率から作る)の LogLoss(1頭あたり。γ は現行値のまま):",
        "",
        "| 対象 | 校正前 | 校正後 |",
        "|---|---:|---:|",
        f"| 2着以内 | {res['hen'][2][0]:.5f} | {res['hen'][2][1]:.5f} |",
        f"| 3着以内 | {res['hen'][3][0]:.5f} | {res['hen'][3][1]:.5f} |",
        "",
        "フィットした帯オフセット c(0が無補正。プラスはその帯を市場より厚く見る):",
        "",
        "| 年 | a | " + " | ".join(BAND_LABELS) + " |",
        "|---|---:|" + "---:|" * (1 + len(BAND_LABELS)),
    ]
    for y in years:
        offs = " | ".join(f"{v:+.3f}" for v in y["offsets"])
        lines.append(f"| {y['年']} | {y['a']:.3f} | {offs} |")
    recent_txt = "、".join(f"{yr}年 {d:+.5f}" for yr, d in res["recent"]) or "対象年なし"
    worst_band, worst_gain = res["band_worst"]
    lines += [
        "",
        "採否基準(結果を見る前に固定したもの)と、今回の測定値:",
        "",
        f"1. 対象年の7割以上で勝ち馬LogLossが改善し、プールした差の95%区間が0を含まない"
        f" → 改善した年 {res['n_improved']}/{len(years)} 年、区間 {lo:+.5f} 〜 {hi:+.5f}",
        f"2. 直近年({RECENT_FROM}年〜)で悪化していない → {recent_txt}",
        f"3. どの帯でも reliability の絶対ずれが悪化していない → 一番悪化した帯は {worst_band} で {worst_gain:+.4f}"
        "(プラスが悪化)",
        f"4. 改善の目安はレースあたり約{DIFF_TARGET} → プールした差 {res['pooled_diff']:+.5f}",
        "",
        "採用可否: (結果を見て記入)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="市場の勝率の校正(温度とオッズ帯オフセット)を年ウォークフォワードで比べる(DB は読むだけ)")
    ap.add_argument("--db", default="data/keiba.db")
    ap.add_argument("--test-from", default="2024-01-01")
    ap.add_argument("--test-to", default="2025-01-01")
    ap.add_argument("--fit-from", default="2019-01-01", help="校正を学ぶ期間の開始(最初の評価年より前までさかのぼる)")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAM, help="帯オフセットの L2 正則化の強さ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args(argv)
    try:
        d_fit, d_test = dt.date.fromisoformat(args.fit_from), dt.date.fromisoformat(args.test_from)
        dt.date.fromisoformat(args.test_to)
    except ValueError as e:
        raise SystemExit(f"日付は YYYY-MM-DD で指定してください: {e}") from e
    if d_fit >= d_test:
        raise SystemExit(f"校正を学ぶ期間が評価期間にかかっています: 開始 {args.fit_from} >= 評価開始 {args.test_from}")
    conn = sqlite3.connect(f"file:{Path(args.db).as_posix()}?mode=ro", uri=True)
    df = load_market(conn, args.fit_from, args.test_to)
    if df.empty:
        raise SystemExit("評価できるレースがありません(確定オッズと勝ち馬がそろったレースが要ります)")
    res = walk_forward(df, args.test_from, args.test_to, lam=args.lam, seed=args.seed, reps=args.reps)
    print(markdown_report(res, test_from=args.test_from, test_to=args.test_to,
                          fit_from=args.fit_from, lam=args.lam))


if __name__ == "__main__":
    main()
