"""市場の勝率の作り方(比例方式・Shin・Power)を、市場だけで比べる(DB は読むだけ)。

使い方:
    python -m keibaai.devig_trials --db data/keiba.db --test-from 2024-01-01 --test-to 2025-01-01

モデルの勝率は使わず、確定オッズから作った市場確率だけで、勝ち馬の対数尤度(1レース平均)と
全出走馬の LogLoss を比べる。差はレースを引き直した 95% 区間(ペア・ブートストラップ)つき。
評価期間は、判定に使い込んでいない年(2024年など)に固定すること。

Power の指数 gamma は、評価期間より前の期間(--fit-from 〜 --fit-to、既定は 2019-01-01 〜 評価開始の前)
で勝ち馬の対数尤度(比較の主指標と同じ物差し)が最大になるよう選ぶ(仕様 8-25 と同じ、前の期間で選び
後の期間で比べる形)。
評価期間の結果は gamma の選択に一切使わない。Shin の z はレースごとに推定するので選択期間は要らない。

JRA の単勝はパリミュチュエルなので、比例方式はシェアの復元として正確で、Shin と Power は
控除の剥がし方というより favourite-longshot bias の補正として比べる。

出力は Markdown で、docs/spec.md の第8章に貼れる形(結果の表と文章の下書き)にする。
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .devig import market_probs, market_z

GAMMA_GRID = np.round(np.arange(0.90, 1.2001, 0.01), 2)  # gamma の候補(1 が比例方式。FLB 補正は 1 の少し上を想定)


def load_market(conn: sqlite3.Connection, date_from: str, date_to: str) -> pd.DataFrame:
    """確定オッズが全頭そろい、勝ち馬が1頭いるレースだけの (race_id, date, number, win, win_odds)。"""
    df = pd.read_sql_query(
        "SELECT r.race_id, c.date, r.number, r.finish, r.win_odds FROM results r JOIN races c USING (race_id) "
        "WHERE c.date >= ? AND c.date < ? AND (r.finish IS NOT NULL OR r.status IN ('中', '失'))",
        conn, params=(date_from, date_to))
    df["win"] = (pd.to_numeric(df["finish"], errors="coerce") == 1).astype(int)
    odds = pd.to_numeric(df["win_odds"], errors="coerce")
    ok = df.groupby("race_id")["win"].transform("sum") == 1
    full = (odds > 0).groupby(df["race_id"]).transform("all")  # 1頭でもオッズが欠けたら、そのレースは使わない
    many = df.groupby("race_id")["number"].transform("count") >= 3  # 頭数が少ないレースは確率の作り方を比べる対象にしない
    return df[ok & full & many].reset_index(drop=True)


def resolve_fit_to(fit_from: str, fit_to: str | None, test_from: str) -> str:
    """gamma の選択期間の終わり(その日は含まない)を決め、評価期間にかかっていないことを確かめる。"""
    end = fit_to or test_from
    try:
        d_from, d_end, d_test = dt.date.fromisoformat(fit_from), dt.date.fromisoformat(end), dt.date.fromisoformat(test_from)
    except ValueError as e:
        raise ValueError(f"日付は YYYY-MM-DD で指定してください: {e}") from e
    if d_end > d_test:
        raise ValueError(f"gamma の選択期間が評価期間にかかっています: 終わり {end} > 評価開始 {test_from}")
    if d_from >= d_end:
        raise ValueError(f"gamma の選択期間が空です: {fit_from} 〜 {end}")
    return end


def fit_power_gamma(df: pd.DataFrame, grid: np.ndarray = GAMMA_GRID) -> float:
    """渡された df(評価より前の期間だけ)の中で、勝ち馬の対数尤度(レース平均)が最大になる Power の gamma を選ぶ。

    選ぶ物差しは compare が報告する主指標(勝ち馬の対数尤度)とそろえる。別の物差し(全馬の二値 LogLoss 等)
    で選ぶと、Power だけ違う物差しで選んだことになり、比較がぶれる。
    評価期間の df は渡さないこと。ここは渡された範囲だけで選ぶので、評価期間を混ぜると漏れる。
    df は load_market の形(race_id, win, win_odds。オッズは全頭そろっていること)。
    """
    if df.empty:
        raise ValueError("gamma を選ぶためのレースがありません")
    odds = pd.to_numeric(df["win_odds"], errors="coerce").to_numpy(dtype=float)
    win = df["win"].to_numpy(dtype=float)
    if not np.isfinite(odds).all() or (odds <= 0).any():
        raise ValueError("オッズに有限でない値・0以下が混ざっています")
    codes, _ = pd.factorize(df["race_id"].to_numpy())
    inv = 1.0 / odds
    best_gamma, best_score = float(grid[0]), -np.inf
    for g in grid:
        w = inv ** g
        p = w / np.bincount(codes, weights=w)[codes]
        score = float(np.log(np.clip(p[win == 1], 1e-12, None)).mean())
        if score > best_score:
            best_gamma, best_score = float(g), score
    if best_gamma in (float(grid[0]), float(grid[-1])):
        warnings.warn(f"gamma が候補の端 {best_gamma:.2f} に張り付きました。真の最適は範囲の外かもしれません",
                      stacklevel=2)
    return best_gamma


def _winner_loglik(d: pd.DataFrame, col: str) -> np.ndarray:
    """レースの並び順の、勝ち馬の市場確率の対数(差のブートストラップ用)。"""
    return np.log(d.loc[d["win"] == 1, col].clip(1e-12)).to_numpy()


def _diff_row(name: str, base: np.ndarray, other: np.ndarray, seed: int, reps: int,
              base_label: str = "比例") -> dict:
    """base との差の平均と、レースを引き直した 95% 区間。"""
    diff = other - base
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(reps, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"方法": f"差 ({name} − {base_label})", "レース数": len(diff),
            "勝ち馬の対数尤度": float(diff.mean()), "LogLoss": np.nan,
            "区間下限": float(lo), "区間上限": float(hi)}


def compare(df: pd.DataFrame, *, gamma: float | None = None, seed: int = 0, reps: int = 1000) -> pd.DataFrame:
    """方法ごとの成績と、比例との差(勝ち馬の対数尤度)の 95% 区間。gamma を渡すと Power も比べる。"""
    d = df.copy()
    d["p_prop"] = market_probs(d, "proportional")
    d["p_shin"] = market_probs(d, "shin")
    cols = [("比例", "p_prop", None), ("Shin", "p_shin", "Shin")]
    if gamma is not None:
        d["p_power"] = market_probs(d, "power", gamma=gamma)
        cols.append((f"Power (γ={gamma:.2f})", "p_power", "Power"))
    d = d.dropna(subset=[c for _, c, _ in cols])
    rows = []
    for name, col, _ in cols:
        p = d[col].clip(1e-12, 1 - 1e-12)
        rows.append({"方法": name, "レース数": d["race_id"].nunique(),
                     "勝ち馬の対数尤度": float(np.log(p[d["win"] == 1]).mean()),
                     "LogLoss": float(-(d["win"] * np.log(p) + (1 - d["win"]) * np.log(1 - p)).mean())})
    base = _winner_loglik(d, "p_prop")
    for _, col, label in cols[1:]:
        rows.append(_diff_row(label, base, _winner_loglik(d, col), seed, reps))
    if gamma is not None:  # 採用判断で要るのは Shin との直接の差も同じなので、そろえて出す
        rows.append(_diff_row("Power", _winner_loglik(d, "p_shin"), _winner_loglik(d, "p_power"),
                              seed, reps, base_label="Shin"))
    return pd.DataFrame(rows)


def evaluate(fit_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0, reps: int = 1000) -> tuple[float, pd.DataFrame]:
    """gamma を fit_df(評価より前)だけで選び、test_df で3方式を比べる。(選んだ gamma, 比較表)を返す。

    fit と test でレースが重なっていたら止める(重なりは期間の重なり、つまり漏れの兆候)。
    """
    if not set(fit_df["race_id"]).isdisjoint(test_df["race_id"]):
        raise ValueError("gamma の選択期間と評価期間に同じレースが含まれています")
    gamma = fit_power_gamma(fit_df)
    return gamma, compare(test_df, gamma=gamma, seed=seed, reps=reps)


def _per1000(row: pd.Series) -> str:
    """差の行を、仕様の記録の形式(1000レースあたり、区間つき)にする。"""
    return (f"{row['勝ち馬の対数尤度'] * 1000:+.1f}"
            f" ({row['区間下限'] * 1000:+.1f} 〜 {row['区間上限'] * 1000:+.1f})")


def markdown_report(out: pd.DataFrame, *, test_from: str, test_to: str, fit_from: str, fit_to: str,
                    gamma: float, n_races: int, z_mean: float, z_failed: int, gamma_note: str = "") -> str:
    """docs/spec.md の第8章に貼れる形の Markdown(結果の表と文章の下書き)。見出しの番号は貼るときに振る。"""
    lines = [
        "### 8-XX 市場の勝率の作り方(比例・Shin・Power)の比較",
        "",
        f"評価は {test_from} 〜 {test_to} の {n_races:,} レース(確定オッズ、モデルの勝率は使わない)。"
        f"Power の γ は評価より前の {fit_from} 〜 {fit_to} で勝ち馬の対数尤度が最大になる {gamma:.2f} を選んだ。",
    ]
    if gamma_note:
        lines.append(gamma_note)
    lines += [
        "",
        "| 方法 | レース数 | 勝ち馬の対数尤度 | LogLoss |",
        "|---|---:|---:|---:|",
    ]
    for _, r in out.iterrows():
        if pd.notna(r["LogLoss"]):
            lines.append(f"| {r['方法']} | {int(r['レース数']):,} | {r['勝ち馬の対数尤度']:.5f} | {r['LogLoss']:.5f} |")
    lines += ["", "比例との差(勝ち馬の対数尤度、1000レースあたり、レース単位 bootstrap の95%区間):", ""]
    diffs = {}
    for _, r in out.iterrows():
        if pd.isna(r["LogLoss"]):
            diffs[r["方法"]] = _per1000(r)
            lines.append(f"- {r['方法']}: {_per1000(r)}")
    lines += [
        "",
        f"Shin の z は平均 {z_mean:.3f}。推定できず比例方式に倒れたレースは {z_failed} 件"
        "(この件数が多いときは、Shin の差の数字を割り引いて読むこと)。",
        "",
        "文章の下書き:",
        "",
        f"市場の勝率の作り方として比例・Shin・Power を比べた。Power の γ は {gamma:.2f}"
        f"(選択期間 {fit_from} 〜 {fit_to})。評価期間 {test_from} 〜 {test_to} の {n_races:,} レースで、"
        f"比例との差(勝ち馬の対数尤度、1000レースあたり)は Shin が {diffs.get('差 (Shin − 比例)', '—')}、"
        f"Power が {diffs.get('差 (Power − 比例)', '—')} だった。",
        "",
        "採用可否: (結果を見て記入)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="市場の勝率の作り方(比例・Shin・Power)を市場だけで比べる(DB は読むだけ)")
    ap.add_argument("--db", default="data/keiba.db")
    ap.add_argument("--test-from", default="2024-01-01")
    ap.add_argument("--test-to", default="2025-01-01")
    ap.add_argument("--fit-from", default="2019-01-01")
    ap.add_argument("--fit-to", default=None, help="gamma の選択期間の終わり(その日は含まない)。既定は評価の開始日")
    args = ap.parse_args(argv)
    try:
        fit_to = resolve_fit_to(args.fit_from, args.fit_to, args.test_from)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    conn = sqlite3.connect(f"file:{Path(args.db).as_posix()}?mode=ro", uri=True)
    fit = load_market(conn, args.fit_from, fit_to)
    if fit.empty:
        raise SystemExit("gamma を選ぶためのレースがありません(選択期間に確定オッズと勝ち馬がそろったレースが要ります)")
    df = load_market(conn, args.test_from, args.test_to)
    if df.empty:
        raise SystemExit("評価できるレースがありません(確定オッズと勝ち馬がそろったレースが要ります)")
    gamma, out = evaluate(fit, df)
    z = market_z(df)
    gamma_note = ""
    if gamma in (float(GAMMA_GRID[0]), float(GAMMA_GRID[-1])):
        gamma_note = (f"注意: γ が候補範囲({GAMMA_GRID[0]:.2f}〜{GAMMA_GRID[-1]:.2f})の端 {gamma:.2f} に張り付いた。"
                      "真の最適は範囲の外かもしれないので、範囲を広げて選び直してから記録すること。")
    print(markdown_report(out, test_from=args.test_from, test_to=args.test_to,
                          fit_from=args.fit_from, fit_to=fit_to, gamma=gamma,
                          n_races=int(df["race_id"].nunique()),
                          z_mean=float(z.mean()), z_failed=int(z.isna().sum()),
                          gamma_note=gamma_note))


if __name__ == "__main__":
    main()
