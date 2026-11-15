"""市場の勝率の作り方(比例方式 vs Shin)を、市場だけで比べる(DB は読むだけ)。

使い方:
    python -m keibaai.devig_trials --db data/keiba.db --test-from 2024-01-01 --test-to 2025-01-01

モデルの勝率は使わず、確定オッズから作った市場確率だけで、勝ち馬の対数尤度(1レース平均)と
全出走馬の LogLoss を比べる。差はレースを引き直した 95% 区間(ペア・ブートストラップ)つき。
評価期間は、判定に使い込んでいない年(2024年など)に固定すること。推定期間は要らない(どちらの方法も学習しない)。
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .devig import market_probs, market_z


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


def compare(df: pd.DataFrame, seed: int = 0, reps: int = 1000) -> pd.DataFrame:
    """方法ごとの成績と、Shin − 比例 のレース単位の差(勝ち馬の対数尤度)の 95% 区間。"""
    d = df.copy()
    d["p_prop"] = market_probs(d, "proportional")
    d["p_shin"] = market_probs(d, "shin")
    d = d.dropna(subset=["p_prop", "p_shin"])
    rows = []
    for name, col in (("比例", "p_prop"), ("Shin", "p_shin")):
        p = d[col].clip(1e-12, 1 - 1e-12)
        rows.append({"方法": name, "レース数": d["race_id"].nunique(),
                     "勝ち馬の対数尤度": float(np.log(p[d["win"] == 1]).mean()),
                     "LogLoss": float(-(d["win"] * np.log(p) + (1 - d["win"]) * np.log(1 - p)).mean())})
    win_rows = d[d["win"] == 1]
    diff = (np.log(win_rows["p_shin"].clip(1e-12)) - np.log(win_rows["p_prop"].clip(1e-12))).to_numpy()
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(reps, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append({"方法": "差 (Shin − 比例)", "レース数": len(diff),
                 "勝ち馬の対数尤度": float(diff.mean()), "LogLoss": np.nan,
                 "区間下限": float(lo), "区間上限": float(hi)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="市場の勝率の作り方(比例 vs Shin)を市場だけで比べる(DB は読むだけ)")
    ap.add_argument("--db", default="data/keiba.db")
    ap.add_argument("--test-from", default="2024-01-01")
    ap.add_argument("--test-to", default="2025-01-01")
    args = ap.parse_args(argv)
    conn = sqlite3.connect(f"file:{Path(args.db).as_posix()}?mode=ro", uri=True)
    df = load_market(conn, args.test_from, args.test_to)
    if df.empty:
        raise SystemExit("評価できるレースがありません(確定オッズと勝ち馬がそろったレースが要ります)")
    out = compare(df)
    z = market_z(df)
    print(f"評価 {args.test_from} 〜 {args.test_to}: {df['race_id'].nunique():,} レース")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"Shin の z: 平均 {z.mean():.3f}、推定できず比例方式に倒れたレース {(z.isna()).sum()} 件"
          "(この件数が多いときは、差の数字を割り引いて読むこと)")


if __name__ == "__main__":
    main()
