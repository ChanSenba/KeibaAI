"""クッション値・含水率(当日の朝の測定)を足して、能力のみ・オッズありのモデルを前後で比べる。

使い方: python baba_trials.py <keiba.db> <jra_baba.db> <out_dir>
比べ方は keibaai.feature_trials.run と同じ(2025-01〜・2026-01〜、乱数3通りの平均、LogLoss の差の 95% 区間)。
"""
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from keibaai.feature_trials import TEST_PERIODS, run, to_markdown
from keibaai.features import build_features, load_results

keiba_db, baba_db, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
t0 = time.time()
df = build_features(load_results(sqlite3.connect(f"file:{keiba_db}?mode=ro", uri=True)))
print(f"特徴量 {len(df):,} 行 {time.time() - t0:.0f}s", flush=True)

baba = pd.read_sql_query("SELECT * FROM baba", sqlite3.connect(f"file:{baba_db}?mode=ro", uri=True))
baba["date"] = pd.to_datetime(baba["date"])
baba = baba.sort_values(["venue", "date"])
# 前の測定(同じ競馬場で、その日より前のいちばん新しい測定。ふつうは前日か金曜)との差
for k in ("turf_goal", "dirt_goal"):
    baba[f"{k}_prev"] = baba.groupby("venue")[k].shift(1)
    gap = (baba["date"] - baba.groupby("venue")["date"].shift(1)).dt.days
    baba.loc[gap > 4, f"{k}_prev"] = np.nan
baba["cushion_prev"] = baba.groupby("venue")["cushion"].shift(1)

venue = df["venue"].astype(str)
key = pd.DataFrame({"venue": venue, "date": pd.to_datetime(df["date"])})
m = key.merge(baba, on=["venue", "date"], how="left")  # 当日の朝の測定だけ(その日の分は 9:15 ごろ公開。1R より前)
m.index = df.index
turf = df["course_type"].astype(str) != "ダ"  # 障害は芝の値
df["baba_cushion"] = m["cushion"].where(turf)
df["baba_moist_goal"] = np.where(turf, m["turf_goal"], m["dirt_goal"])
df["baba_moist_4c"] = np.where(turf, m["turf_4c"], m["dirt_4c"])
df["baba_moist_chg"] = np.where(turf, m["turf_goal"] - m["turf_goal_prev"], m["dirt_goal"] - m["dirt_goal_prev"])
df["baba_cushion_chg"] = (m["cushion"] - m["cushion_prev"]).where(turf)
# 馬ごとの、その馬が過去に走ったときのクッション値・含水率との差(同じ芝ダで、直前の走り)
df = df.sort_values(["horse_id", "date"])
for c in ("baba_cushion", "baba_moist_goal"):
    prev = df.groupby(["horse_id", df["course_type"].astype(str)])[c].shift(1)
    df[f"{c}_vs_last"] = df[c] - prev
df = df.sort_index()

CUSHION = ["baba_cushion", "baba_cushion_chg"]
MOIST = ["baba_moist_goal", "baba_moist_4c", "baba_moist_chg"]
VS = ["baba_cushion_vs_last", "baba_moist_goal_vs_last"]
variants = {
    "base": ([], []),
    "含水率(当日朝・その馬場)": (MOIST, []),
    "クッション値(当日朝・芝)": (CUSHION, []),
    "含水率+クッション値": (MOIST + CUSHION, []),
    "含水率+クッション値+前走との差": (MOIST + CUSHION + VS, []),
}
for p in TEST_PERIODS:
    t = df[df["date"] >= pd.Timestamp(p)]
    races = t.drop_duplicates("race_id")
    print(f"{p}〜: {len(races):,} レース。含水率あり {races['baba_moist_goal'].notna().mean():.1%}、"
          f"芝でクッション値あり {races.loc[races['course_type'].astype(str) == '芝', 'baba_cushion'].notna().mean():.1%}", flush=True)
tr = df[df["date"] < pd.Timestamp(TEST_PERIODS[0])].drop_duplicates("race_id")
print(f"学習側(2025-01 より前): {len(tr):,} レース。含水率あり {tr['baba_moist_goal'].notna().mean():.1%}、"
      f"クッション値あり(芝) {tr.loc[tr['course_type'].astype(str) == '芝', 'baba_cushion'].notna().mean():.1%}", flush=True)

res = run(df, variants, 3, 300, out, TEST_PERIODS)
text = "## クッション値・含水率\n" + to_markdown(res)
(out / "results.md").write_text(text, encoding="utf-8")
res.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
print(text)
print(f"{time.time() - t0:.0f}s")
