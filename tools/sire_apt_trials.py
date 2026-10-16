"""種牡馬・母父の得意不得意(実際 − 期待)を特徴量にして、能力のみ・オッズありのモデルを前後で比べる。

使い方: python sire_apt_trials.py <out_dir>
産駒の出走は jravan_races.db(2014〜2018)と keiba.db(2019〜)を合わせて数える(どちらも読み取りだけ)。
率はどれも、その日より前の出走だけ(sire_aptitude.prior_sums)。
"""
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from keibaai.feature_trials import TEST_PERIODS, market_top3, run, to_markdown
from keibaai.features import build_features, load_results
from keibaai.jvrace import course_of
from keibaai.sire_aptitude import add_sire_aptitude

out = Path(sys.argv[1])
KEIBA = "file:D:/KeibaAI/pr1-check/data/keiba.db?mode=ro"
JVR = "file:D:/KeibaAI/pr1-check/data/jravan_races.db?mode=ro"
t0 = time.time()
kc = sqlite3.connect(KEIBA, uri=True)
df = build_features(load_results(kc))
print(f"特徴量 {len(df):,} 行 {time.time() - t0:.0f}s", flush=True)

q_runs = """SELECT r.race_id, r.number, r.finish, r.win_odds, r.horse_id, c.date, c.venue, c.course_type, c.distance,
                   c.ground, {course} AS course, c.name AS race_name, c.pace_first3f, c.pace_last3f
            FROM results r JOIN races c USING (race_id) WHERE r.finish IS NOT NULL AND c.course_type IN ('芝', 'ダ')"""
new = pd.read_sql_query(q_runs.format(course="c.course"), kc)
jc = sqlite3.connect(JVR, uri=True)
old = pd.read_sql_query(q_runs.format(course="c.track_code"), jc)
old["course"] = [course_of(str(t), v) for t, v in zip(old["course"], old["venue"])]
old = old[old["date"] < new["date"].min()]
runs = pd.concat([old, new], ignore_index=True)
ped = pd.read_sql_query("SELECT horse_id, sire_id, damsire_id, sire_line FROM pedigree", kc)
runs = runs.merge(ped[["horse_id", "sire_id", "damsire_id"]], on="horse_id", how="left")
runs["date"] = pd.to_datetime(runs["date"])
runs["field_size"] = runs.groupby("race_id")["number"].transform("count")
print(f"産駒の出走 {len(runs):,}({runs['date'].min():%Y-%m}〜)・種牡馬あり {runs['sire_id'].notna().mean():.1%}", flush=True)

# 期待: 市場(オッズの写し方は評価期間より前だけで作る)と、頭数だけ
runs["q"] = market_top3(runs, TEST_PERIODS[0])
top3 = (runs["finish"] <= 3).astype(float)
runs["ex_mkt"] = top3 - runs["q"]
runs["ex_raw"] = top3 - np.minimum(3, runs["field_size"]) / runs["field_size"]
# 展開: 前半3F − 後半3F を、同じ芝ダ・距離の評価期間より前の平均・ばらつきで標準化
d = runs.drop_duplicates("race_id")[["race_id", "date", "course_type", "distance", "pace_first3f", "pace_last3f"]].copy()
d["pace"] = d["pace_first3f"] - d["pace_last3f"]
ref = d[d["date"] < pd.Timestamp(TEST_PERIODS[0])].groupby(["course_type", "distance"])["pace"].agg(["mean", "std"])
d = d.join(ref, on=["course_type", "distance"])
d["pace_z"] = (d["pace"] - d["mean"]) / d["std"]
runs = runs.merge(d[["race_id", "pace_z"]], on="race_id", how="left")

# 種牡馬の系統: その種牡馬を父に持つ馬の sire_line(いちばん多いもの)
line_of = ped.dropna(subset=["sire_line", "sire_id"]).groupby("sire_id")["sire_line"].agg(lambda s: s.mode().iloc[0]).to_dict()

feats = {}
for exn in ("ex_mkt", "ex_raw"):
    for k in (20, 50, 100):
        a = add_sire_aptitude(runs.drop(columns=[c for c in ("ex_mkt", "ex_raw") if c != exn]).rename(columns={exn: "ex"}),
                              k=k, line_of=line_of)
        cols = [c for c in a.columns if c.endswith("_apt") or c.endswith("_ex_shr")]
        tag = f"{'mkt' if exn == 'ex_mkt' else 'raw'}{k}"
        feats[tag] = a[["race_id", "number"] + cols].rename(columns={c: f"{c}_{tag}" for c in cols})
        print(f"  {tag} {time.time() - t0:.0f}s", flush=True)
for f in feats.values():
    df = df.merge(f, on=["race_id", "number"], how="left")
print(f"合わせた {len(df):,} 行 {time.time() - t0:.0f}s", flush=True)

GROUPS = {
    "全体の超過": ["sire_ex_shr", "dam_ex_shr"],
    "芝ダ×距離帯": ["sire_ctdist_apt", "dam_ctdist_apt"],
    "競馬場×芝ダ": ["sire_venuect_apt", "dam_venuect_apt"],
    "コース": ["sire_course_apt", "dam_course_apt"],
    "レース名": ["sire_racename_apt", "dam_racename_apt"],
    "芝の道悪": ["sire_turfwet_apt", "dam_turfwet_apt"],
    "ダートのスロー": ["sire_dslow_apt", "dam_dslow_apt"],
}
ALL = [c for g in GROUPS.values() for c in g]
variants = {"base": ([], [])}
for tag in feats:
    variants[f"すべて({tag})"] = ([f"{c}_{tag}" for c in ALL], [])
for name, cols in GROUPS.items():  # 条件ごとの効き目は k=50 で見る
    for e in ("mkt", "raw"):
        variants[f"{name}({e}50)"] = ([f"{c}_{e}50" for c in cols], [])
import os
if os.environ.get("SMOKE"):
    t = df[df["date"] >= pd.Timestamp(TEST_PERIODS[0])]
    print(t[[f"{c}_mkt50" for c in ALL]].describe().T[["count", "mean", "std"]].to_string())
    sys.exit(0)
res = run(df, variants, 3, 300, out, TEST_PERIODS)
text = "## 血統の得意不得意(種牡馬・母父 × 条件、実際 − 期待)\n" + to_markdown(res)
(out / "results.md").write_text(text, encoding="utf-8")
res.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
print(text)
print(f"{time.time() - t0:.0f}s")
