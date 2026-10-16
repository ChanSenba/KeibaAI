"""種牡馬・母父ごとの得意不得意が本当にあるか(偶然でないか)を、年を奇数・偶数に分けて確かめる。

各出走の超過成績 = 3着以内(0/1) − 市場の見込み(単勝オッズから出した3着以内の確率)。
得意の度合い = その条件での超過成績の平均 − その種牡馬のほかの条件での超過成績の平均。
奇数年と偶数年で別々に出し、種牡馬をまたいだ相関を見る(0 なら偶然、正なら本当にある)。
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

from keibaai.feature_trials import market_top3

db = sys.argv[1]
MIN_N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
df = pd.read_sql_query("""
    SELECT r.race_id, r.number, r.finish, r.win_odds, r.horse_id, c.date, c.venue, c.course_type, c.distance, c.ground,
           c.course, c.name AS race_name, c.pace_first3f, c.pace_last3f, p.sire_id, p.sire, p.damsire_id, p.damsire
    FROM results r JOIN races c USING (race_id) LEFT JOIN pedigree p USING (horse_id)
    WHERE r.finish IS NOT NULL AND c.course_type IN ('芝', 'ダ')""", c)
df["date"] = pd.to_datetime(df["date"])
df["field_size"] = df.groupby("race_id")["number"].transform("count")
df["q"] = market_top3(df, "2099-01-01")
df = df[df["q"].notna()].copy()
import os
EXPECT = os.environ.get("EXPECT", "market")
base = df["q"] if EXPECT == "market" else np.minimum(3, df["field_size"]) / df["field_size"]
df["ex"] = (df["finish"] <= 3).astype(float) - base
print("期待:", "市場(単勝オッズ)" if EXPECT == "market" else "頭数だけ(3/頭数。強さを差し引かない)")
df["half"] = np.where(df["date"].dt.year % 2 == 1, "odd", "even")
print(f"出走 {len(df):,}・{df['date'].min():%Y-%m}〜{df['date'].max():%Y-%m}・平均の超過 {df['ex'].mean():+.4f}", flush=True)

df["wet"] = np.where(df["ground"] == "良", "良", "道悪")
df["dist_band"] = pd.cut(df["distance"], [0, 1300, 1600, 1900, 2200, 9999],
                         labels=["〜1300", "1400-1600", "1700-1900", "2000-2200", "2300〜"]).astype(str)
df["course_key"] = (df["venue"] + df["course_type"] + df["distance"].astype(str) + df["course"].fillna(""))
# 展開: 前半3F − 後半3F を、同じ芝ダ・距離の中で比べる。前半が速い(差が小さい)ほどハイペース
d = df.drop_duplicates("race_id").copy()
d["pace"] = d["pace_first3f"] - d["pace_last3f"]
d["pace_z"] = d.groupby(["course_type", "distance"])["pace"].transform(lambda s: (s - s.mean()) / s.std())
df = df.merge(d[["race_id", "pace_z"]], on="race_id", how="left")
df["pace_band"] = pd.cut(df["pace_z"], [-99, -0.5, 0.5, 99], labels=["ハイ", "平均", "スロー"]).astype(str)


def aptitude(g: pd.DataFrame, by: str, level, parent: str) -> pd.DataFrame:
    """parent(種牡馬など)ごとに、by == level の超過の平均 − それ以外の超過の平均。半分ずつ。"""
    rows = []
    for half, h in g.groupby("half"):
        on = h[h[by] == level].groupby(parent)["ex"].agg(["mean", "count"])
        off = h[h[by] != level].groupby(parent)["ex"].agg(["mean", "count"])
        t = on.join(off, lsuffix="_on", rsuffix="_off", how="inner")
        t = t[(t["count_on"] >= MIN_N) & (t["count_off"] >= MIN_N)]
        t["apt"] = t["mean_on"] - t["mean_off"]
        rows.append(t["apt"].rename(half))
    return pd.concat(rows, axis=1).dropna()


def cell_aptitude(g: pd.DataFrame, by: str, parent: str) -> pd.DataFrame:
    """parent × by の各組で、その組の超過の平均 − parent 全体の超過の平均。半分ずつ。"""
    rows = []
    for half, h in g.groupby("half"):
        all_ = h.groupby(parent)["ex"].mean()
        cell = h.groupby([parent, by])["ex"].agg(["mean", "count"])
        cell = cell[cell["count"] >= MIN_N]
        apt = cell["mean"] - all_.reindex(cell.index.get_level_values(0)).to_numpy()
        rows.append(apt.rename(half))
    return pd.concat(rows, axis=1).dropna()


def report(name: str, t: pd.DataFrame) -> None:
    if len(t) < 5:
        print(f"{name}: 対象 {len(t)}(少なすぎる)")
        return
    r = np.corrcoef(t["odd"], t["even"])[0, 1]
    rs = t["odd"].rank().corr(t["even"].rank())
    se = 1 / np.sqrt(len(t) - 3)
    print(f"{name}: 対象 {len(t)}、奇数年と偶数年の相関 {r:+.3f}(順位 {rs:+.3f})、偶然なら ±{1.96 * se:.3f} に入る", flush=True)


print(f"\n(条件ごとに、各半分で産駒 {MIN_N} 走以上の組だけ)")
for parent in ("sire_id", "damsire_id"):
    lab = "種牡馬" if parent == "sire_id" else "母父"
    print(f"\n== {lab}")
    for half, h in df.groupby("half"):
        pass
    overall = pd.concat([h.groupby(parent)["ex"].agg(["mean", "count"]).query("count >= @MIN_N")["mean"].rename(half)
                         for half, h in df.groupby("half")], axis=1).dropna()
    report(f"{lab}の全体の超過成績(市場より走るか)", overall)
    for ct in ("芝", "ダ"):
        g = df[df["course_type"] == ct]
        report(f"{lab} {ct}: 道悪の得意(道悪 − 良)", aptitude(g, "wet", "道悪", parent))
        report(f"{lab} {ct}: ハイペースの得意(ハイ − それ以外)", aptitude(g, "pace_band", "ハイ", parent))
        report(f"{lab} {ct}: スローの得意(スロー − それ以外)", aptitude(g, "pace_band", "スロー", parent))
    report(f"{lab}: 芝ダの得意(芝 − ダ)", aptitude(df, "course_type", "芝", parent))
    report(f"{lab}: 距離帯の得意(各距離帯 − 全体)", cell_aptitude(df, "dist_band", parent))
    report(f"{lab}: 競馬場×芝ダの得意", cell_aptitude(df.assign(vc=df["venue"] + df["course_type"]), "vc", parent))
    report(f"{lab}: コース(競馬場×芝ダ×距離×内外)の得意", cell_aptitude(df, "course_key", parent))
    report(f"{lab}: 同じレース名の得意", cell_aptitude(df[df["race_name"].notna()], "race_name", parent))

# 道悪巧者・良巧者(全期間、芝ダ別、産駒の道悪・良それぞれ100走以上)
names = df.drop_duplicates("sire_id").set_index("sire_id")["sire"]
for ct in ("芝", "ダ"):
    g = df[df["course_type"] == ct]
    t = g.groupby(["sire_id", "wet"])["ex"].agg(["mean", "count"]).unstack("wet")
    t = t[(t[("count", "道悪")] >= 100) & (t[("count", "良")] >= 100)]
    apt = (t[("mean", "道悪")] - t[("mean", "良")]).rename("apt")
    out = pd.DataFrame({"種牡馬": names.reindex(apt.index), "道悪−良(3着内の超過)": apt,
                        "道悪の走数": t[("count", "道悪")], "良の走数": t[("count", "良")],
                        "道悪の超過": t[("mean", "道悪")], "良の超過": t[("mean", "良")]})
    print(f"\n== {ct}の道悪巧者(上位10)\n" + out.sort_values("道悪−良(3着内の超過)", ascending=False).head(10).to_string(
        index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"\n== {ct}の良巧者(道悪が苦手、上位10)\n" + out.sort_values("道悪−良(3着内の超過)").head(10).to_string(
        index=False, float_format=lambda x: f"{x:+.3f}"))
