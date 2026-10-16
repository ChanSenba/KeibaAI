"""全券種の買い方の過去検証。モデルの勝率から各券種の的中確率を作り、確定オッズで期待値を出し、払戻で回収率を出す。

使い方: python bet_backtest.py <out_dir>
- 勝率: 能力のみ・オッズありのモデル(feature_trials と同じ学習。2025 年は 2025-01-01 より前、2026 年は 2026-01-01 より前で学習)。
  比べる基準として、単勝オッズだけから出した勝率(市場)も使う。
- 的中確率: ハーヴィル(ヘネリー版、betting.harville_outcomes)で 1〜3 着の並びの確率を作って足す。
- 期待値 = 的中確率 × 確定オッズ(複勝・ワイドは下限)。回収率は払戻(同着も正しく入る)で数える。100 円ずつ。
- 区間: レースを引き直した(1,000 回)回収率の 95% 区間。
keiba.db・jravan_odds.db は読み取りだけ。
"""
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from keibaai.betting import GAMMA_2ND, GAMMA_3RD, harville_outcomes, place_slots
from keibaai.feature_trials import _fit_predict
from keibaai.features import ABILITY_MODEL_FEATURES, MARKET_MODEL_FEATURES, build_features, load_results

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
kc = sqlite3.connect("file:D:/KeibaAI/pr1-check/data/keiba.db?mode=ro", uri=True)
df = build_features(load_results(kc))
print(f"特徴量 {len(df):,} 行 {time.time() - t0:.0f}s", flush=True)

YEARS = {2025: "2025-01-01", 2026: "2026-01-01"}
preds = []
for year, cut in YEARS.items():
    c = pd.Timestamp(cut)
    train = df[df["date"] < c]
    test = df[(df["date"] >= c) & (df["date"] < pd.Timestamp(f"{year + 1}-01-01"))].copy()
    for name, feats in (("ability", ABILITY_MODEL_FEATURES), ("market", MARKET_MODEL_FEATURES)):
        test[f"p_{name}"] = np.mean([_fit_predict(train, test, feats, s, 300) for s in range(3)], axis=0)
    inv = 1 / pd.to_numeric(test["win_odds"], errors="coerce")
    test["p_odds"] = inv / inv.groupby(test["race_id"]).transform("sum")
    test["year"] = year
    preds.append(test[["race_id", "number", "bracket", "year", "p_ability", "p_market", "p_odds"]])
    print(f"{year} 予測 {test['race_id'].nunique():,} レース {time.time() - t0:.0f}s", flush=True)
pred = pd.concat(preds, ignore_index=True)
pred = pred[pred.groupby("race_id")["p_odds"].transform(lambda s: s.notna().all())]  # 取消などでオッズのない馬がいるレースは除く

# オッズと払戻(読み取りだけ)
oc = sqlite3.connect("file:D:/KeibaAI/pr1-check/data/jravan_odds.db?mode=ro", uri=True)
ids = sorted(pred["race_id"].unique())
oc.execute("CREATE TEMP TABLE ids (race_id TEXT PRIMARY KEY)")
oc.executemany("INSERT INTO ids VALUES (?)", [(i,) for i in ids])
TABLES = {"単勝": ("odds_win", 1, "odds"), "複勝": ("odds_place", 1, "odds_min"), "枠連": ("odds_bracket", 2, "odds"),
          "馬連": ("odds_quinella", 2, "odds"), "ワイド": ("odds_wide", 2, "odds_min"), "馬単": ("odds_exacta", 2, "odds"),
          "3連複": ("odds_trio", 3, "odds"), "3連単": ("odds_trifecta", 3, "odds")}
PAY_KIND = {"単勝": "win", "複勝": "place", "枠連": "bracket", "馬連": "quinella", "ワイド": "wide", "馬単": "exacta",
            "3連複": "trio", "3連単": "trifecta"}
odds = {}
for kind, (table, n, col) in TABLES.items():
    cols = ", ".join("abc"[:n])
    odds[kind] = pd.read_sql_query(f"SELECT o.race_id, {cols}, o.{col} AS odds FROM {table} o JOIN ids USING (race_id)", oc)
pay = pd.read_sql_query("SELECT p.race_id, p.kind, p.combo, p.payout FROM payouts p JOIN ids USING (race_id)", oc)
print(f"オッズ・払戻 {time.time() - t0:.0f}s", flush=True)


def race_probs(g: pd.DataFrame, pcol: str) -> dict:
    """1レースの各券種の的中確率 {券種: {組(馬番): 確率}}。"""
    nums = g["number"].to_numpy().astype(int)
    br = g["bracket"].to_numpy()
    p = g[pcol].to_numpy(float)
    p = p / p.sum()
    n = len(p)
    order, prob = harville_outcomes(p, GAMMA_2ND, GAMMA_3RD)
    a, b, c = order[:, 0], order[:, 1], order[:, 2]
    win = p
    k = place_slots(n)
    place = np.zeros(n)
    np.add.at(place, a, prob)
    np.add.at(place, b, prob)
    if k == 3:
        np.add.at(place, c, prob)
    ex = np.zeros((n, n))
    np.add.at(ex, (a, b), prob)
    qu = ex + ex.T
    wi = np.zeros((n, n))
    for x, y in ((a, b), (a, c), (b, c)):
        np.add.at(wi, (np.minimum(x, y), np.maximum(x, y)), prob)
    out = {"単勝": {(nums[i],): win[i] for i in range(n)},
           "複勝": {(nums[i],): place[i] for i in range(n)} if k else {},
           "馬連": {}, "ワイド": {}, "馬単": {}, "枠連": {}}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            out["馬単"][(nums[i], nums[j])] = ex[i, j]
            if i < j:
                lo, hi = sorted((nums[i], nums[j]))
                out["馬連"][(lo, hi)] = qu[i, j]
                out["ワイド"][(lo, hi)] = wi[min(i, j), max(i, j)]
                if not np.isnan(br[i]) and not np.isnan(br[j]):
                    bk = tuple(sorted((int(br[i]), int(br[j]))))
                    out["枠連"][bk] = out["枠連"].get(bk, 0.0) + qu[i, j]
    s = np.sort(order, axis=1)
    tri = pd.Series(prob).groupby([nums[s[:, 0]], nums[s[:, 1]], nums[s[:, 2]]]).sum()
    out["3連複"] = tri.to_dict()
    out["3連単"] = dict(zip(zip(nums[a], nums[b], nums[c]), prob))
    return out


def combo_text(kind: str, key: tuple) -> str:
    if kind == "枠連":
        return "-".join(str(x) for x in key)
    return "-".join(f"{x:02d}" for x in key)


rows = []
by_race = {rid: g for rid, g in pred.groupby("race_id", sort=False)}
year_of = pred.drop_duplicates("race_id").set_index("race_id")["year"]
odds_idx = {k: {rid: g for rid, g in v.groupby("race_id", sort=False)} for k, v in odds.items()}
pay_idx = {(r.race_id, r.kind, r.combo): r.payout for r in pay.itertuples()}
for m, (rid, g) in enumerate(by_race.items()):
    if len(g) < 5:
        continue
    for src in ("p_ability", "p_market", "p_odds"):
        probs = race_probs(g, src)
        for kind, table in probs.items():
            og = odds_idx[kind].get(rid)
            if og is None or not table:
                continue
            n = TABLES[kind][1]
            keys = list(zip(*[og[x].astype(int) for x in "abc"[:n]]))
            pr = np.array([table.get(kk, 0.0) for kk in keys])
            od = og["odds"].to_numpy(float)
            best = int(np.argmax(pr))
            payout = np.array([pay_idx.get((rid, PAY_KIND[kind], combo_text(kind, kk)), 0) for kk in keys])
            rows.append(pd.DataFrame({"race_id": rid, "year": year_of[rid], "src": src, "kind": kind, "prob": pr,
                                      "odds": od, "payout": payout, "best": np.arange(len(pr)) == best}))
    if m % 500 == 0:
        print(f"  {m:,}/{len(by_race):,} レース {time.time() - t0:.0f}s", flush=True)
bets = pd.concat(rows, ignore_index=True)
bets["ev"] = bets["prob"] * bets["odds"]
bets.to_parquet(out / "bets.parquet") if hasattr(bets, "to_parquet") else None
print(f"買い目の候補 {len(bets):,} {time.time() - t0:.0f}s", flush=True)

STRATS = [("本命(確率が最も高い1点)", lambda d: d["best"])] + [
    (f"期待値≥{e}・確率≥{p:g}", (lambda e, p: lambda d: (d["ev"] >= e) & (d["prob"] >= p))(e, p))
    for e in (1.0, 1.2, 1.5, 2.0) for p in (0.0, 0.02, 0.05)]


def summary(d: pd.DataFrame) -> dict:
    if d.empty:
        return {"races": 0, "bets": 0, "hit": np.nan, "roi": np.nan, "lo": np.nan, "hi": np.nan}
    r = d.groupby("race_id").agg(stake=("payout", "size"), ret=("payout", "sum"), hit=("payout", lambda s: (s > 0).any()))
    stake, ret = r["stake"].to_numpy() * 100, r["ret"].to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(r), (1000, len(r)))
    boot = ret[idx].sum(1) / stake[idx].sum(1) * 100
    return {"races": len(r), "bets": int(stake.sum() / 100), "hit": float((d["payout"] > 0).mean()),
            "roi": float(ret.sum() / stake.sum() * 100), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5))}


res = []
for (src, kind), g in bets.groupby(["src", "kind"]):
    for sname, f in STRATS:
        sel = g[f(g)]
        for year in (2025, 2026):
            res.append({"src": src, "kind": kind, "strategy": sname, "year": year, **summary(sel[sel["year"] == year])})
res = pd.DataFrame(res)
res.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
SRC = {"p_ability": "能力のみのモデル", "p_market": "オッズありのモデル", "p_odds": "市場(単勝オッズだけ)"}
lines = ["## 全券種の買い方の過去検証(100円ずつ・確定オッズで期待値・払戻で回収)"]
for src in SRC:
    for kind in TABLES:
        g = res[(res["src"] == src) & (res["kind"] == kind)]
        if g.empty:
            continue
        lines.append(f"\n### {SRC[src]} / {kind}\n")
        lines.append("| 買い方 | 2025 レース・点 | 的中率 | 回収率(95%区間) | 2026 レース・点 | 的中率 | 回収率(95%区間) |")
        lines.append("|---|---|---|---|---|---|---|")
        for sname, _ in STRATS:
            a = g[(g["strategy"] == sname) & (g["year"] == 2025)].iloc[0]
            b = g[(g["strategy"] == sname) & (g["year"] == 2026)].iloc[0]
            cell = lambda r: (f"{r['races']:,}・{r['bets']:,} | {r['hit']:.1%} | {r['roi']:.1f}%({r['lo']:.0f}〜{r['hi']:.0f})"
                              if r["races"] else "0 | - | -")
            lines.append(f"| {sname} | {cell(a)} | {cell(b)} |")
(out / "results.md").write_text("\n".join(lines), encoding="utf-8")
print(f"done {time.time() - t0:.0f}s", flush=True)
