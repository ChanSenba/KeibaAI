"""ほかの券種のオッズ(複勝・ワイド・馬連・馬単・三連複・三連単)に、単勝オッズにない勝ちやすさの情報があるかを確かめる (2026-11-21)。

PC で、D:\\KeibaAI\\pr1-check の .venv の Python で実行する(元の DB には書かない。keiba.db は写しを作って使う):
    .venv\\Scripts\\python.exe ..\\pool-trial-1121\\pools.py --repo D:\\KeibaAI\\pr1-check --out D:\\KeibaAI\\pool-trial-1121

入力
- <repo>\\data\\keiba.db: 成績・特徴量(写しを <out>\\data\\keiba.db に作り、model_cache の results・other_runs・jv_training・nnyear も写して使い回す)
- <repo>\\data\\jravan_odds.db: 全券種の確定オッズ(読むだけ)

比べるもの(勝ち馬の対数尤度、1000レースあたり、95%区間は開催日単位で引き直し)
  M  単勝オッズだけ(今の市場の勝率)
  B  単勝と各券種から出した勝率を、対数の重みで混ぜる(重みは学習期間で決める)
  R  今の★・期待値のモデル(単勝の市場を出発点にした LightGBM)
  RP R に、各券種から出した見立てと単勝との差を特徴量として足す
  RB RP の出発点を B にする
期間: 開発用(2024年までで学習、2025-01〜を評価)、確認用(2022年までで学習、2023年を評価)。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=r"D:\KeibaAI\pr1-check")
ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--no-trifecta", action="store_true", help="三連単を読まない(重いとき)")
args = ap.parse_args()
REPO, OUT = Path(args.repo), Path(args.out)
sys.path.insert(0, str(REPO / "src"))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- 1. DB の写しと特徴量
DATA = OUT / "data"
DB_COPY = DATA / "keiba.db"
FEATS = DATA / "feats_pools.pkl"
if not DB_COPY.exists():
    DATA.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{REPO / 'data' / 'keiba.db'}?mode=ro", uri=True)
    dst = sqlite3.connect(DB_COPY)
    src.backup(dst)
    dst.close()
    src.close()
    (DATA / "model_cache").mkdir(exist_ok=True)
    for f in (REPO / "data" / "model_cache").glob("*.pkl"):
        if f.name.startswith(("results-", "other_runs-", "jv_training-", "nnyear-")):
            shutil.copy2(f, DATA / "model_cache" / f.name)
    log("DB を写した")

from keibaai import nn_year  # noqa: E402
from keibaai.features import CATEGORICAL, features_from_db, result_mask  # noqa: E402
from keibaai.model import RESIDUAL_PARAMS, RESIDUAL_ROUNDS, TRAIN_FROM, has_market, train_rows  # noqa: E402

nn_year.use_db_path(DB_COPY)

POOLS = ["place", "wide", "quinella", "exacta1", "exacta2", "trio", "tri1"]
if args.no_trifecta:
    POOLS.remove("tri1")


def pool_shares(odds_db: Path) -> pd.DataFrame:
    """馬ごとの、各券種のオッズの逆数の合計(レース内で合計1にする前)。"""
    c = sqlite3.connect(f"file:{odds_db}?mode=ro", uri=True)
    lo = "'2018'"
    q = {
        # 複勝・ワイドは幅の下限と上限の幾何平均
        "place": f"SELECT race_id, a AS number, 1.0/sqrt(odds_min*odds_max) AS s FROM odds_place WHERE race_id >= {lo} AND odds_min > 0 AND odds_max > 0",
        "wide": f"""SELECT race_id, number, SUM(s) AS s FROM (
                      SELECT race_id, a AS number, 1.0/sqrt(odds_min*odds_max) AS s FROM odds_wide WHERE race_id >= {lo} AND odds_min > 0 AND odds_max > 0
                      UNION ALL SELECT race_id, b, 1.0/sqrt(odds_min*odds_max) FROM odds_wide WHERE race_id >= {lo} AND odds_min > 0 AND odds_max > 0)
                    GROUP BY race_id, number""",
        "quinella": f"""SELECT race_id, number, SUM(s) AS s FROM (
                      SELECT race_id, a AS number, 1.0/odds AS s FROM odds_quinella WHERE race_id >= {lo} AND odds > 0
                      UNION ALL SELECT race_id, b, 1.0/odds FROM odds_quinella WHERE race_id >= {lo} AND odds > 0)
                    GROUP BY race_id, number""",
        "exacta1": f"SELECT race_id, a AS number, SUM(1.0/odds) AS s FROM odds_exacta WHERE race_id >= {lo} AND odds > 0 GROUP BY race_id, a",
        "exacta2": f"SELECT race_id, b AS number, SUM(1.0/odds) AS s FROM odds_exacta WHERE race_id >= {lo} AND odds > 0 GROUP BY race_id, b",
        "trio": f"""SELECT race_id, number, SUM(s) AS s FROM (
                      SELECT race_id, a AS number, 1.0/odds AS s FROM odds_trio WHERE race_id >= {lo} AND odds > 0
                      UNION ALL SELECT race_id, b, 1.0/odds FROM odds_trio WHERE race_id >= {lo} AND odds > 0
                      UNION ALL SELECT race_id, c, 1.0/odds FROM odds_trio WHERE race_id >= {lo} AND odds > 0)
                    GROUP BY race_id, number""",
        "tri1": f"SELECT race_id, a AS number, SUM(1.0/odds) AS s FROM odds_trifecta WHERE race_id >= {lo} AND odds > 0 GROUP BY race_id, a",
    }
    out = None
    for name in POOLS:
        t = time.time()
        d = pd.read_sql_query(q[name], c).rename(columns={"s": f"s_{name}"})
        d["race_id"] = d["race_id"].astype(str)
        d["number"] = d["number"].astype(int)
        log(f"  {name}: {len(d):,} 行 {d.race_id.nunique():,} レース ({time.time() - t:.0f}s)")
        out = d if out is None else out.merge(d, on=["race_id", "number"], how="outer")
    c.close()
    return out


if FEATS.exists():
    df = pd.read_pickle(FEATS)
    log(f"特徴量を読んだ {len(df):,} 行")
else:
    conn = sqlite3.connect(DB_COPY)
    df = features_from_db(conn)
    df = df[result_mask(df)].copy()
    log(f"特徴量 {len(df):,} 行")
    # 年ごとのネットは、keibaai train・予想と同じく 2019年より前の行も入れた表で作る(保存の指紋と過去走の塔が同じになる)
    df = nn_year.add_nn_year(df)
    log("年ごとのネットの予想を足した")
    df = df[df["date"] >= pd.Timestamp(TRAIN_FROM)].copy()
    sh = pool_shares(REPO / "data" / "jravan_odds.db")
    df["race_id"] = df["race_id"].astype(str)
    df["number"] = pd.to_numeric(df["number"]).astype(int)
    df = df.merge(sh, on=["race_id", "number"], how="left")
    df.to_pickle(FEATS)
    log("特徴量を保存した")

# 1着がちょうど1頭で、全馬に単勝オッズがあるレースだけ
df = df[has_market(df)]
df = df[df.groupby("race_id")["win"].transform("sum") == 1].copy()
odds = pd.to_numeric(df["win_odds"], errors="coerce")
inv = 1 / odds
df["p_mkt"] = inv / inv.groupby(df["race_id"]).transform("sum")
df["lp_mkt"] = np.log(df["p_mkt"])
for name in POOLS:
    s = df[f"s_{name}"]
    ok = s.notna().groupby(df["race_id"]).transform("all") & (s > 0).groupby(df["race_id"]).transform("all")
    share = s / s.groupby(df["race_id"]).transform("sum")
    df[f"lp_{name}"] = np.where(ok, np.log(share.clip(lower=1e-9)), np.nan)
    df[f"d_{name}"] = df[f"lp_{name}"] - df["lp_mkt"]  # 単勝の市場との対数の差(★のモデルの特徴量)
full = np.ones(len(df), bool)
for name in POOLS:
    full &= df[f"lp_{name}"].notna().groupby(df["race_id"]).transform("all").to_numpy()
log(f"全券種のそろうレース {df.loc[full, 'race_id'].nunique():,} / {df.race_id.nunique():,}"
    f"(年ごと: {df[full].drop_duplicates('race_id')['date'].dt.year.value_counts().sort_index().to_dict()})")
df = df[full].sort_values(["race_id", "number"]).reset_index(drop=True)
POOL_FEATS = [f"d_{n}" for n in POOLS]


# ---------------------------------------------------------------- 2. 評価の道具
def winner_lp(d: pd.DataFrame, logit: np.ndarray) -> pd.Series:
    """レースごとの勝ち馬の対数確率(logit をレース内のソフトマックスにする)。"""
    z = pd.Series(logit, index=d.index)
    z = z - z.groupby(d["race_id"]).transform("max")
    lse = np.log(np.exp(z).groupby(d["race_id"]).transform("sum"))
    lp = z - lse
    return lp[d["win"].to_numpy() == 1].groupby(d.loc[d["win"] == 1, "race_id"]).first()


def diff_ci(a: pd.Series, b: pd.Series, day: pd.Series, n: int = 2000) -> str:
    """a − b(1000レースあたり)と、開催日単位で引き直した95%区間。"""
    x = (a - b)
    g = x.groupby(day.reindex(x.index)).agg(["sum", "count"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    boot = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1) * 1000
    return f"{x.mean() * 1000:+.1f}({np.percentile(boot, 2.5):+.1f}〜{np.percentile(boot, 97.5):+.1f})"


def fit_blend(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """勝率 ∝ exp(Σ w_k·log p_k) の重みを、学習のレースの勝ち馬の対数尤度で決める。"""
    from scipy.optimize import minimize

    X = d[cols].to_numpy()
    rid = d["race_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1]
    win = d["win"].to_numpy() == 1

    def nll(w):
        z = X @ w
        m = np.maximum.reduceat(z, starts)
        zz = z - np.repeat(m, np.diff(np.r_[starts, len(z)]))
        lse = np.log(np.add.reduceat(np.exp(zz), starts))
        grad_p = np.exp(zz - np.repeat(lse, np.diff(np.r_[starts, len(z)])))
        f = -(zz[win].sum() - lse.sum())
        g = -(X[win].sum(0) - (grad_p[:, None] * X).sum(0))
        return f / len(starts), g / len(starts)

    w0 = np.zeros(len(cols))
    w0[0] = 1.0
    return minimize(nll, w0, jac=True, method="L-BFGS-B").x


def fit_res(train: pd.DataFrame, feats: list[str], init: np.ndarray, seed: int):
    import lightgbm as lgb

    cats = [c for c in CATEGORICAL if c in feats]
    ds = lgb.Dataset(train[feats], train["win"], categorical_feature=cats, init_score=init, free_raw_data=False)
    return lgb.train(dict(RESIDUAL_PARAMS, seed=seed), ds, num_boost_round=RESIDUAL_ROUNDS)


def res_logit(train, test, feats, init_col, seeds):
    """★のモデルの、テストの行の logit(乱数 seeds 通りの平均)。init_col は出発点の対数勝率の列。"""
    tr = train_rows(train)
    def to_logit(lp):
        p = np.exp(lp).clip(1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    zs = []
    for s in range(seeds):
        m = fit_res(tr, feats, to_logit(tr[init_col].to_numpy()), s)
        zs.append(m.predict(test[feats], raw_score=True) + to_logit(test[init_col].to_numpy()))
    z = np.mean(zs, axis=0)
    return z, m


# ---------------------------------------------------------------- 3. 実行
from keibaai.model import residual_features  # noqa: E402

BASE = residual_features()
BLEND_COLS = ["lp_mkt"] + [f"lp_{n}" for n in POOLS]
SPLITS = {"開発用(2025-01〜)": ("2025-01-01", None), "確認用(2023年)": ("2023-01-01", "2024-01-01")}
rows = []
for label, (cut, end) in SPLITS.items():
    cut = pd.Timestamp(cut)
    train = df[df["date"] < cut].copy()
    test = df[(df["date"] >= cut) & ((df["date"] < pd.Timestamp(end)) if end else True)].copy()
    day = test.drop_duplicates("race_id").set_index("race_id")["date"]
    log(f"=== {label}: 学習 {train.race_id.nunique():,} レース、評価 {test.race_id.nunique():,} レース")
    res = {"M": winner_lp(test, test["lp_mkt"].to_numpy())}
    # 単勝 + 各券種を1つずつ、と全部
    for n in POOLS:
        w = fit_blend(train, ["lp_mkt", f"lp_{n}"])
        res[f"B:{n}"] = winner_lp(test, test[["lp_mkt", f"lp_{n}"]].to_numpy() @ w)
        log(f"  B 単勝+{n}: 重み {np.round(w, 3).tolist()}  市場との差 {diff_ci(res[f'B:{n}'], res['M'], day)}")
    w = fit_blend(train, BLEND_COLS)
    log(f"  B 全部: 重み {dict(zip(BLEND_COLS, np.round(w, 3)))}")
    for d in (train, test):
        z = d[BLEND_COLS].to_numpy() @ w
        zz = pd.Series(z, index=d.index)
        zz = zz - zz.groupby(d["race_id"]).transform("max")
        d["lp_blend"] = zz - np.log(np.exp(zz).groupby(d["race_id"]).transform("sum"))
    res["B"] = winner_lp(test, test["lp_blend"].to_numpy())
    log(f"  B 全部: 市場との差 {diff_ci(res['B'], res['M'], day)}")
    for key, feats, init in (("R", BASE, "lp_mkt"), ("RP", BASE + POOL_FEATS, "lp_mkt"), ("RB", BASE + POOL_FEATS, "lp_blend")):
        z, model = res_logit(train, test, feats, init, args.seeds)
        test[f"z_{key}"] = z
        res[key] = winner_lp(test, z)
        log(f"  {key}: 市場との差 {diff_ci(res[key], res['M'], day)}")
        if key == "RP":
            imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
            log("  RP の特徴量の重要度(上位15、券種の列の順位): "
                + ", ".join(f"{k}={v:.0f}" for k, v in imp.head(15).items())
                + " / " + ", ".join(f"{c}:{list(imp.index).index(c) + 1}位" for c in POOL_FEATS))
    for a, b in (("RP", "R"), ("RB", "R"), ("RB", "RP"), ("B", "M")):
        s = diff_ci(res[a], res[b], day)
        yrs = " ".join(f"{y}年 {((res[a] - res[b])[day.dt.year == y]).mean() * 1000:+.1f}"
                       for y in sorted(day.dt.year.unique()))
        log(f"  {a} − {b}: {s}  [{yrs}]")
        rows.append(dict(split=label, cmp=f"{a}-{b}", diff=s, years=yrs))
    # ★の単勝回収率(確定オッズ、期待値≧1.1・勝率≧3%・30倍以下)
    for key in ("R", "RP", "RB"):
        z = pd.Series(test[f"z_{key}"].to_numpy(), index=test.index)
        z = z - z.groupby(test["race_id"]).transform("max")
        p = np.exp(z) / np.exp(z).groupby(test["race_id"]).transform("sum")
        o = pd.to_numeric(test["win_odds"])
        sel = (p * o >= 1.1) & (p >= 0.03) & (o <= 30)
        pay = (test["win"] * o)[sel]
        log(f"  ★ {key}: {sel.sum():,} 頭 的中 {test['win'][sel].mean() * 100:.1f}% 回収率 {pay.mean() * 100:.1f}%")
pd.DataFrame(rows).to_csv(OUT / "result_pools.csv", index=False, encoding="utf-8-sig")
log("おわり")
