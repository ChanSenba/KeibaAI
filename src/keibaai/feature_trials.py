"""特徴量の候補を作り、足す前後で予想の当たりやすさを比べる(検証用。予想のモデルには入れない)。

使い方(DB は読むだけ。書き込まない):
    python -m keibaai.feature_trials --db data/keiba.db

候補は2種類。
- トラックバイアスの残り: 同じ日の先に終わったレースの偏り、市場(単勝オッズ)の見込みとの差で強い馬の分を
  差し引いた偏り、コース(内回り・外回り、A〜E、距離帯)ごとの偏り(件数が少ないときは競馬場・芝ダ全体に寄せる)。
- 相性: 厩舎×騎手の率にサンプルの少なさの割り引き((該当数 + k×全体の率) ÷ (回数 + k))、勝率・連対率の版、
  騎手×種牡馬、調教師×種牡馬。

どの候補も、そのレースより前に分かっている結果だけから作る(同じ日のものは、先に終わったレースだけ)。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import ABILITY_MODEL_FEATURES, CATEGORICAL, MARKET_MODEL_FEATURES, _expanding_rate
from .model import PARAMS, auc, logloss, normalize_within_race

TEST_PERIODS = ["2025-01-01", "2026-01-01"]
TJ_KS = [5, 10, 20, 50]
COURSE_K = 30  # コースごとの率を競馬場・芝ダ全体の率に寄せる強さ(頭数)
BIAS_WINDOW = "10D"


# ---------------------------------------------------------------- 市場の見込み(強い馬の差し引きに使う)

def market_top3(df: pd.DataFrame, fit_before: str) -> pd.Series:
    """単勝オッズから見た3着以内の見込み。fit_before より前のレースで、オッズ → 3着以内率の対応を作る。

    単勝の支持率(1/オッズ をレース内で合計1に)を、頭数の帯ごとに、過去の実際の3着以内率へ写す。
    レース内の合計が3(頭数が3未満ならその数)になるよう整え、1を超えないようにする。
    """
    odds = pd.to_numeric(df["win_odds"], errors="coerce").where(lambda s: s > 0)
    inv = 1.0 / odds
    p = inv / inv.groupby(df["race_id"]).transform("sum")
    lp = np.log(p)
    fb = pd.cut(df["field_size"], [0, 10, 14, 99], labels=False)
    fit = (df["date"] < pd.Timestamp(fit_before)) & lp.notna() & df["finish"].notna()
    q = pd.Series(np.nan, index=df.index)
    for b in range(3):
        m_fit = fit & (fb == b)
        if m_fit.sum() < 1000:
            continue
        bins = pd.qcut(lp[m_fit], 40, duplicates="drop")
        tab = pd.DataFrame({"x": lp[m_fit], "y": (df.loc[m_fit, "finish"] <= 3).astype(float), "b": bins})
        g = tab.groupby("b", observed=True).agg(x=("x", "mean"), y=("y", "mean")).sort_values("x")
        y = np.maximum.accumulate(g["y"].to_numpy())  # 支持率が高いほど率は下がらない
        m = (fb == b) & lp.notna()
        q[m] = np.interp(lp[m], g["x"].to_numpy(), y)
    target = df.groupby("race_id")["number"].transform("count").clip(upper=3)
    q = q * target / q.groupby(df["race_id"]).transform("sum")
    return q.clip(upper=1.0)


# ---------------------------------------------------------------- トラックバイアス

def _race_sums(df: pd.DataFrame, value: pd.Series) -> pd.DataFrame:
    """レースごとに、前に付けた馬・ほかの馬・内の馬・外の馬の value の合計と頭数。"""
    done = df["finish"].notna() & value.notna()
    pos = df["number"] / df["field_size"]
    front = df["early_pos"] <= 1 / 3
    groups = {"f": done & df["early_pos"].notna() & front, "r": done & df["early_pos"].notna() & ~front,
              "i": done & (pos <= 1 / 3), "o": done & (pos >= 2 / 3)}
    out = pd.DataFrame({"race_id": df["race_id"]})
    for k, m in groups.items():
        out[f"{k}_s"] = value.where(m, 0.0)
        out[f"{k}_n"] = m.astype(float)
    return out.groupby("race_id", sort=False).sum()


def _diff(s: pd.DataFrame, prior: pd.DataFrame | None = None, k: float = 0.0) -> pd.DataFrame:
    """合計と頭数から、前−ほか と 内−外 の差。prior を渡すと、各組の率を prior の率に k 頭ぶん寄せる。"""
    def rate(c: str) -> pd.Series:
        n = s[f"{c}_n"]
        if prior is None:
            return s[f"{c}_s"] / n.where(n > 0)
        pr = prior[f"{c}_s"] / prior[f"{c}_n"].where(prior[f"{c}_n"] > 0)
        return (s[f"{c}_s"] + k * pr) / (n + k)
    return pd.DataFrame({"front": rate("f") - rate("r"), "inner": rate("i") - rate("o")}, index=s.index)


SUM_COLS = [f"{k}_{x}" for k in "frio" for x in "sn"]


def _today(info: pd.DataFrame, sums: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """同じ日・同じ keys の、そのレースより前の R のレースの合計(そのレース自身は含めない)。"""
    t = info.join(sums).sort_values(keys + ["date", "round"])
    cum = t.groupby(keys + ["date"], dropna=False)[SUM_COLS].cumsum() - t[SUM_COLS]
    cum["n_races"] = t.groupby(keys + ["date"], dropna=False).cumcount()
    return cum


def _prev_days(info: pd.DataFrame, sums: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """同じ keys の、直近 BIAS_WINDOW 以内の前の開催日の合計(当日は含めない)。"""
    day = info.join(sums).groupby(keys + ["date"], dropna=False)[SUM_COLS].sum().reset_index()
    out = []
    for _, g in day.sort_values("date").groupby(keys, dropna=False):
        r = g.set_index("date")[SUM_COLS].rolling(BIAS_WINDOW, closed="left", min_periods=1).sum()
        r = r.reset_index()
        for c in keys:
            r[c] = g[c].iloc[0]
        out.append(r)
    day = pd.concat(out, ignore_index=True)
    res = info.reset_index().merge(day, on=keys + ["date"], how="left").set_index("race_id")
    return res[SUM_COLS]


def add_bias_candidates(df: pd.DataFrame, q_top3: pd.Series) -> pd.DataFrame:
    """トラックバイアスの候補の列を足す(レースごとの値を各馬に配る)。"""
    top3 = (df["finish"] <= 3).astype(float).where(df["finish"].notna())
    resid = top3 - q_top3
    info = df.groupby("race_id", sort=False).agg(
        venue=("venue", "first"), course_type=("course_type", "first"), date=("date", "first"),
        round=("round", "first"), course=("course", "first"), rail=("rail", "first"),
        dist_bucket=("dist_bucket", "first"))
    for c in ("venue", "course_type", "course", "rail"):
        info[c] = info[c].astype(object).where(info[c].notna(), "")
    info["dist_bucket"] = info["dist_bucket"].astype(float)
    vc = ["venue", "course_type"]
    cc = vc + ["course", "rail", "dist_bucket"]

    raw_s, res_s = _race_sums(df, top3), _race_sums(df, resid)
    cols = {}
    t_raw, t_res = _today(info, raw_s, vc), _today(info, res_s, vc)
    d = _diff(t_raw)
    cols["today_front_bias"], cols["today_inner_bias"] = d["front"], d["inner"]
    d = _diff(t_res)
    cols["today_front_res"], cols["today_inner_res"] = d["front"], d["inner"]
    cols["today_n_races"] = t_raw["n_races"]
    # 当日のコースごと(件数が少ないので、当日の競馬場・芝ダ全体に寄せる)
    tc = _today(info, res_s, cc)
    d = _diff(tc, prior=t_res.reindex(tc.index), k=COURSE_K)
    cols["course_today_front_res"], cols["course_today_inner_res"] = d["front"], d["inner"]
    # 前の開催日(直近10日)の、強い馬を差し引いた偏り
    p_vc = _prev_days(info, res_s, vc)
    d = _diff(p_vc)
    cols["track_front_res"], cols["track_inner_res"] = d["front"], d["inner"]
    # 前の開催日の、コースごと(競馬場・芝ダ全体に寄せる)
    p_cc = _prev_days(info, res_s, cc)
    d = _diff(p_cc, prior=p_vc.reindex(p_cc.index), k=COURSE_K)
    cols["course_front_res"], cols["course_inner_res"] = d["front"], d["inner"]

    per_race = pd.DataFrame(cols)
    return df.join(per_race, on="race_id")


BIAS_COLUMNS = ["today_front_bias", "today_inner_bias", "today_front_res", "today_inner_res", "today_n_races",
                "course_today_front_res", "course_today_inner_res", "track_front_res", "track_inner_res",
                "course_front_res", "course_inner_res"]


# ---------------------------------------------------------------- 相性

def shrunk_rate(df: pd.DataFrame, key: list[str], flag: str, k: float, prior: pd.Series) -> pd.Series:
    """key ごとの『そのレースより前』の率を、回数が少ないときは prior に寄せる: (該当数 + k×prior) ÷ (回数 + k)。"""
    n, rate = _expanding_rate(df, key, flag)
    hits = (rate * n).fillna(0.0)
    return (hits + k * prior) / (n + k)


def add_affinity_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df["top2"] = (df["finish"] <= 2).astype(int)
    df["_all"] = 0
    prior = {f: _expanding_rate(df, "_all", f)[1] for f in ("win", "top2", "top3")}  # 前日までの全体の率
    _, jockey_top3 = _expanding_rate(df, "jockey_id", "top3")
    cols = {}
    for k in TJ_KS:
        for f in ("win", "top2", "top3"):
            cols[f"tj_{f}_shr{k}"] = shrunk_rate(df, ["trainer_id", "jockey_id"], f, k, prior[f])
            cols[f"js_{f}_shr{k}"] = shrunk_rate(df, ["jockey_id", "sire_id"], f, k, prior[f])
            cols[f"ts_{f}_shr{k}"] = shrunk_rate(df, ["trainer_id", "sire_id"], f, k, prior[f])
        # 全体の率ではなく、その騎手の率に寄せる版
        cols[f"tj_top3_jshr{k}"] = shrunk_rate(df, ["trainer_id", "jockey_id"], "top3", k,
                                               jockey_top3.fillna(prior["top3"]))
    cols["js_n"] = _expanding_rate(df, ["jockey_id", "sire_id"], "top3")[0]
    cols["ts_n"] = _expanding_rate(df, ["trainer_id", "sire_id"], "top3")[0]
    # 種牡馬が分からない馬は、相性も分からない
    no_sire = df["sire_id"].isna()
    for c in list(cols):
        if c.startswith(("js_", "ts_")):
            cols[c] = cols[c].where(~no_sire)
    return pd.concat([df.drop(columns=["_all"]), pd.DataFrame(cols, index=df.index)], axis=1)


# ---------------------------------------------------------------- 比べ方

def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int, rounds: int) -> np.ndarray:
    cats = [c for c in CATEGORICAL if c in features]
    params = dict(PARAMS, seed=seed)
    ds = lgb.Dataset(train[features], train["win"], categorical_feature=cats, free_raw_data=False)
    m = lgb.train(params, ds, num_boost_round=rounds)
    return normalize_within_race(test, m.predict(test[features]))


def metrics(test: pd.DataFrame, p: np.ndarray) -> dict:
    y = test["win"].to_numpy()
    t = test.assign(_p=p)
    top = t.loc[t.groupby("race_id")["_p"].idxmax()]
    roi = float((top["win"] * top["win_odds"].fillna(0)).sum() / len(top) * 100)
    return {"auc": auc(y, p), "logloss": logloss(y, p), "hit": float(top["win"].mean()), "roi": roi}


def race_logloss(test: pd.DataFrame, p: np.ndarray) -> pd.Series:
    y = test["win"].to_numpy()
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    return pd.Series(ll, index=test.index).groupby(test["race_id"]).sum()


def bootstrap_ci(diff: pd.Series, n_horses: int, reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    """レース単位で引き直した、1頭あたり LogLoss の差の 95% 区間。"""
    rng = np.random.default_rng(seed)
    v = diff.to_numpy()
    idx = rng.integers(0, len(v), (reps, len(v)))
    means = v[idx].sum(axis=1) / n_horses
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def variant_features(base: list[str], add: list[str], drop: list[str]) -> list[str]:
    return [c for c in base if c not in drop] + [c for c in add if c not in base]


def run(df: pd.DataFrame, variants: dict[str, tuple[list[str], list[str]]], seeds: int, rounds: int,
        out_dir: Path, periods: list[str]) -> pd.DataFrame:
    """各候補について、能力のみ・オッズあり × 評価期間 × 乱数 で学習し、指標を集める。途中の結果は保存して再開できる。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "runs.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    rows = []
    for period in periods:
        cut = pd.Timestamp(period)
        train, test = df[df["date"] < cut], df[df["date"] >= cut]
        n_h = len(test)
        for market in (False, True):
            base = MARKET_MODEL_FEATURES if market else ABILITY_MODEL_FEATURES
            base_ll = None
            for name, (add, drop) in variants.items():
                feats = variant_features(base, add, drop)
                per_seed, race_ll = [], []
                for s in range(seeds):
                    key = f"{period}|{int(market)}|{name}|{s}"
                    ll_file = out_dir / f"ll_{key.replace('|', '_')}.pkl"
                    if key in cache and ll_file.exists():
                        per_seed.append(cache[key])
                        race_ll.append(pd.read_pickle(ll_file))
                        continue
                    t0 = time.time()
                    p = _fit_predict(train, test, feats, s, rounds)
                    m = metrics(test, p)
                    rl = race_logloss(test, p)
                    rl.to_pickle(ll_file)
                    cache[key] = m
                    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
                    per_seed.append(m)
                    race_ll.append(rl)
                    print(f"  {key}: AUC {m['auc']:.4f} LL {m['logloss']:.5f} ({time.time() - t0:.0f}s)", flush=True)
                avg = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
                rl_mean = sum(race_ll) / len(race_ll)
                if name == "base":
                    base_ll = rl_mean
                lo, hi = bootstrap_ci(rl_mean - base_ll, n_h) if base_ll is not None and name != "base" else (0.0, 0.0)
                rows.append({"period": period, "model": "オッズあり" if market else "能力のみ", "variant": name,
                             "n_races": int(test["race_id"].nunique()), **avg, "dll_lo": lo, "dll_hi": hi})
    return pd.DataFrame(rows)


def bias_variants() -> dict[str, tuple[list[str], list[str]]]:
    old = ["track_front_bias", "track_inner_bias"]
    return {
        "base": ([], []),
        "当日の偏り(着順)": (["today_front_bias", "today_inner_bias", "today_n_races"], []),
        "当日の偏り(強い馬を差し引き)": (["today_front_res", "today_inner_res", "today_n_races"], []),
        "前の開催日(強い馬を差し引き)を追加": (["track_front_res", "track_inner_res"], []),
        "前の開催日(強い馬を差し引き)に置き換え": (["track_front_res", "track_inner_res"], old),
        "コースごと・前の開催日(差し引き)": (["course_front_res", "course_inner_res"], []),
        "コースごと・当日(差し引き)": (["course_today_front_res", "course_today_inner_res", "today_n_races"], []),
        "バイアス候補すべて": (["today_front_res", "today_inner_res", "today_n_races", "track_front_res",
                         "track_inner_res", "course_front_res", "course_inner_res",
                         "course_today_front_res", "course_today_inner_res"], []),
    }


def tj_k_variants() -> dict[str, tuple[list[str], list[str]]]:
    v = {"base": ([], [])}
    for k in TJ_KS:
        v[f"厩舎×騎手 3着内率 割引 k={k}"] = ([f"tj_top3_shr{k}"], ["tj_top3_rate"])
    return v


def affinity_variants(k: int) -> dict[str, tuple[list[str], list[str]]]:
    tj = [f"tj_win_shr{k}", f"tj_top2_shr{k}", f"tj_top3_shr{k}"]
    js = ["js_n", f"js_win_shr{k}", f"js_top3_shr{k}"]
    ts = ["ts_n", f"ts_win_shr{k}", f"ts_top3_shr{k}"]
    return {
        "base": ([], []),
        f"厩舎×騎手 騎手の率に寄せる k={k}": ([f"tj_top3_jshr{k}"], ["tj_top3_rate"]),
        f"厩舎×騎手 勝率・連対率・3着内率 k={k}": (tj, ["tj_top3_rate"]),
        f"騎手×種牡馬 k={k}": (js, []),
        f"調教師×種牡馬 k={k}": (ts, []),
        f"相性すべて k={k}": (tj + js + ts, ["tj_top3_rate"]),
    }


def best_k(res: pd.DataFrame) -> int:
    """LogLoss の改善(4つの組の平均)がいちばん大きい k。"""
    base = res[res["variant"] == "base"].set_index(["period", "model"])["logloss"]
    best, best_d = TJ_KS[0], np.inf
    for k in TJ_KS:
        r = res[res["variant"] == f"厩舎×騎手 3着内率 割引 k={k}"].set_index(["period", "model"])["logloss"]
        d = float((r - base).mean())
        if d < best_d:
            best, best_d = k, d
    return best


def to_markdown(res: pd.DataFrame) -> str:
    lines = []
    for (period, model), g in res.groupby(["period", "model"], sort=False):
        b = g[g["variant"] == "base"].iloc[0]
        lines.append(f"\n### 評価 {period} から({b['n_races']:,} レース) / {model}\n")
        lines.append("| 候補 | AUC | LogLoss | ΔLogLoss の95%区間(×1e-4) | 1番手の的中 | 単勝回収率 |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            ci = "" if r["variant"] == "base" else f"{r['dll_lo'] * 1e4:+.1f} 〜 {r['dll_hi'] * 1e4:+.1f}"
            lines.append(f"| {r['variant']} | {r['auc']:.4f} | {r['logloss']:.5f} | {ci} | {r['hit']:.1%} | {r['roi']:.1f}% |")
    return "\n".join(lines)


def prepare(conn: sqlite3.Connection) -> pd.DataFrame:
    from .features import build_features, load_results

    df = build_features(load_results(conn))
    extra = pd.read_sql_query("SELECT race_id, course, rail FROM races", conn)
    df = df.merge(extra, on="race_id", how="left")
    q = market_top3(df, TEST_PERIODS[0])
    df = add_bias_candidates(df, q)
    return add_affinity_candidates(df)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="特徴量の候補を、足す前後で比べる(DB は読むだけ)")
    ap.add_argument("--db", default="data/keiba.db")
    ap.add_argument("--out", default="data/feature_trials")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--only", choices=["bias", "affinity"], default=None)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    conn = sqlite3.connect(f"file:{Path(args.db).as_posix()}?mode=ro", uri=True)  # 読み取り専用で開く
    df = prepare(conn)
    print(f"特徴量を作りました: {len(df):,} 行, {time.time() - t0:.0f}s", flush=True)
    cand = [c for c in BIAS_COLUMNS + ["tj_top3_shr10", "js_top3_shr10", "ts_top3_shr10", "js_n"] if c in df]
    print(df.loc[df["date"] >= TEST_PERIODS[0], cand].describe().T[["count", "mean", "std"]].to_string(), flush=True)

    parts = []
    if args.only in (None, "bias"):
        parts.append(("トラックバイアスの残り", run(df, bias_variants(), args.seeds, args.rounds, out, TEST_PERIODS)))
    if args.only in (None, "affinity"):
        r1 = run(df, tj_k_variants(), args.seeds, args.rounds, out, TEST_PERIODS)
        k = best_k(r1)
        r2 = run(df, affinity_variants(k), args.seeds, args.rounds, out, TEST_PERIODS)
        parts.append(("相性: 厩舎×騎手の割り引き(k の比較)", r1))
        parts.append((f"相性: k={k} での版と、騎手×種牡馬・調教師×種牡馬", r2))
    text = "\n\n".join(f"## {title}\n{to_markdown(r)}" for title, r in parts)
    (out / "results.md").write_text(text, encoding="utf-8")
    pd.concat([r.assign(part=t) for t, r in parts]).to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    print(text)
    print(f"\n保存しました: {out / 'results.md'}  (全体 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
