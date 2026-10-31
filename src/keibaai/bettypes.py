"""単勝以外(複勝・馬連・ワイド)でも、モデルの見込みが市場より当たるかを、過去の確定オッズと払戻で確かめる(検証用)。

使い方(DB は読むだけ):
    python -m keibaai.bettypes --db data/keiba.db --odds-db data/jravan_odds.db --test-from 2025-01-01

各馬の勝率(モデル)から、ハーヴィルの式(betting.harville_outcomes。2・3着は買い方の画面と同じ GAMMA_2ND・GAMMA_3RD で補正)で1〜3着の並びの確率を作り、券種ごとの買い目の的中確率を出す。
期待値 = 的中確率 × 確定オッズ(複勝・ワイドは最低のほう)。期待値が基準以上の買い目を1点100円ずつ買ったことにして、
実際の払戻(jravan_odds.db の payouts。同着もそのまま)で回収率を数える。基準が「全買い目」の回収率(控除後の平均)より高いか、
その差が偶然の範囲を超えるかを、レースを引き直した95%区間で見る。
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .betting import GAMMA_2ND, GAMMA_3RD, harville_outcomes, hits, place_slots

# 券種 → (betting の券種名, オッズの表, 組の列, オッズの列)
KINDS = {
    "win": ("単勝", "odds_win", ["a"], ["odds"]),
    "place": ("複勝", "odds_place", ["a"], ["odds_min"]),
    "quinella": ("馬連", "odds_quinella", ["a", "b"], ["odds"]),
    "wide": ("ワイド", "odds_wide", ["a", "b"], ["odds_min"]),
}
EV_MINS = (1.0, 1.2, 1.5)
MIN_HIT = 0.03  # 的中確率がこれ未満の買い目は買わない(確率の推定の誤差が大きく、大穴の期待値は過大に出る)


def load_odds(conn: sqlite3.Connection, race_ids: list[str], kind: str) -> dict[str, dict[tuple[int, ...], float]]:
    """race_id → {馬番の組: オッズ}。"""
    _, table, keys, ocols = KINDS[kind]
    out: dict[str, dict[tuple[int, ...], float]] = {}
    marks = ",".join("?" * len(race_ids))
    for row in conn.execute(f"SELECT race_id, {', '.join(keys)}, {ocols[0]} FROM {table} WHERE race_id IN ({marks})", race_ids):
        if row[-1]:
            out.setdefault(row[0], {})[tuple(int(x) for x in row[1:-1])] = float(row[-1])
    return out


def load_payouts(conn: sqlite3.Connection, race_ids: list[str], kind: str) -> dict[tuple[str, tuple[int, ...]], int]:
    """(race_id, 馬番の組) → 100円あたりの払戻(円)。"""
    out = {}
    marks = ",".join("?" * len(race_ids))
    for rid, combo, pay in conn.execute(
            f"SELECT race_id, combo, payout FROM payouts WHERE kind = ? AND race_id IN ({marks})", [kind, *race_ids]):
        out[(rid, tuple(int(x) for x in combo.split("-")))] = int(pay)
    return out


def race_bets(numbers: list[int], p: np.ndarray, odds: dict[tuple[int, ...], float], kind: str,
              g2: float = GAMMA_2ND, g3: float = GAMMA_3RD) -> list[tuple[tuple[int, ...], float, float]]:
    """1レースの (馬番の組, 的中確率, オッズ)。オッズのない組、複勝の発売がない頭数は除く。"""
    n = len(numbers)
    name = KINDS[kind][0]
    if n < 3 or (kind == "place" and place_slots(n) == 0):
        return []
    order, prob = harville_outcomes(p, g2, g3)
    idx = {num: i for i, num in enumerate(numbers)}
    out = []
    for key, o in odds.items():
        if any(k not in idx for k in key):
            continue
        h = hits(order, name, tuple(idx[k] for k in key), n)
        out.append((key, float(prob[h].sum()), o))
    return out


def evaluate_kind(test: pd.DataFrame, odds_conn: sqlite3.Connection, kind: str, g2: float = GAMMA_2ND,
                  g3: float = GAMMA_3RD) -> pd.DataFrame:
    """評価期間の全レース・全買い目について、的中確率・オッズ・期待値・実際の払戻を並べる。

    test は race_id, number, p_win(レース内で合計1)を持つ。オッズか払戻のないレースは入れない。
    """
    ids = sorted(test["race_id"].unique())
    odds = load_odds(odds_conn, ids, kind)
    pays = load_payouts(odds_conn, ids, kind)
    paid = {r for r, _ in pays}
    rows = []
    for rid, g in test.groupby("race_id", sort=True):
        if rid not in odds or rid not in paid:
            continue
        for key, pr, o in race_bets(g["number"].astype(int).tolist(), g["p_win"].to_numpy(float), odds[rid], kind, g2, g3):
            rows.append((rid, key, pr, o, pr * o, pays.get((rid, key), 0)))
    return pd.DataFrame(rows, columns=["race_id", "combo", "prob", "odds", "ev", "payout"])


def roi_table(bets: pd.DataFrame, ev_mins=EV_MINS, min_hit: float = MIN_HIT, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """基準ごとの回収率(1点100円)。先頭の行は、全買い目を買った場合(控除後の平均)。区間はレースを引き直して出す。"""
    rng = np.random.default_rng(seed)
    races = bets["race_id"].unique()
    rows = []
    for label, sel in [("全買い目", bets)] + [(f"期待値≧{e:g}", bets[(bets["ev"] >= e) & (bets["prob"] >= min_hit)]) for e in ev_mins]:
        n = len(sel)
        if n == 0:
            rows.append({"rule": label, "n_bets": 0, "hit_rate": np.nan, "roi": np.nan, "roi_low": np.nan, "roi_high": np.nan})
            continue
        per = sel.groupby("race_id").agg(n=("payout", "size"), pay=("payout", "sum")).reindex(races, fill_value=0)
        idx = rng.integers(0, len(per), size=(n_boot, len(per)))
        rois = per["pay"].to_numpy()[idx].sum(axis=1) / (100 * np.maximum(per["n"].to_numpy()[idx].sum(axis=1), 1)) * 100
        lo, hi = np.percentile(rois, [2.5, 97.5])
        rows.append({"rule": label, "n_bets": n, "hit_rate": float((sel["payout"] > 0).mean()),
                     "roi": float(sel["payout"].sum() / (100 * n) * 100), "roi_low": float(lo), "roi_high": float(hi)})
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| 基準 | 買い目 | 的中率 | 回収率 | 95%区間 |", "|---|---:|---:|---:|---|"]
    for _, r in table.iterrows():
        if r["n_bets"] == 0:
            lines.append(f"| {r['rule']} | 0 | - | - | - |")
        else:
            lines.append(f"| {r['rule']} | {int(r['n_bets']):,} | {r['hit_rate']:.1%} | {r['roi']:.1f}% | "
                         f"{r['roi_low']:.1f}〜{r['roi_high']:.1f}% |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="複勝・馬連・ワイドで、モデルの期待値の買い目が回収率で市場に勝つかを確かめる(DB は読むだけ)")
    ap.add_argument("--db", default="data/keiba.db")
    ap.add_argument("--odds-db", default="data/jravan_odds.db")
    ap.add_argument("--test-from", default="2025-01-01")
    ap.add_argument("--out", default="data/bettypes")
    ap.add_argument("--kinds", nargs="*", default=list(KINDS), choices=list(KINDS))
    ap.add_argument("--max-races", type=int, default=0, help="評価するレース数の上限(0なら全部)。試すとき用")
    args = ap.parse_args(argv)

    from .blend import predict_test_blend
    from .features import build_features, load_results

    conn = sqlite3.connect(f"file:{Path(args.db).as_posix()}?mode=ro", uri=True)
    feats = build_features(load_results(conn))
    odds_conn = sqlite3.connect(f"file:{Path(args.odds_db).as_posix()}?mode=ro", uri=True)
    parts = []
    for use_market, label in ((False, "能力のみ(オッズなし)の勝率を市場と混ぜたもの"), (True, "オッズありの勝率を市場と混ぜたもの")):
        test, blend = predict_test_blend(feats, args.test_from, use_market=use_market)
        if args.max_races:
            test = test[test["race_id"].isin(sorted(test["race_id"].unique())[:args.max_races])]
        print(f"{label}: 評価 {test['race_id'].nunique():,} レース", flush=True)
        for kind in args.kinds:
            bets = evaluate_kind(test, odds_conn, kind)
            parts.append(f"## {KINDS[kind][0]} ― {label}\n評価 {bets['race_id'].nunique():,} レース\n\n{to_markdown(roi_table(bets))}")
            print(parts[-1], flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text("\n\n".join(parts), encoding="utf-8")
    print(f"\n保存しました: {out / 'results.md'}")


if __name__ == "__main__":
    main()
