"""★・期待値のモデル(オッズ込み)に入れる、単勝以外の券種のオッズの特徴量。

複勝・馬単・三連単のオッズから、馬ごとの「その券種の市場が見ている強さ」(オッズの逆数の合計をレース内で合計1にしたもの)を出し、
単勝の市場の勝率との対数の差を特徴量にする。単勝と組合せの券種で値付けがずれている馬を、木が拾う。

  d_place   複勝(幅の下限と上限の幾何平均)
  d_exacta1 馬単の1着側(その馬が1着の組の合計)
  d_exacta2 馬単の2着側
  d_tri1    三連単の1着側

2024年までで学習し 2025-01〜の 6,153 レースで、勝ち馬の対数尤度が今の★のモデルより +10.1/1000(95%区間 +7.1〜+13.3)、
2022年までで学習し 2023年の 3,453 レースで +9.6(+5.7〜+13.6)。確定オッズでの数字(docs/spec.md 8-26)。
6券種すべて(ワイド・馬連・三連複も)を使っても +11.1 / +9.4 で変わらないので、JRA-VAN を開く回数の少ない3券種にした。

学習には jravan_odds.db の確定オッズ、予想には予想のときに取った速報オッズ(bet_odds_snapshots)を使う。
どれもそのレースの、その時点のオッズだけから作るので、ほかのレースや結果の情報は入らない。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["d_place", "d_exacta1", "d_exacta2", "d_tri1"]
SHARES = [f"s_{c[2:]}" for c in FEATURES]
# 差を取る単勝の市場も、券種と同じ時点のオッズにする(s_win。学習は確定、予想は同じ取得の単勝)
ALL_SHARES = ["s_win", *SHARES]
# 予想のときに取る速報オッズ(JVRTOpen の種別)と、bet_odds_snapshots の券種名
RT_SPECS = {"0B31": "O1", "0B34": "O4", "0B36": "O6"}
SNAPSHOT_KINDS = ("単勝", "複勝", "馬単", "三連単")
ODDS_DB = "jravan_odds.db"  # keiba.db と同じフォルダ


def final_shares(odds_db: str | Path, since: str = "2018") -> pd.DataFrame:
    """jravan_odds.db の確定オッズから、馬ごとの各券種のオッズの逆数の合計(race_id, number, s_*)。"""
    c = sqlite3.connect(f"file:{Path(odds_db).as_posix()}?mode=ro", uri=True)
    try:
        q = {
            "s_win": "SELECT race_id, a AS number, 1.0/odds AS s FROM odds_win WHERE race_id >= ? AND odds > 0",
            "s_place": "SELECT race_id, a AS number, 1.0/sqrt(odds_min*odds_max) AS s FROM odds_place "
                       "WHERE race_id >= ? AND odds_min > 0 AND odds_max > 0",
            "s_exacta1": "SELECT race_id, a AS number, SUM(1.0/odds) AS s FROM odds_exacta WHERE race_id >= ? AND odds > 0 "
                         "GROUP BY race_id, a",
            "s_exacta2": "SELECT race_id, b AS number, SUM(1.0/odds) AS s FROM odds_exacta WHERE race_id >= ? AND odds > 0 "
                         "GROUP BY race_id, b",
            "s_tri1": "SELECT race_id, a AS number, SUM(1.0/odds) AS s FROM odds_trifecta WHERE race_id >= ? AND odds > 0 "
                      "GROUP BY race_id, a",
        }
        out = None
        for name, sql in q.items():
            d = pd.read_sql_query(sql, c, params=(since,)).rename(columns={"s": name})
            out = d if out is None else out.merge(d, on=["race_id", "number"], how="outer")
    finally:
        c.close()
    return _typed(out)


def snapshot_shares(conn: sqlite3.Connection, race_ids) -> pd.DataFrame:
    """bet_odds_snapshots の、レースごとに券種がそろった最新の取得時刻のオッズから、final_shares と同じ形の表。

    そろった記録 = 単勝・複勝(全馬に幅の上限つき)・馬単と三連単(全部の組にオッズ)が同じ取得時刻にある。
    買い方の画面が残す記録は複勝の上限がないので使わない(学習の複勝は下限と上限の幾何平均)。
    発売直後で売れていない組がある記録も、合計が確定オッズと同じ意味にならないので使わない。
    """
    ids = list(race_ids)
    if not ids:
        return _typed(None)
    marks = ",".join("?" * len(ids))
    kinds = ",".join("?" * len(SNAPSHOT_KINDS))
    rows = pd.read_sql_query(
        f"""SELECT race_id, snapped_at, kind, combo, odds, odds_max FROM bet_odds_snapshots
            WHERE race_id IN ({marks}) AND kind IN ({kinds}) AND odds > 0""", conn, params=[*ids, *SNAPSHOT_KINDS])
    if rows.empty:
        return _typed(None)
    cnt = rows.pivot_table(index=["race_id", "snapped_at"], columns="kind", values="combo", aggfunc="count")
    cnt = cnt.reindex(columns=list(SNAPSHOT_KINDS))
    place_hi = rows[rows["kind"] == "複勝"].assign(ok=lambda d: d["odds_max"] > 0).groupby(
        ["race_id", "snapped_at"])["ok"].all()
    n = cnt["複勝"]
    good = ((cnt["単勝"] == n) & (cnt["馬単"] == n * (n - 1)) & (cnt["三連単"] == n * (n - 1) * (n - 2))
            & place_hi.reindex(cnt.index, fill_value=False) & (n >= 3))
    full = good[good].reset_index()[["race_id", "snapped_at"]]
    if full.empty:
        return _typed(None)
    last = full.groupby("race_id")["snapped_at"].max().reset_index()
    rows = rows.merge(last, on=["race_id", "snapped_at"])
    nums = rows["combo"].str.split("-")
    rows["a"] = pd.to_numeric(nums.str[0])
    rows["b"] = pd.to_numeric(nums.str[1])  # 単勝・複勝は空
    place = rows[rows["kind"] == "複勝"]
    win = rows[rows["kind"] == "単勝"]
    parts = [
        pd.DataFrame({"race_id": win["race_id"], "number": win["a"].astype(int), "s_win": 1.0 / win["odds"]}),
        pd.DataFrame({"race_id": place["race_id"], "number": place["a"].astype(int),
                      "s_place": 1.0 / np.sqrt(place["odds"] * place["odds_max"])}),
        _sum(rows[rows["kind"] == "馬単"], "a", "s_exacta1"),
        _sum(rows[rows["kind"] == "馬単"], "b", "s_exacta2"),
        _sum(rows[rows["kind"] == "三連単"], "a", "s_tri1"),
    ]
    out = parts[0]
    for d in parts[1:]:
        out = out.merge(d, on=["race_id", "number"], how="outer")
    return _typed(out)


def _sum(rows: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    s = (1.0 / rows["odds"]).groupby([rows["race_id"], rows[col].astype(int)]).sum()
    return s.rename(name).rename_axis(["race_id", "number"]).reset_index()


def _typed(d: pd.DataFrame | None) -> pd.DataFrame:
    if d is None:
        d = pd.DataFrame(columns=["race_id", "number", *ALL_SHARES])
    d = d.copy()
    d["race_id"] = d["race_id"].astype(str)
    d["number"] = pd.to_numeric(d["number"]).astype("int64")
    return d


def add_pool_features(df: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """df に FEATURES を足す。全馬の単勝と3券種のオッズがそろわないレースは空(★は今のモデルで出す)。

    df の行の並びと index はそのまま。shares は final_shares・snapshot_shares の表。
    """
    key = pd.DataFrame({"race_id": df["race_id"].astype(str).to_numpy(),
                        "number": pd.to_numeric(df["number"], errors="coerce").to_numpy()})
    m = key.merge(shares[["race_id", "number", *ALL_SHARES]].drop_duplicates(["race_id", "number"]),
                  on=["race_id", "number"], how="left")
    m.index = df.index
    race = df["race_id"]
    # 単勝の市場は券種と同じ時点のもの。df の馬すべてにあること(取消の馬が券種のオッズにだけない、などは使わない)
    inv = pd.to_numeric(m["s_win"], errors="coerce").where(lambda x: x > 0)
    ok = inv.notna().groupby(race).transform("all")
    lp_mkt = np.log(inv / inv.groupby(race).transform("sum"))
    out = {}
    for f, s in zip(FEATURES, SHARES):
        v = pd.to_numeric(m[s], errors="coerce").where(lambda x: x > 0)
        good = ok & v.notna().groupby(race).transform("all")
        share = v / v.groupby(race).transform("sum")
        out[f] = (np.log(share) - lp_mkt).where(good)
    feats = pd.DataFrame(out, index=df.index)
    # 1つでも欠けたレースは全部空にする(学習と予想で同じ形にそろえる)
    feats[~feats.notna().all(axis=1).groupby(race).transform("all")] = np.nan
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df]), feats], axis=1)


def has_pools(df: pd.DataFrame) -> pd.Series:
    """券種の特徴量がそろった行。"""
    return df[FEATURES].notna().all(axis=1) if all(c in df for c in FEATURES) else pd.Series(False, index=df.index)
