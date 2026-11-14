"""JRA-VAN の調教タイム(坂路 HC・ウッドチップ WC)を読んで DB に入れ、調教の特徴量を作る。

書き出し(jvexport.py、32bit Python)で作った生データを読む。1行は「ファイル名<タブ>レコード」。
レコードは固定長。位置は 1 始まり。タイムは 0.1 秒単位で、0 は計測なし。
  共通: 1-2 種別, 3 区分(0 は削除), 4-11 作成日, 12 トレセン(0 美浦 / 1 栗東), 13-20 調教日, 21-24 時刻, 25-34 馬の番号(血統登録番号)
  坂路 HC(60文字):  35-38 4F合計, 39-41 ラップ, 42-45 3F合計, 46-48 ラップ, 49-52 2F合計, 53-55 ラップ, 56-58 1Fのラップ
  ウッド WC(105文字): 35 コース, 36 回り(0 右 / 1 左), 37 予備, 38- 10F〜2F の「合計(4)＋ラップ(3)」が9組, 最後に 1Fのラップ(3)
馬の番号は netkeiba の馬 ID と同じ10桁。4F − 3F = 最初の1Fのラップ、になっていることで読み方を確かめてある。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = {"HC": "slope", "WC": "wood"}
# 追い切りとして扱う4Fの上限(秒)。これより遅いのは流し(キャンター)
FAST_F4 = {"slope": 58.0, "wood": 56.0}
# 基準(同じトレセン・コース・時期の追い切りの中央値)との差と順位。美浦と栗東・坂路とウッド・改修の前後で
# 時計の水準が違うので、タイムそのものではなく「そのコースの追い切りの中でどれだけ速いか」で見る
BASE_FEATURES = ["jv_n14", "jv_best_f4", "jv_best_l1", "jv_last_f4", "jv_last_l1", "jv_days_since",
                 "jv_f4_vs_base", "jv_f4_vs_field"]
COURSE_FEATURES = ["jv_best_f4_z", "jv_best_pct", "jv_last_f4_z", "jv_last_pct", "jv_last_f2_z", "jv_last_l1_z",
                   "jv_last_accel"]
# 4Fのラップ(1Fごとの時計)。lap1 = 4F−3F(最初の1F)、lap2 = 3F−2F、lap3 = 2F−最後の1F。最後の1Fは l1。
# 最後の追い切りのラップと、そのコースの標準(中央値)との差。ラップの形(前半を踏んだか、終いを伸ばしたか)を見る
LAP_FEATURES = ["jv_last_lap1", "jv_last_lap2", "jv_last_lap3", "jv_last_lap1_z", "jv_last_lap2_z", "jv_last_lap3_z",
                "jv_best_lap1", "jv_best_lap2", "jv_best_lap3"]
# 同じ日・同じトレセン・同じコースの追い切り(速いもの)の中央値との差。その日の馬場・天気・時間帯による時計の違いを消す。
# 直近14日の最後の1本・一番時計の4Fと最後の1F、最後の1本の時刻(時)
DAY_FEATURES = ["jv_last_day_f4", "jv_last_day_l1", "jv_best_day_f4", "jv_best_day_l1", "jv_last_hour"]
# 調教の量: 流しも含めた本数(14日・28日)、28日の追い切りの本数、そのうちウッドの割合、最後の追い切りがウッドか
VOLUME_FEATURES = ["jv_n_all14", "jv_n_all28", "jv_n_fast28", "jv_wood_share28", "jv_last_is_wood"]
# DAY・VOLUME は能力のみのモデルで 2025-01〜の 6,441 レースで勝ち馬の対数尤度 +3.0/1000(95%区間 +0.0〜+6.2)、
# 2023年までで学習して 2024-01〜の 9,889 レースでは +4.3(+2.1〜+6.5)。新馬戦で +18。数字は docs/spec.md 8-23
JV_FEATURES = BASE_FEATURES + COURSE_FEATURES + LAP_FEATURES + DAY_FEATURES + VOLUME_FEATURES
DAY_MIN_WORKOUTS = 8  # 同じ日・同じコースの追い切りがこれより少ない日は、その日の基準を出さない
# ウッドの調教タイムは 2021-07-27 から。28日ぶんが揃う前のレースは、量の列をすべて欠損にする(坂路だけの本数は数が違う)
VOLUME_FROM = "2021-08-24"
# 量はレースの3日前までの調教で数える。予想は木曜の夜に出すので、金・土の調教(多くは前日の流し)はまだ DB にない。
# 学習のときも同じ範囲で数え、当日の予想と学習で数がずれないようにする
VOLUME_MIN_AGO = 3
MIHO_NEW_SLOPE = "2023-09-28"  # 美浦坂路は2023年に改修(高低差18m→33m)。この日から新コース
COLUMNS = ["horse_id", "date", "time", "kind", "center", "course", "turn", "f4", "f3", "f2", "l1"]


def _t(s: str) -> float | None:
    """0.1秒単位の数字を秒にする。空・0・数字でないものは None。"""
    s = s.strip()
    if not s.isdigit() or int(s) == 0:
        return None
    return int(s) / 10


def parse_record(rec: str) -> tuple | None:
    """HC / WC のレコード1件を COLUMNS の順の組にする。対象外(削除・別種別・短すぎ)は None。"""
    kind = KINDS.get(rec[:2])
    if kind is None or rec[2:3] == "0":
        return None
    date, time_, horse = rec[12:20], rec[20:24], rec[24:34]
    if len(date) != 8 or not date.isdigit() or not horse.strip():
        return None
    iso = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    if kind == "slope":
        if len(rec) < 58:
            return None
        f4, f3, f2, l1 = _t(rec[34:38]), _t(rec[41:45]), _t(rec[48:52]), _t(rec[55:58])
        course = turn = None
    else:
        if len(rec) < 103:
            return None
        course, turn = rec[34:35], rec[35:36]
        pair = lambda k: _t(rec[37 + 7 * k: 41 + 7 * k])  # k=0 が10F … k=6 が4F, 7 が3F, 8 が2F
        f4, f3, f2, l1 = pair(6), pair(7), pair(8), _t(rec[100:103])
    return horse, iso, time_, kind, rec[11:12], course, turn, f4, f3, f2, l1


def import_raw(conn: sqlite3.Connection, raw_dir: str | Path, chunk: int = 50_000) -> int:
    """raw_dir の *.txt(jvexport の出力)を読んで jv_training に入れる。入れた件数を返す。"""
    n = 0
    sql = f"INSERT OR REPLACE INTO jv_training ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})"
    for path in sorted(Path(raw_dir).glob("*.txt")):
        if path.stem in ("RACE", "DIFN"):  # 成績と馬のマスタは jvrace.py が読む
            continue
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                _, _, rec = line.rstrip("\n").partition("\t")
                row = parse_record(rec)
                if row:
                    rows.append(row)
                if len(rows) >= chunk:
                    conn.executemany(sql, rows)
                    conn.commit()  # まとめて最後に確定すると、その間ほかの処理(収集や画面)が DB に書けない
                    n += len(rows)
                    rows = []
        conn.executemany(sql, rows)
        conn.commit()
        n += len(rows)
    return n


def add_course_relative(tr: pd.DataFrame, min_history: int = 30) -> pd.DataFrame:
    """追い切りごとに、同じ「トレセン・坂路/ウッド・(ウッドは)コース・時期」の追い切りと比べた値を足す。

    f4_z / f2_z / l1_z: そのグループの中央値との差(秒。負なら速い)。f4_pct: グループ内の4Fの順位(0 が最速、1 が最遅)。
    accel: 最初の1Fのラップ − 最後の1Fのラップ(正なら加速、負なら失速)。美浦坂路は改修の前後で別グループ。
    基準は調教日の前日まで。同日・未来の追い切りは使わず、過去 min_history 本未満は欠損。
    f4_pct は過去分布での位置(同値は中点)で、全期間内の順位ではない。
    """
    original_index = tr.index
    tr = tr.reset_index(drop=True).copy()
    new_miho = (tr["kind"] == "slope") & (tr["center"] == "0") & (tr["date"].astype(str) >= MIHO_NEW_SLOPE)
    key = pd.MultiIndex.from_arrays([tr["kind"], tr["center"].fillna(""),
                                    tr["course"].fillna(""), new_miho])
    # 1Fごとのラップ(4F−3F、3F−2F、2F−最後の1F)。これも同じ基準(前日まで)との差を出す
    tr["lap1"], tr["lap2"], tr["lap3"] = (tr["f4"] - tr["f3"]).round(1), (tr["f3"] - tr["f2"]).round(1), (tr["f2"] - tr["l1"]).round(1)
    for col in ("f4_z", "f2_z", "l1_z", "f4_pct", "lap1_z", "lap2_z", "lap3_z"):
        tr[col] = np.nan
    # タイムの種類は少ないので累積ヒストグラムで中央値・分位を求める。
    # 全期間の値は座標としてだけ使い、度数へ加えるのはその日の計算が終わってから。
    groups = tr.groupby(key, sort=False).indices
    for col in ("f4", "f2", "l1", "lap1", "lap2", "lap3"):
        values = pd.to_numeric(tr[col], errors="coerce").to_numpy(dtype=float)
        z = np.full(len(tr), np.nan)
        pct = np.full(len(tr), np.nan)
        for positions in groups.values():
            g = tr.iloc[positions]
            coords = np.unique(values[positions][np.isfinite(values[positions])])
            counts = np.zeros(len(coords), dtype=np.int64)
            n = 0
            for _, indices in g.groupby("date", sort=True).groups.items():
                idx = np.asarray(indices)
                idx = idx[np.isfinite(values[idx])]
                bins = np.searchsorted(coords, values[idx])
                if n >= min_history and n > 0:
                    cumulative = counts.cumsum()
                    lo = np.searchsorted(cumulative, (n - 1) // 2 + 1)
                    hi = np.searchsorted(cumulative, n // 2 + 1)
                    z[idx] = values[idx] - (coords[lo] + coords[hi]) / 2
                    pct[idx] = (cumulative[bins] - counts[bins] / 2) / n
                np.add.at(counts, bins, 1)
                n += len(idx)
        tr[f"{col}_z"] = z
        if col == "f4":
            tr["f4_pct"] = pct
    tr["accel"] = (tr["f4"] - tr["f3"]) - tr["l1"]
    tr.index = original_index
    return tr


def add_day_relative(tr: pd.DataFrame, min_workouts: int = DAY_MIN_WORKOUTS) -> pd.DataFrame:
    """追い切りごとに、同じ日・同じトレセン・坂路/ウッド・(ウッドは)コースの追い切りの中央値との差を足す。

    day_f4 / day_l1: その日の中央値との差(秒。負なら速い)。その日の追い切りが min_workouts 本未満なら欠損。
    hour: 追い切りの時刻(時)。tr は速い追い切りだけ・date は日付型であること。
    同じ日の追い切りはどれもレースの前日以前なので、レース当日以後の情報は入らない。
    """
    tr = tr.copy()
    key = [tr["date"], tr["kind"], tr["center"].fillna(""), tr["course"].fillna("")]
    for c in ("f4", "l1"):
        g = tr.groupby(key)[c]
        tr[f"day_{c}"] = (tr[c] - g.transform("median")).where(g.transform("count") >= min_workouts)
    tr["hour"] = pd.to_numeric(tr["time"].astype(str).str[:2], errors="coerce")
    return tr


def add_jv_features(conn: sqlite3.Connection, df: pd.DataFrame) -> pd.DataFrame:
    """出走の表(horse_id と date の列がある)に、レース前の調教タイムの特徴量(JV_FEATURES)を足す。

    調教はそのレースの日より前のものだけを使う。直近14日の追い切り(坂路4F58秒以内・ウッド56秒以内)について、
    本数・一番時計の4Fと最後の1F・最後の1本・前回の追い切りからの日数を出し、
    15〜180日前の追い切りの4F平均(3本以上のとき)との差と、同じレースの他馬との差も出す。
    同じ日・同じコースの追い切りとの差(DAY_FEATURES)と、流しも含めた直近28日の調教の量(VOLUME_FEATURES)も出す。
    調教のデータが DB になければ、列は欠損のまま。
    """
    for c in JV_FEATURES:
        df[c] = np.nan
    n = conn.execute("SELECT COUNT(*) FROM jv_training").fetchone()[0]
    if not n or df.empty:
        return df
    every = pd.read_sql_query("SELECT horse_id, date, time, kind, center, course, f4, f3, f2, l1 FROM jv_training", conn)
    every["date"] = pd.to_datetime(every["date"])
    fast = every["f4"].notna() & (every["f4"] <= every["kind"].map(FAST_F4))
    tr = add_day_relative(add_course_relative(every[fast]))
    horses = set(df["horse_id"].dropna())
    tr = tr[tr["horse_id"].isin(horses)].sort_values(["date", "time"])
    keys = df[["horse_id"]].assign(day=pd.to_datetime(df["date"])).drop_duplicates()
    m = keys.merge(tr, on="horse_id")
    ago = (m["day"] - m["date"]).dt.days
    m = m[(ago >= 1) & (ago <= 180)].assign(ago=ago)
    m = m.sort_values(["date", "time"])
    near = m[m["ago"] <= 14]
    g = near.groupby(["horse_id", "day"], sort=False)
    agg = pd.DataFrame({
        "jv_n14": g.size(), "jv_best_f4": g["f4"].min(), "jv_best_l1": g["l1"].min(),
        "jv_last_f4": g["f4"].last(), "jv_last_l1": g["l1"].last(), "jv_days_since": g["ago"].min(),
        "jv_best_f4_z": g["f4_z"].min(), "jv_best_pct": g["f4_pct"].min(),
        "jv_last_f4_z": g["f4_z"].last(), "jv_last_pct": g["f4_pct"].last(),
        "jv_last_f2_z": g["f2_z"].last(), "jv_last_l1_z": g["l1_z"].last(), "jv_last_accel": g["accel"].last(),
        **{f"jv_last_{c}": g[c].last() for c in ("lap1", "lap2", "lap3", "lap1_z", "lap2_z", "lap3_z")},
    })
    # 一番時計(4Fが最速)の追い切りのラップ
    best = near.loc[near.groupby(["horse_id", "day"], sort=False)["f4"].idxmin()].set_index(["horse_id", "day"])
    for c in ("lap1", "lap2", "lap3"):
        agg[f"jv_best_{c}"] = best[c]
    base = m[m["ago"] > 14].groupby(["horse_id", "day"], sort=False)["f4"].agg(["mean", "size"])
    agg["jv_f4_vs_base"] = agg["jv_best_f4"] - base["mean"].where(base["size"] >= 3).reindex(agg.index)
    agg["jv_last_day_f4"], agg["jv_last_day_l1"] = g["day_f4"].last(), g["day_l1"].last()
    agg["jv_best_day_f4"], agg["jv_best_day_l1"] = g["day_f4"].min(), g["day_l1"].min()
    agg["jv_last_hour"] = g["hour"].last()
    agg["jv_last_is_wood"] = (g["kind"].last() == "wood").astype(float)
    # 調教の量(28日)。流し(4Fの時計がない・遅いもの)も数える
    v = keys.merge(every[every["horse_id"].isin(horses)][["horse_id", "date", "kind"]].assign(fast=fast.astype(bool)),
                   on="horse_id")
    v_ago = (v["day"] - v["date"]).dt.days
    v = v[(v_ago >= VOLUME_MIN_AGO) & (v_ago <= 28)].assign(ago=v_ago)
    v["fast"] = v["fast"].astype(bool)
    by = [v["horse_id"], v["day"]]
    vol = pd.DataFrame({"jv_n_all28": v.groupby(by, sort=False).size(), "jv_n_all14": (v["ago"] <= 14).groupby(by).sum(),
                        "jv_n_fast28": v["fast"].groupby(by).sum()})
    vf = v[v["fast"].to_numpy()]
    vol["jv_wood_share28"] = (vf["kind"] == "wood").groupby([vf["horse_id"], vf["day"]]).mean()
    agg = agg.join(vol, how="outer")
    out = df[["horse_id"]].assign(day=pd.to_datetime(df["date"])).merge(
        agg.reset_index(), on=["horse_id", "day"], how="left")
    for c in agg.columns:
        df[c] = out[c].to_numpy()
    # 本数の 0 は「調教していない」の意味なので、調教タイムがそのレースの3日前まで DB に入っているときだけ入れる。
    # 取り込みが遅れているときや、ウッドの揃う前のレースは欠損にする
    day = pd.to_datetime(df["date"])
    latest = every["date"].max()
    covered = ((day >= VOLUME_FROM) & (day - pd.Timedelta(days=VOLUME_MIN_AGO) <= latest)).to_numpy()
    for c in ("jv_n_all14", "jv_n_all28", "jv_n_fast28"):
        df.loc[covered, c] = df.loc[covered, c].fillna(0)
    df.loc[(day < VOLUME_FROM).to_numpy(), VOLUME_FEATURES] = np.nan
    df["jv_f4_vs_field"] = df["jv_best_f4"] - df.groupby("race_id")["jv_best_f4"].transform("mean")
    return df
