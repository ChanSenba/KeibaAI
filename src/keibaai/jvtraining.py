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
JV_FEATURES = BASE_FEATURES + COURSE_FEATURES
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


def add_course_relative(tr: pd.DataFrame) -> pd.DataFrame:
    """追い切りごとに、同じ「トレセン・坂路/ウッド・(ウッドは)コース・時期」の追い切りと比べた値を足す。

    f4_z / f2_z / l1_z: そのグループの中央値との差(秒。負なら速い)。f4_pct: グループ内の4Fの順位(0 が最速、1 が最遅)。
    accel: 最初の1Fのラップ − 最後の1Fのラップ(正なら加速、負なら失速)。美浦坂路は改修の前後で別グループ。
    グループの基準は全期間の追い切りから出す(馬ごとの結果は使わない)。
    """
    tr = tr.copy()
    new_miho = (tr["kind"] == "slope") & (tr["center"] == "0") & (tr["date"].astype(str) >= MIHO_NEW_SLOPE)
    key = (tr["kind"].astype(str) + tr["center"].fillna("").astype(str) + tr["course"].fillna("").astype(str)
           + new_miho.map({True: "new", False: ""}))
    g = tr.groupby(key, sort=False)
    for col in ("f4", "f2", "l1"):
        tr[f"{col}_z"] = tr[col] - g[col].transform("median")
    tr["f4_pct"] = g["f4"].rank(pct=True)
    tr["accel"] = (tr["f4"] - tr["f3"]) - tr["l1"]
    return tr


def add_jv_features(conn: sqlite3.Connection, df: pd.DataFrame) -> pd.DataFrame:
    """出走の表(horse_id と date の列がある)に、レース前の調教タイムの特徴量(JV_FEATURES)を足す。

    調教はそのレースの日より前のものだけを使う。直近14日の追い切り(坂路4F58秒以内・ウッド56秒以内)について、
    本数・一番時計の4Fと最後の1F・最後の1本・前回の追い切りからの日数を出し、
    15〜180日前の追い切りの4F平均(3本以上のとき)との差と、同じレースの他馬との差も出す。
    調教のデータが DB になければ、列は欠損のまま。
    """
    for c in JV_FEATURES:
        df[c] = np.nan
    n = conn.execute("SELECT COUNT(*) FROM jv_training").fetchone()[0]
    if not n or df.empty:
        return df
    tr = pd.read_sql_query(
        "SELECT horse_id, date, time, kind, center, course, f4, f3, f2, l1 FROM jv_training WHERE f4 IS NOT NULL", conn)
    tr = tr[tr["f4"] <= tr["kind"].map(FAST_F4)]
    tr = add_course_relative(tr)
    tr["date"] = pd.to_datetime(tr["date"])
    tr = tr[tr["horse_id"].isin(set(df["horse_id"].dropna()))].sort_values(["date", "time"])
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
    })
    base = m[m["ago"] > 14].groupby(["horse_id", "day"], sort=False)["f4"].agg(["mean", "size"])
    agg["jv_f4_vs_base"] = agg["jv_best_f4"] - base["mean"].where(base["size"] >= 3).reindex(agg.index)
    out = df[["horse_id"]].assign(day=pd.to_datetime(df["date"])).merge(
        agg.reset_index(), on=["horse_id", "day"], how="left")
    for c in agg.columns:
        df[c] = out[c].to_numpy()
    df["jv_f4_vs_field"] = df["jv_best_f4"] - df.groupby("race_id")["jv_best_f4"].transform("mean")
    return df
