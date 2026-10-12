"""予想ファクターを、馬ごとの数字や文字で見られる形にする(画面の「ファクター」ページ用)。

学習はしない。出馬表の各馬について、過去の結果だけから作った値を並べる。
血統・ラップは取得したぶんだけ、調教はまだ取得していないので「未取得」と表示する。
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .predict import day_frame

# 画面に出す区分。cols は (列名, 画面での見出し)。
SECTIONS: list[dict] = [
    {"name": "脚質・馬齢・体重",
     "note": "脚質は過去3走の平均位置(頭数比)から推定: 逃げ ≦0.12 / 先行 ≦0.4 / 差し ≦0.7 / 追込。"
             "差し・追込で、レース中に位置を大きく上げる馬はマクリとしています。",
     "cols": [("style", "脚質(推定)"), ("early_pos_rel_3", "平均位置(頭数比)"), ("age", "馬齢"),
              ("body_weight", "馬体重"), ("body_weight_diff", "前走比"), ("bw_vs_avg", "平均体重との差"),
              ("weight_carried", "斤量"), ("weight_ratio", "斤量体重比"), ("weight_change", "斤量の増減")]},
    {"name": "適性(着度数)",
     "note": "今回と同じ条件での過去の成績を『1着-2着-3着-着外』で表します。距離は今回の±200mです。",
     "cols": [("rec_ct", "同じ芝/ダート"), ("rec_dist", "同じ距離帯"), ("rec_dir", "同じ回り"),
              ("rec_venue", "同じ競馬場"), ("rec_ground", "同じ馬場状態"), ("same_dist_top3_rate", "距離帯の3着内率")]},
    {"name": "近走・ローテーション",
     "note": "近5走は新しい順の着順です。叩き2走目などは、休み明け(90日以上)からの数え方です。",
     "cols": [("last5", "近5走の着順"), ("avg_finish_5", "5走の平均着順"), ("time_diff_5", "タイム差(5走平均)"),
              ("l3f_rank_3", "上がり順位(3走平均・頭数比)"), ("rotation", "ローテーション"),
              ("days_since_last", "前走からの日数")]},
    {"name": "騎手・厩舎",
     "note": "リーディングは、その年の1月から前日までの勝利数の順位です。成績は過去の結果だけで、当日分は含みません。",
     "cols": [("jockey", "騎手"), ("jockey_rank", "リーディング順位"), ("jockey_wins_year", "今年の勝利数"),
              ("jockey_ct_top3_rate", "コース種別の3着内率"), ("jockey_dist_top3_rate", "距離帯の3着内率"),
              ("jockey_venue_top3_rate", "競馬場の3着内率"), ("pair_n", "この馬との騎乗回数"),
              ("pair_top3_rate", "この馬との3着内率"), ("jockey_change", "乗り替わり"),
              ("tj_n", "厩舎×騎手の回数"), ("tj_top3_rate", "厩舎×騎手の3着内率")]},
    {"name": "馬場・枠・展開",
     "note": "トラックバイアスは直近10日以内の同じ競馬場・芝ダートの偏り(正なら先行有利・内有利)。"
             "ペース予想は、メンバーの脚質構成からの目安です。",
     "cols": [("bracket", "枠"), ("number", "馬番"), ("inner_pos", "内外(馬番/頭数)"),
              ("track_front_bias", "先行有利度"), ("track_inner_bias", "内有利度"), ("pace_forecast", "ペース予想"),
              ("n_front_runners", "先行寄りの馬の数"), ("class_change", "クラス変化"), ("distance_change", "距離の変化(m)"),
              ("dist_note", "距離短縮・延長")]},
    {"name": "オッズのゆがみ",
     "note": "騎手の単勝回収率が高いほど過剰人気の傾向、前走の人気 − 着順が大きいほど前走は人気以上に走っています。",
     "cols": [("win_odds", "単勝オッズ"), ("popularity", "人気"), ("jockey_roi", "騎手の単勝回収率(倍)"),
              ("prev_pop_gap", "前走の人気 − 着順")]},
    {"name": "血統",
     "note": "父・母の父の産駒が、同じ条件で過去にどれだけ3着以内に来たか。血統を取得した馬だけ出ます(keibaai pedigree)。",
     "cols": [("sire", "父"), ("damsire", "母の父"), ("sire_line", "父の系統"), ("sire_top3_rate", "父産駒の3着内率"),
              ("sire_ctdist_top3_rate", "父産駒・芝ダ×距離帯"), ("sire_ground_top3_rate", "父産駒・芝ダ×馬場"),
              ("damsire_top3_rate", "母父産駒の3着内率"), ("damsire_ctdist_top3_rate", "母父産駒・芝ダ×距離帯")]},
    {"name": "調教", "note": "調教のデータは、まだ取得していません。", "cols": []},
    {"name": "ラップ・ペース",
     "note": "ペース差 = 前半3F − 後半3F(プラスは後傾=スロー寄り)。RPCI そのものではなく、結果ページのラップからの代用です。",
     "cols": [("prev_pace_diff", "前走のペース差"), ("pace_diff_3", "ペース差(3走平均)"),
              ("l3f_vs_lap_3", "自分の上がり−場の後半3F(3走平均)")]},
]

STYLES = ["逃げ", "先行", "差し", "追込"]


def style_label(early_pos_rel_3: float, pos_gain_3: float) -> str:
    """過去3走の平均位置(頭数比)から脚質を推定する。履歴がなければ「不明」。"""
    if early_pos_rel_3 != early_pos_rel_3:
        return "不明"
    if early_pos_rel_3 <= 0.12:
        return "逃げ"
    if early_pos_rel_3 <= 0.4:
        return "先行"
    label = "差し" if early_pos_rel_3 <= 0.7 else "追込"
    if pos_gain_3 == pos_gain_3 and pos_gain_3 >= 0.25:
        return "マクリ"
    return label


def rotation_label(days: float, runs_since_layoff: float) -> str:
    if days != days:
        return "初出走"
    if days <= 8:
        return "連闘"
    if days >= 90:
        return f"休み明け({days / 7:.0f}週ぶり)"
    n = int(runs_since_layoff)
    if n == 2:
        return "叩き2走目"
    if n == 3:
        return "叩き3走目"
    return f"中{days / 7:.0f}週"


def pace_forecast(styles: pd.Series) -> str:
    """逃げ・先行の数からの目安。逃げ馬がいない/多いでペースが変わる。"""
    known = styles[styles != "不明"]
    if known.empty:
        return "不明"
    n_nige = int((known == "逃げ").sum())
    n_front = n_nige + int((known == "先行").sum())
    if n_nige >= 2 or n_front / len(known) >= 0.45:
        return "ハイペース寄り"
    if n_nige == 0 and n_front / len(known) < 0.25:
        return "スロー寄り"
    return "平均的"


def _record(finishes: pd.Series) -> str:
    f = finishes.dropna()
    return f"{int((f == 1).sum())}-{int((f == 2).sum())}-{int((f == 3).sum())}-{int((f > 3).sum())}"


def _records(target: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """各馬の、今回と同じ条件での過去の着度数。"""
    by_horse = {h: g for h, g in hist.groupby("horse_id", sort=False)}
    rows = []
    for _, t in target.iterrows():
        g = by_horse.get(t["horse_id"])
        if g is None:
            rows.append({k: "0-0-0-0" for k in ("rec_ct", "rec_dist", "rec_dir", "rec_venue", "rec_ground")})
            continue
        rows.append({
            "rec_ct": _record(g.loc[g["course_type"] == t["course_type"], "finish"]),
            "rec_dist": _record(g.loc[(g["distance"] - t["distance"]).abs() <= 200, "finish"]),
            "rec_dir": _record(g.loc[g["direction"] == t["direction"], "finish"]) if isinstance(t["direction"], str)
            else "—",
            "rec_venue": _record(g.loc[g["venue"] == t["venue"], "finish"]),
            "rec_ground": _record(g.loc[g["ground"] == t["ground"], "finish"]),
        })
    return pd.DataFrame(rows, index=target.index)


def _last5(target: pd.DataFrame, hist: pd.DataFrame) -> pd.Series:
    h = hist.sort_values("date", kind="stable")
    recent = {k: "-".join(str(int(x)) for x in g["finish"].dropna().tail(5)[::-1])
              for k, g in h.groupby("horse_id", sort=False)}
    return target["horse_id"].map(recent).fillna("—")


def _jockey_year_wins(hist: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """その年の1月から前日までの騎手別の勝利数と、その順位(同数は同順位)。"""
    y = hist[(hist["date"].dt.year == date.year) & (hist["date"] < date)]
    wins = (y.assign(w=(y["finish"] == 1).astype(int)).groupby("jockey_id")["w"].sum())
    out = pd.DataFrame({"jockey_wins_year": wins})
    out["jockey_rank"] = out["jockey_wins_year"].rank(method="min", ascending=False)
    return out


def _pedigree_names(conn: sqlite3.Connection, horse_ids) -> pd.DataFrame:
    ids = [h for h in set(horse_ids) if isinstance(h, str)]
    if not ids:
        return pd.DataFrame(columns=["horse_id", "sire", "damsire", "sire_line"])
    q = ",".join("?" * len(ids))
    return pd.read_sql_query(
        f"SELECT horse_id, sire, damsire, sire_line FROM pedigree WHERE horse_id IN ({q})", conn, params=ids)


def factor_values(conn: sqlite3.Connection, date: str) -> pd.DataFrame:
    """date の出馬表の各馬について、SECTIONS の列をすべて持つ表を返す(race_id・馬名つき)。"""
    df, hist = day_frame(conn, date)
    t = df[df["finish"].isna()].copy()
    ts = pd.Timestamp(date)

    t["style"] = [style_label(a, b) for a, b in zip(t["early_pos_rel_3"], t["pos_gain_3"])]
    t["pace_forecast"] = t.groupby("race_id")["style"].transform(pace_forecast)
    t = t.join(_records(t, hist)).assign(last5=_last5(t, hist))
    t["rotation"] = [rotation_label(d, r) for d, r in zip(t["days_since_last"], t["runs_since_layoff"])]
    t["jockey_change"] = t["jockey_changed"].map({1.0: "乗り替わり", 0.0: "継続"}).fillna("—")
    t["dist_note"] = np.select([t["dist_shorten300"] == 1, t["dist_extend300"] == 1], ["短縮300m以上", "延長300m以上"], "—")
    yj = _jockey_year_wins(hist, ts)
    t = t.merge(yj, left_on="jockey_id", right_index=True, how="left")
    t["jockey_wins_year"] = t["jockey_wins_year"].fillna(0)  # 今年まだ騎乗がない
    t = t.merge(_pedigree_names(conn, t["horse_id"]), on="horse_id", how="left")
    t["jockey"] = t["jockey"].astype(object)
    t["venue"] = t["venue"].astype(str)
    keep = ["race_id", "venue", "round", "horse_name", *dict.fromkeys(c for s in SECTIONS for c, _ in s["cols"])]
    return t[[c for c in keep if c in t]].sort_values(["race_id", "number"]).reset_index(drop=True)


def section_status(conn: sqlite3.Connection) -> list[dict]:
    """各区分のデータが揃っているか。state は ready / partial / missing。"""
    n_horses = conn.execute("SELECT COUNT(DISTINCT horse_id) FROM results").fetchone()[0]
    n_ped = conn.execute("SELECT COUNT(*) FROM pedigree").fetchone()[0]
    n_flat = conn.execute("SELECT COUNT(*) FROM races WHERE course_type IN ('芝','ダ')").fetchone()[0]
    n_lap = conn.execute("SELECT COUNT(*) FROM races WHERE course_type IN ('芝','ダ') AND pace_first3f IS NOT NULL").fetchone()[0]
    out = []
    for s in SECTIONS:
        name = s["name"]
        if name == "調教":
            state, text = "missing", "未取得(調教データを集める処理がまだありません)"
        elif name == "血統":
            share = n_ped / n_horses if n_horses else 0
            state = "ready" if share >= 0.95 else "partial" if n_ped else "missing"
            text = f"{n_ped:,} / {n_horses:,} 頭ぶん取得済み" if n_ped else "未取得(keibaai pedigree で取得)"
        elif name == "ラップ・ペース":
            share = n_lap / n_flat if n_flat else 0
            state = "ready" if share >= 0.95 else "partial" if n_lap else "missing"
            text = f"{n_lap:,} / {n_flat:,} レース" if n_lap else "未取得(keibaai reparse で読み込み)"
        else:
            state, text = "ready", "取得済み"
        out.append({"name": name, "state": state, "text": text})
    return out
