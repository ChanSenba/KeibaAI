"""各馬が過去に走ったレースと、予想したいレースから見たそれぞれの評価(画面の「過去レース」ページ用)。

結果は対象レースの日より前のものだけを使う。学習はしない。
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

DIST_BAND = 200  # この差までは「ほぼ同じ距離」


def _race(conn: sqlite3.Connection, race_id: str) -> dict:
    row = conn.execute("SELECT race_id, name, date, venue, course_type, direction, distance, ground "
                       "FROM entry_races WHERE race_id = ?", (race_id,)).fetchone()
    if row is None:
        raise ValueError(f"出馬表にないレースです: {race_id}")
    return dict(zip(("race_id", "name", "date", "venue", "course_type", "direction", "distance", "ground"), row))


def _runs(conn: sqlite3.Connection, horse_ids: list[str], before: str) -> pd.DataFrame:
    """馬たちの過去の出走(新しい順)。上がり順位・頭数・通過順の位置も付ける。"""
    if not horse_ids:
        return pd.DataFrame()
    marks = ",".join("?" * len(horse_ids))
    df = pd.read_sql_query(
        f"SELECT r.race_id, r.horse_id, r.number, r.finish, r.status, r.passing, r.last3f, r.margin, r.win_odds, "
        f"r.popularity, r.jockey, r.weight_carried, r.body_weight, r.time_sec, "
        f"a.name AS race_name, a.date, a.venue, a.course_type, a.direction, a.distance, a.ground, "
        f"a.pace_first3f, a.pace_last3f "
        f"FROM results r JOIN races a USING (race_id) WHERE r.horse_id IN ({marks}) AND a.date < ? "
        f"ORDER BY a.date DESC, r.race_id DESC", conn, params=(*horse_ids, before))
    if df.empty:
        return df
    ids = df["race_id"].unique().tolist()
    field = pd.read_sql_query(
        f"SELECT race_id, number, finish, last3f FROM results WHERE race_id IN ({','.join('?' * len(ids))})",
        conn, params=ids)
    field["l3f_rank"] = field.groupby("race_id")["last3f"].rank(method="min")
    size = field.groupby("race_id")["number"].count().rename("field_size")
    df = df.merge(field[["race_id", "number", "l3f_rank"]], on=["race_id", "number"], how="left").merge(
        size, left_on="race_id", right_index=True, how="left")
    return df


def _similarity(run, target: dict) -> tuple[int, list[str], list[str]]:
    """今回との近さの点数と、同じ点・違う点。"""
    same, diff, score = [], [], 0
    if run["course_type"] == target["course_type"]:
        same.append(run["course_type"])
        score += 2
    else:
        diff.append(f"{run['course_type']}(今回は{target['course_type']})")
    d = (run["distance"] or 0) - (target["distance"] or 0)
    if abs(d) <= DIST_BAND:
        same.append("距離" + ("同じ" if d == 0 else f"{d:+d}m"))
        score += 2 if d == 0 else 1
    else:
        diff.append(f"距離{run['distance']}m({'短縮' if d > 0 else '延長'}して今回{target['distance']}m)")
    if run["venue"] == target["venue"]:
        same.append(run["venue"])
        score += 1
    if run["direction"] and run["direction"] == target["direction"]:
        same.append(run["direction"])
        score += 1
    if run["ground"] and run["ground"] == target["ground"]:
        same.append(f"馬場{run['ground']}")
    return score, same, diff


def _result_text(run) -> str:
    if run["finish"] is None or pd.isna(run["finish"]):
        return f"{run['status'] or '不明'}"
    return f"{int(run['finish'])}着/{int(run['field_size'])}頭"


def _comment(run, target: dict, score: int, same: list[str], diff: list[str]) -> tuple[str, str]:
    """この過去レースが今回の目で見てどうか。(評価ラベル, 見解の文)"""
    if run["finish"] is None or pd.isna(run["finish"]):
        return "参考外", f"{run['status'] or '競走中止等'}。今回の判断材料にはしません。"
    size = max(int(run["field_size"] or 0), 1)
    rel = run["finish"] / size
    good = run["finish"] <= 3 or rel <= 0.25
    bad = rel >= 0.6
    parts = [f"{_result_text(run)}"]
    if pd.notna(run["l3f_rank"]):
        parts.append(f"上がり{int(run['l3f_rank'])}位")
    if pd.notna(run["popularity"]) and pd.notna(run["finish"]):
        gap = run["popularity"] - run["finish"]
        if gap >= 3:
            parts.append(f"{int(run['popularity'])}番人気を上回る走り")
        elif gap <= -3:
            parts.append(f"{int(run['popularity'])}番人気を下回る")
    head = "、".join(parts) + "。"
    if score >= 5:
        kind = "直結"
        tail = f"条件が近い({'・'.join(same)})ので、今回に最も参考になる" + ("好走。" if good else "凡走。" if bad else "内容。")
    elif score >= 3:
        kind = "参考"
        tail = f"{'・'.join(same)}は同じ。" + ("今回も通用する材料。" if good else "適性に疑問符。" if bad else "")
        if diff:
            tail += f"ただし{'、'.join(diff)}。"
    else:
        kind = "別条件"
        tail = f"{'、'.join(diff) or '条件がだいぶ違う'}ため、着順ほど今回の評価には使えない。"
        if good:
            tail += "条件が違っても好走している点は能力面でプラス。"
    return kind, head + tail


def _verdict(rows: pd.DataFrame, target: dict) -> str:
    """馬ごとの総評。"""
    if rows.empty:
        return "過去の出走がありません(新馬など)。能力は他のファクターで見ます。"
    done = rows[rows["finish"].notna()]
    close = done[done["score"] >= 5]
    mid = done[done["score"] >= 3]
    parts = [f"過去{len(rows)}走。"]
    if len(close):
        top3 = int((close["finish"] <= 3).sum())
        parts.append(f"今回とほぼ同条件は{len(close)}走で3着内{top3}回(最高{int(close['finish'].min())}着)。")
    elif len(mid):
        top3 = int((mid["finish"] <= 3).sum())
        parts.append(f"同条件に近い走りは{len(mid)}走で3着内{top3}回。ぴったり同じ条件は未経験。")
    else:
        parts.append(f"{target['course_type']}{target['distance']}m前後の経験がなく、適性は未知数。")
    last = rows.iloc[0]
    if pd.notna(last["finish"]):
        parts.append(f"前走は{last['結果']}({last['kind']})。")
    return "".join(parts)


def horse_past_races(conn: sqlite3.Connection, race_id: str, limit: int = 10) -> list[dict]:
    """出馬表の各馬(馬番順)について、{number, horse_name, verdict, runs: DataFrame} を返す。"""
    target = _race(conn, race_id)
    ents = pd.read_sql_query("SELECT number, horse_id, horse_name FROM entries WHERE race_id = ? ORDER BY number",
                             conn, params=(race_id,))
    runs = _runs(conn, [h for h in ents["horse_id"].dropna().unique()], target["date"])
    out = []
    for e in ents.itertuples():
        mine = runs[runs["horse_id"] == e.horse_id].head(limit).copy() if len(runs) else pd.DataFrame()
        recs = []
        for _, run in mine.iterrows():
            score, same, diff = _similarity(run, target)
            kind, text = _comment(run, target, score, same, diff)
            recs.append({"日付": run["date"], "レース": run["race_name"], "場": run["venue"],
                         "条件": f"{run['course_type']}{run['distance']}m {run['ground'] or ''}".strip(),
                         "結果": _result_text(run), "通過": run["passing"] or "", "上がり": run["last3f"],
                         "上がり順": run["l3f_rank"], "人気": run["popularity"], "騎手": run["jockey"],
                         "score": score, "finish": run["finish"], "kind": kind, "見解": text})
        frame = pd.DataFrame(recs)
        out.append({"number": int(e.number), "horse_name": e.horse_name,
                    "verdict": _verdict(frame, target), "runs": frame})
    return out
