"""各馬の過去走の評価から、予想したいレースでその馬がどう評価できるかの見込み(画面の「過去レース」ページ用)。

1走ずつ「その走りの評価(0〜100点)」を付け、今回と条件が近い走り・新しい走りほど重く見て平均し、
今回の評価点にする。その点をメンバーの中で比べて、今回の見込み(上位評価〜下位評価)を決める。
結果は対象レースの日より前のものだけを使う。学習はしない。
"""
from __future__ import annotations

import datetime as dt
import sqlite3

import numpy as np
import pandas as pd

from .features import same_course

DIST_BAND = 200  # この差までは「ほぼ同じ距離」
SAME_COURSE_BONUS = 2  # 同コース(同じ競馬場で、芝ダ・内回り外回り・距離が同じ)の走りに足す近さの点数
MAX_SIM = 8  # 近さの点数の最大(芝ダ2・距離2・競馬場1・回り1・同コース2)
RECENCY = 0.85  # 1走さかのぼるごとの重み
PRIOR = 0.6  # 走りが少ない馬の点を50点に寄せる強さ(重み何走ぶんか)
LAYOFF_DAYS = 90  # これ以上あいていれば休み明け


def _race(conn: sqlite3.Connection, race_id: str) -> dict:
    row = conn.execute("SELECT race_id, name, date, venue, course_type, direction, course, distance, ground "
                       "FROM entry_races WHERE race_id = ?", (race_id,)).fetchone()
    if row is None:
        raise ValueError(f"出馬表にないレースです: {race_id}")
    return dict(zip(("race_id", "name", "date", "venue", "course_type", "direction", "course", "distance", "ground"),
                    row))


def _runs(conn: sqlite3.Connection, horse_ids: list[str], before: str) -> pd.DataFrame:
    """馬たちの過去の出走(新しい順)。上がり順位・頭数・勝ち馬とのタイム差も付ける。"""
    if not horse_ids:
        return pd.DataFrame()
    marks = ",".join("?" * len(horse_ids))
    df = pd.read_sql_query(
        f"SELECT r.race_id, r.horse_id, r.number, r.finish, r.status, r.passing, r.last3f, r.margin, r.win_odds, "
        f"r.popularity, r.jockey, r.weight_carried, r.body_weight, r.time_sec, "
        f"a.name AS race_name, a.date, a.venue, a.course_type, a.direction, a.course, a.distance, a.ground, "
        f"a.pace_first3f, a.pace_last3f "
        f"FROM results r JOIN races a USING (race_id) WHERE r.horse_id IN ({marks}) AND a.date < ? "
        f"ORDER BY a.date DESC, r.race_id DESC", conn, params=(*horse_ids, before))
    if df.empty:
        return df
    ids = df["race_id"].unique().tolist()
    field = pd.read_sql_query(
        f"SELECT race_id, number, finish, last3f, time_sec FROM results WHERE race_id IN ({','.join('?' * len(ids))})",
        conn, params=ids)
    field["l3f_rank"] = field.groupby("race_id")["last3f"].rank(method="min")
    size = field.groupby("race_id")["number"].count().rename("field_size")
    best = field.groupby("race_id")["time_sec"].min().rename("winner_time")
    df = df.merge(field[["race_id", "number", "l3f_rank"]], on=["race_id", "number"], how="left").merge(
        size, left_on="race_id", right_index=True, how="left").merge(best, left_on="race_id", right_index=True, how="left")
    df["time_gap"] = df["time_sec"] - df["winner_time"]
    return df


def _similarity(run, target: dict) -> tuple[int, list[str], list[str]]:
    """今回との近さの点数(0〜MAX_SIM)と、同じ点・違う点。

    同コースは、同じ競馬場で芝ダ・内回り外回り・距離が同じ走り。距離だけ同じなら「同じ距離」の経験とする。
    """
    if same_course(run, target):
        same = ["同コース", f"{target['venue']}{target['course_type']}{target['distance']}m"]
        if run["ground"] and run["ground"] == target["ground"]:
            same.append(f"馬場{run['ground']}")
        return MAX_SIM, same, []
    same, diff, score = [], [], 0
    if run["course_type"] == target["course_type"]:
        same.append(run["course_type"])
        score += 2
    else:
        diff.append(f"{run['course_type']}(今回は{target['course_type']})")
    d = (run["distance"] or 0) - (target["distance"] or 0)
    if abs(d) <= DIST_BAND:
        same.append("同じ距離" if d == 0 else f"距離{d:+d}m")
        score += 2 if d == 0 else 1
    else:
        diff.append(f"距離{run['distance']}m(今回は{target['distance']}m)")
    if run["venue"] == target["venue"]:
        same.append(run["venue"])
        score += 1
    if run["direction"] and run["direction"] == target["direction"]:
        same.append(run["direction"])
        score += 1
    if run["ground"] and run["ground"] == target["ground"]:
        same.append(f"馬場{run['ground']}")
    return score, same, diff


def _kind(run, target: dict, score: int) -> str:
    """今回との近さの区分。同コース > 同じ距離(芝ダと距離が同じで、競馬場かコースが違う) > 参考 > 別条件。"""
    if score >= MAX_SIM:
        return "同コース"
    if run["course_type"] == target["course_type"] and run["distance"] == target["distance"]:
        return "同じ距離"
    return "参考" if score >= 3 else "別条件"


def _result_text(run) -> str:
    if run["finish"] is None or pd.isna(run["finish"]):
        return f"{run['status'] or '不明'}"
    return f"{int(run['finish'])}着/{int(run['field_size'])}頭"


def run_rating(run) -> float | None:
    """その走りそのものの評価(0〜100点)。着順の位置 5割・勝ち馬とのタイム差 3割・上がり順位 2割。"""
    if run["finish"] is None or pd.isna(run["finish"]):
        return None
    size = max(int(run["field_size"] or 0), 2)
    pos = 1 - (run["finish"] - 1) / (size - 1)
    parts = [(pos, 0.5)]
    if pd.notna(run.get("time_gap")):
        parts.append((float(np.clip(1 - run["time_gap"] / 2.0, 0, 1)), 0.3))  # 勝ち馬から2秒以上の差で0
    if pd.notna(run.get("l3f_rank")):
        parts.append((1 - (run["l3f_rank"] - 1) / (size - 1), 0.2))
    w = sum(p[1] for p in parts)
    return round(100 * float(np.clip(sum(v * k for v, k in parts) / w, 0, 1)), 1)


def _grade(rating: float) -> str:
    return "好走" if rating >= 70 else "善戦" if rating >= 50 else "凡走"


def _run_comment(run, target: dict, rating: float | None, kind: str, same: list[str], diff: list[str],
                 share: float) -> str:
    """1走ずつの見解:その走りの評価と、今回の見込みにどれだけ効いているか。"""
    if rating is None:
        return f"{run['status'] or '競走中止等'}。今回の見込みには入れていません。"
    parts = [_result_text(run)]
    if pd.notna(run["time_gap"]) and run["finish"] > 1:
        parts.append(f"勝ち馬と{run['time_gap']:.1f}秒差")
    if pd.notna(run["l3f_rank"]):
        parts.append(f"上がり{int(run['l3f_rank'])}位")
    if pd.notna(run["popularity"]):
        gap = run["popularity"] - run["finish"]
        if gap >= 3:
            parts.append(f"{int(run['popularity'])}番人気を上回る")
        elif gap <= -3:
            parts.append(f"{int(run['popularity'])}番人気を下回る")
    head = "、".join(parts) + f"で、この走りは{_grade(rating)}({rating:.0f}点)。"
    weight = "重く" if share >= 0.25 else "やや重く" if share >= 0.12 else "軽く"
    if kind == "同コース":
        why = f"今回と同コース({'・'.join(same[1:])})なので"
    elif kind == "同じ距離":
        other = "競馬場が違う" if run["venue"] != target["venue"] else "内回り・外回りが違う"
        why = f"今回と同じ距離での経験({run['venue']}{run['course_type']}{run['distance']}m。{other}ので同コースではない)で"
    elif kind == "参考":
        why = f"{'・'.join(same)}が同じで" + (f"、{'、'.join(diff)}が違うので" if diff else "")
    else:
        why = f"{'、'.join(diff) or '条件がだいぶ違う'}ので"
    return head + f"{why}、今回の見込みには{weight}入れています(重み{share:.0%})。"


def _horse_runs(mine: pd.DataFrame, target: dict) -> pd.DataFrame:
    """1頭ぶんの過去走に、評価・近さ・今回への重みと見解を付けた表。"""
    recs = []
    for i, (_, run) in enumerate(mine.iterrows()):
        score, same, diff = _similarity(run, target)
        rating = run_rating(run)
        weight = 0.0 if rating is None else (1 + score) / (1 + MAX_SIM) * RECENCY ** i
        recs.append({"日付": run["date"], "レース": run["race_name"] or f"{run['venue']}のレース", "場": run["venue"],
                     "条件": f"{run['course_type']}{run['distance']}m {run['ground'] or ''}".strip(),
                     "結果": _result_text(run), "通過": run["passing"] or "", "上がり": run["last3f"],
                     "上がり順": run["l3f_rank"], "人気": run["popularity"], "騎手": run["jockey"],
                     "score": score, "finish": run["finish"], "distance": run["distance"],
                     "course_type": run["course_type"], "kind": _kind(run, target, score), "評価": rating, "weight": weight,
                     "_same": same, "_diff": diff, "_run": run})
    frame = pd.DataFrame(recs)
    if frame.empty:
        return frame
    total = frame["weight"].sum()
    frame["重み"] = frame["weight"] / total if total > 0 else 0.0
    frame["見解"] = [_run_comment(r["_run"], target, r["評価"], r["kind"], r["_same"], r["_diff"], r["重み"])
                   for _, r in frame.iterrows()]
    return frame.drop(columns=["_same", "_diff", "_run"])


def _expected(frame: pd.DataFrame) -> float | None:
    """今回の評価点。重みつき平均を、走りが少ないほど50点に寄せる。"""
    if frame.empty or frame["weight"].sum() == 0:
        return None
    done = frame[frame["weight"] > 0]
    return round(float((done["weight"] * done["評価"]).sum() + 50 * PRIOR) / float(done["weight"].sum() + PRIOR), 1)


def _confidence(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "—"
    done = frame[frame["評価"].notna()]
    if (done["kind"] == "同コース").sum() >= 2:
        return "高"
    if (done["kind"] != "別条件").sum() >= 1:
        return "中"
    return "低"


def _head_to_head(runs: pd.DataFrame, ents: pd.DataFrame) -> dict[str, dict]:
    """今回の出走馬どうしが過去に同じレースで走ったときの先着数。{horse_id: {wins, losses, detail}}"""
    out = {h: {"wins": 0, "losses": 0, "detail": {}} for h in ents["horse_id"].dropna()}
    if runs.empty:
        return out
    names = dict(zip(ents["horse_id"], zip(ents["number"], ents["horse_name"])))
    done = runs[runs["finish"].notna()][["race_id", "horse_id", "finish"]]
    pairs = done.merge(done, on="race_id", suffixes=("", "_o"))
    pairs = pairs[pairs["horse_id"] != pairs["horse_id_o"]]
    for p in pairs.itertuples():
        me, rec = out[p.horse_id], out[p.horse_id]["detail"].setdefault(p.horse_id_o, [0, 0])
        if p.finish < p.finish_o:
            me["wins"] += 1
            rec[0] += 1
        else:
            me["losses"] += 1
            rec[1] += 1
    for h, v in out.items():
        v["detail"] = {f"{names[o][0]} {names[o][1]}": wl for o, wl in v["detail"].items()}
    return out


def _h2h_text(h2h: dict) -> str:
    if not h2h["detail"]:
        return ""
    best = sorted(h2h["detail"].items(), key=lambda kv: (kv[1][1] - kv[1][0], -kv[1][0]))
    note = "、".join(f"{k}に{w}勝{l}敗" for k, (w, l) in best[:3])
    return f"{h2h['wins']}勝{h2h['losses']}敗({note})"


def _cautions(frame: pd.DataFrame, target: dict) -> tuple[list[str], list[str]]:
    """今回の条件と過去走の違いから、上積みの材料と不安な点。"""
    plus, minus = [], []
    done = frame[frame["評価"].notna()]
    if done.empty:
        return plus, minus
    last = frame.iloc[0]
    days = (dt.date.fromisoformat(target["date"]) - dt.date.fromisoformat(last["日付"])).days
    if days >= LAYOFF_DAYS:
        minus.append(f"休み明け({days}日ぶり)")
    if last["course_type"] != target["course_type"]:
        if not (done["course_type"] == target["course_type"]).any():
            minus.append(f"{target['course_type']}は初めて")
        else:
            minus.append(f"前走{last['course_type']}から{target['course_type']}に替わる")
    d = (target["distance"] or 0) - (last["distance"] or 0)
    if abs(d) > DIST_BAND:
        near = done[(done["distance"] - target["distance"]).abs() <= DIST_BAND]
        change = f"前走から距離{'延長' if d > 0 else '短縮'}({d:+d}m)"
        if near.empty:
            minus.append(change + f"で、{target['distance']}m前後は未経験")
        elif near["評価"].max() >= 70:
            plus.append(change + f"だが、{target['distance']}m前後で好走歴あり")
        else:
            minus.append(change)
    recent = done.head(3)["評価"].tolist()
    if len(recent) == 3:
        if recent[0] > recent[1] > recent[2] and recent[0] - recent[2] >= 10:
            plus.append("直近3走で評価が上向き")
        elif recent[0] < recent[1] < recent[2] and recent[2] - recent[0] >= 10:
            minus.append("直近3走で評価が下向き")
    course = done[done["kind"] == "同コース"]
    dist = done[done["kind"] == "同じ距離"]
    if len(course) and course["評価"].max() >= 70:
        plus.append("同コースで好走歴あり")
    elif len(dist) and dist["評価"].max() >= 70:
        plus.append(f"同じ距離({target['course_type']}{target['distance']}m)で好走歴あり(同コースではない)")
    return plus, minus


def _outlook(rank: int | None, n: int) -> str:
    if rank is None:
        return "評価できず"
    if rank == 1:
        return "最上位評価"
    if rank <= 3:
        return "上位評価"
    if rank <= max(4, (n + 1) // 2):
        return "中位評価"
    return "下位評価"


def _outlook_text(h: dict, n: int) -> str:
    """馬ごとの見解:過去走の評価から、今回のレースでどう評価できるか。"""
    frame = h["runs"]
    if h["rating"] is None:
        if frame.empty:
            return "過去の出走がありません(新馬など)。過去走からは今回の評価を出せないので、血統や調教など他のファクターで見ます。"
        return "完走した過去走がなく、今回の評価を出せません。"
    parts = [f"今回の見込みは**{h['outlook']}**(メンバー{n}頭中{h['rank']}位・{h['rating']:.0f}点、信頼度{h['confidence']})。"]
    done = frame[frame["評価"].notna()]
    top = done.sort_values("重み", ascending=False).iloc[0]
    near = done[done["kind"] != "別条件"]
    if len(near):
        good = int((near["評価"] >= 70).sum())
        n_course = int((near["kind"] == "同コース").sum())
        parts.append(f"今回に近い条件の走りは{len(near)}走(うち同コース{n_course}走)で、そのうち好走{good}回。")
    else:
        parts.append("今回に近い条件の走りがなく、条件の違う走りから推しているので当てになりにくい見込みです。")
    parts.append(f"いちばん重く見たのは{top['日付']} {top['レース']}({top['条件']})の{top['結果']}で、"
                 f"{_grade(top['評価'])}({top['評価']:.0f}点)。")
    if h["h2h"]:
        parts.append(f"今回の相手との過去の対戦は{h['h2h']}。")
    if h["plus"]:
        parts.append("上積み:" + "、".join(h["plus"]) + "。")
    if h["minus"]:
        parts.append("不安:" + "、".join(h["minus"]) + "。")
    return "".join(parts)


def horse_past_races(conn: sqlite3.Connection, race_id: str, limit: int = 10) -> list[dict]:
    """出馬表の各馬(馬番順)について、過去走と、そこから見た今回の見込みを返す。

    {number, horse_name, rating(今回の評価点), rank(メンバー中の順位), outlook(見込み), confidence(信頼度),
     h2h(今回の相手との対戦), plus, minus, verdict(見解の文), runs(過去走の表)}
    """
    target = _race(conn, race_id)
    ents = pd.read_sql_query("SELECT number, horse_id, horse_name FROM entries WHERE race_id = ? ORDER BY number",
                             conn, params=(race_id,))
    runs = _runs(conn, [h for h in ents["horse_id"].dropna().unique()], target["date"])
    h2h = _head_to_head(runs, ents)
    out = []
    for e in ents.itertuples():
        mine = runs[runs["horse_id"] == e.horse_id].head(limit) if len(runs) else pd.DataFrame()
        frame = _horse_runs(mine, target)
        plus, minus = _cautions(frame, target) if len(frame) else ([], [])
        out.append({"number": int(e.number), "horse_name": e.horse_name, "rating": _expected(frame),
                    "confidence": _confidence(frame), "h2h": _h2h_text(h2h.get(e.horse_id, {"detail": {}})),
                    "plus": plus, "minus": minus, "runs": frame})
    rated = sorted((h for h in out if h["rating"] is not None), key=lambda h: -h["rating"])
    for i, h in enumerate(rated, start=1):
        h["rank"] = i
    for h in out:
        h.setdefault("rank", None)
        h["outlook"] = _outlook(h["rank"], len(out))
        h["verdict"] = _outlook_text(h, len(out))
    return out
