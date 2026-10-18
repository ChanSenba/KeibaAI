"""回顧用の展開: 発走前の展開予想(予想と一緒に保存)と、結果から出した実際の展開を比べる。

判定は基準に当てはめただけで、理由の推測や講評は付けない。基準の数字は下の定数。外れが続けば見直す。
- ペース差 = 前半3F − 後半3F(秒)。正なら後傾(スロー)、負なら前傾(ハイ)。±PACE_EDGE 秒以内は平均。
- 前の有利 = 序盤の位置が前の3分の1だった馬の3着内率 − それ以外の馬の3着内率。±FRONT_EDGE 以上で前有利・後有利。
  それが基準に届かない(または出せない)ときは、スローなら前有利、ハイなら後有利とする。
- 発走前の予想は、出走馬の脚質(過去の位置取り)で先行しそうな馬の割合 front_share と、同コースの直近5開催日の
  前の有利(course_front_edge_5)から出す。front_share が FRONT_SHARE_HIGH 以上ならハイ、FRONT_SHARE_LOW 以下ならスロー。
""" 
from __future__ import annotations

import datetime as dt
import sqlite3

import pandas as pd

from .features import parse_early_position

PACE_EDGE = 1.0  # ペース差(秒)がこれ以上ならスロー、−これ以下ならハイ
FRONT_EDGE = 0.2  # 前の馬と他の馬の3着内率の差がこれ以上なら有利とみなす
FRONT_SHARE_HIGH = 0.40  # 先行しそうな馬の割合がこれ以上なら、ハイペースの予想
FRONT_SHARE_LOW = 0.20  # これ以下なら、スローペースの予想

ADVANTAGE = {"前": "前有利", "後": "後ろ有利", None: "有利不利なし"}


def position_group(passing, field_size: int) -> str | None:
    """序盤の位置: 前・中・後。通過順位が取れなければ None。"""
    pos = parse_early_position(passing)
    if pos != pos or field_size <= 0:
        return None
    rel = pos / field_size
    return "前" if rel <= 1 / 3 else "後" if rel > 2 / 3 else "中"


def _advantage(edge: float, pace: str) -> tuple[str | None, str]:
    if edge == edge and abs(edge) >= FRONT_EDGE:
        return ("前" if edge > 0 else "後"), f"前に付けた馬の3着内率が他より{abs(edge) * 100:.0f}pt{'高い' if edge > 0 else '低い'}"
    if pace in ("スロー", "ハイ"):
        return ("前" if pace == "スロー" else "後"), f"ペース{pace}"
    return None, "前後の有利不利は出ていない"


def race_flow(runners: pd.DataFrame, first3f, last3f) -> dict:
    """結果から出した、1レースの実際の展開。runners は全出走馬の finish / passing。"""
    n = len(runners)
    g = pd.Series([position_group(p, n) for p in runners["passing"]])
    finish = runners["finish"].reset_index(drop=True)
    top3 = (finish <= 3).astype(float)
    valid = finish.notna() & g.notna()
    front, rest = valid & (g == "前"), valid & (g != "前")
    edge = float(top3[front].mean() - top3[rest].mean()) if front.any() and rest.any() else float("nan")
    ok = first3f is not None and last3f is not None and first3f == first3f and last3f == last3f
    diff = first3f - last3f if ok else float("nan")
    pace = "不明" if diff != diff else "スロー" if diff >= PACE_EDGE else "ハイ" if diff <= -PACE_EDGE else "平均"
    adv, basis = _advantage(edge, pace)
    return {"pace": pace, "pace_diff": diff, "front_edge": edge, "advantage": adv, "basis": basis}


def forecast_flow(entries: pd.DataFrame) -> pd.DataFrame:
    """出馬表の特徴量(race_id, front_share, n_front_runners, course_front_edge_5, course_pace_diff_5)から、
    レースごとの発走前の展開予想を出す。"""
    rows = []
    for rid, g in entries.groupby("race_id", sort=False):
        share = float(g["front_share"].iloc[0])
        edge = float(g["course_front_edge_5"].iloc[0])
        pace = "不明" if share != share else "ハイ" if share >= FRONT_SHARE_HIGH else "スロー" if share <= FRONT_SHARE_LOW else "平均"
        adv, basis = _advantage(edge, pace)
        if share == share:
            basis += f"・先行しそうな馬{int(g['n_front_runners'].iloc[0])}頭({share:.0%})"
        rows.append({"race_id": rid, "n_front": g["n_front_runners"].iloc[0], "front_share": share,
                     "course_edge": edge, "course_pace_diff": float(g["course_pace_diff_5"].iloc[0]),
                     "pace": pace, "advantage": adv, "basis": basis})
    return pd.DataFrame(rows)


def save_forecast(conn: sqlite3.Connection, forecast: pd.DataFrame, race_ids: set[str], now: dt.datetime) -> int:
    """予想を保存したレースの分だけ、展開予想を保存する(予想と同じく、凍結したレースは書き換えない)。"""
    f = forecast[forecast["race_id"].isin(race_ids)]
    cols = ["race_id", "n_front", "front_share", "course_edge", "course_pace_diff", "pace", "advantage", "basis"]
    rows = [(*(None if (isinstance(v, float) and v != v) else v for v in r), now.isoformat(timespec="seconds"))
            for r in f[cols].itertuples(index=False, name=None)]
    with conn:
        conn.executemany(f"INSERT OR REPLACE INTO flow_forecast ({','.join(cols)}, saved_at) VALUES ({','.join('?' * 9)})", rows)
    return len(rows)


def flow_text(actual: dict) -> str:
    pace = "" if actual["pace"] == "不明" else f"・ペース{actual['pace']}(前半3F−後半3F {actual['pace_diff']:+.1f}秒)"
    return f"{ADVANTAGE[actual['advantage']]}({actual['basis']}){pace}"


def forecast_text(f: dict) -> str:
    pace = "" if f["pace"] == "不明" else f"・ペース{f['pace']}"
    return f"{ADVANTAGE[f['advantage']]}({f['basis']}){pace}"
