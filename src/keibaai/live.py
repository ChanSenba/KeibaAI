"""実運用の記録(発走前に保存した予想)を、結果と市場の動きで測る。

回収率は、市場との差(検証で +10/1000 程度)を確かめるのに約2,800レース(約1年分)が要る。
CLV (Closing Line Value) は結果を見ずに測れる先行指標で、数百レースで傾向が出る:
予想を保存した時点の市場の勝率と、確定オッズの市場の勝率の比の対数。
プラスなら、その馬は保存したあと市場が寄ってきた(オッズが下がった)側で、
「締め切り時点の市場より先に、その見立てに寄っていた」ことを示す。儲けの証明ではない。

市場の勝率はどちらも比例方式(オッズの逆数を正規化)で作る。Shin 方式との比較は devig_trials の担当で、
ここは「保存時点と確定の差」だけを見るので、作り方は両時点で揃える。

注意: 「期待値≧1」の対象は、保存時点のオッズで計算した期待値で選んでいる。その時点のオッズが
高すぎた馬が選ばれやすく、早い時刻のオッズは確定に向けて平均へ戻るため、予想に情報がなくても
この対象の CLV はプラスに寄る。◎(能力のみの勝率で付ける印)はその偏りが小さいので、主にこちらを見る。
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .devig import proportional

GROUPS = ("◎", "期待値≧1", "1番人気", "全馬")


def _is_late(saved_at: str, date: str, start_time: str | None) -> bool:
    """発走より後の保存かどうか。after_result の印がない古い保存も、時刻で判定できる分は判定する。"""
    if not saved_at:
        return True
    if saved_at[:10] > date:
        return True
    if start_time:  # 発走時刻は「9:55」のように時の桁がそろわないことがあるので、0埋めしてから比べる
        h, _, m = start_time.partition(":")
        if saved_at > f"{date}T{h.zfill(2)}:{m}":
            return True
    return False


def race_clv(conn: sqlite3.Connection) -> pd.DataFrame:
    """発走前に保存した予想の、レース×対象ごとの CLV。列は date, race_id, 対象, n, clv(対象馬の平均)。

    対象は ◎(mark)と、期待値≧1(ev_market、なければ ev_ability)の馬。比べる基準として、
    保存時点で1番人気だった馬と、全馬の平均も出す。
    結果のあとに保存した予想(after_result=1 と、保存時刻が発走を過ぎたもの)は除く。
    市場の勝率は、保存時点・確定のどちらも「両方にオッズのある馬」だけで正規化する(取消馬などの扱いを揃える)。
    """
    pred = pd.read_sql_query(
        "SELECT race_id, number, date, saved_at, win_odds, mark, ev_market, ev_ability, after_result FROM predictions", conn)
    if pred.empty:
        return pd.DataFrame(columns=["date", "race_id", "対象", "n", "clv"])
    final = pd.read_sql_query(
        "SELECT r.race_id, r.number, r.win_odds AS final_odds, c.start_time FROM results r JOIN races c USING (race_id) "
        "WHERE r.win_odds > 0", conn)
    by_race: dict[str, dict[int, float]] = {}
    start: dict[str, str | None] = {}
    for r in final.itertuples():
        by_race.setdefault(r.race_id, {})[r.number] = r.final_odds
        start[r.race_id] = r.start_time
    rows = []
    for race_id, g in pred.groupby("race_id"):
        first = g.iloc[0]
        if first["after_result"] == 1 or _is_late(first["saved_at"], first["date"], start.get(race_id)):
            continue
        fmap = by_race.get(race_id, {})
        g = g[g["number"].isin(fmap)]
        odds = pd.to_numeric(g["win_odds"], errors="coerce")
        if len(g) < 3 or not (odds > 0).all():
            continue
        saved_p = proportional(odds.to_numpy())
        final_p = proportional(np.array([fmap[n] for n in g["number"]], dtype=float))
        clv = np.log(final_p / saved_p)
        ev = g["ev_market"].fillna(g["ev_ability"])
        # 基準の行(1番人気・全馬)も一緒に出す。人気馬は締め切りに向けてオッズが下がりやすいので、
        # ◎の CLV は単体でなく、基準との差で読む(基準より上なら、人気という理由だけでは説明できない)
        for name, mask in (("◎", g["mark"] == "◎"), ("期待値≧1", ev >= 1),
                           ("1番人気", odds == odds.min()), ("全馬", pd.Series(True, index=g.index))):
            if mask.any():
                rows.append({"date": first["date"], "race_id": race_id, "対象": name,
                             "n": int(mask.sum()), "clv": float(clv[mask.to_numpy()].mean())})
    return pd.DataFrame(rows, columns=["date", "race_id", "対象", "n", "clv"])


def clv_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """対象ごとの通算 CLV(レース平均)。週次評価の実運用の節に載せる形。"""
    rc = race_clv(conn)
    if rc.empty:
        return pd.DataFrame(columns=["対象", "レース数", "頭数", "平均CLV(/1000)", "プラス割合(%)"])
    out = []
    for name in GROUPS:
        g = rc[rc["対象"] == name]
        if not g.empty:
            out.append({"対象": name, "レース数": len(g), "頭数": int(g["n"].sum()),
                        "平均CLV(/1000)": round(float(g["clv"].mean()) * 1000, 1),
                        "プラス割合(%)": round(float((g["clv"] > 0).mean()) * 100, 1)})
    return pd.DataFrame(out, columns=["対象", "レース数", "頭数", "平均CLV(/1000)", "プラス割合(%)"])
