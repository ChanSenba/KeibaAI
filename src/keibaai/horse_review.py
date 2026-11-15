"""全頭の回顧: 予想した全馬について、発走前の見立て(印・勝率・市場比・強かった要素と弱かった要素)と、
結果(着順・通過順・上がり)、予想と結果のずれ、展開の読みとのずれを、定型の短い文にする。

要素は予想のニューラルネットの要素ごとの勝率(factors.COLUMN の p_win_*。予想と一緒に保存)を、同じレースの中で
順位にしたもの。強み・弱みは順位で決めるだけで、勝ち負けの原因とは断定しない。
"""
from __future__ import annotations

import glob
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .factors import COLUMN
from .flow_review import ADVANTAGE, position_group

STRONG_SHARE, WEAK_SHARE = 1 / 3, 2 / 3  # レースの中でこの割合より上の順位なら強み、下なら弱み
N_STRONG, N_WEAK = 3, 2  # 文に出す強み・弱みの数


def _num(v) -> float:
    v = pd.to_numeric(v, errors="coerce")
    return float(v) if v == v and v is not None else float("nan")


def factor_ranks(g: pd.DataFrame) -> pd.DataFrame:
    """要素ごとの、レースの中の順位(1が最上位)。値が全頭そろって無い要素は列を作らない。"""
    out = pd.DataFrame(index=g.index)
    for name, col in COLUMN.items():
        if col in g and pd.to_numeric(g[col], errors="coerce").notna().any():
            out[name] = pd.to_numeric(g[col], errors="coerce").rank(ascending=False, method="min")
    return out


def strengths(ranks: pd.Series, n: int) -> tuple[list[str], list[str]]:
    """1頭の要素の順位から、(強み, 弱み)の「騎手・厩舎1位」の並び。"""
    r = ranks.dropna().sort_values(kind="stable")
    strong = [f"{k}{int(v)}位" for k, v in r.items() if v <= max(1.0, np.ceil(n * STRONG_SHARE))][:N_STRONG]
    weak = [f"{k}{int(v)}位" for k, v in r.sort_values(ascending=False, kind="stable").items() if v > n * WEAK_SHARE][:N_WEAK]
    return strong, weak


def market_probs(g: pd.DataFrame) -> pd.Series:
    """確定の単勝オッズ(なければ予想時のオッズ)から出した市場の勝率(レースの中で合計1)。"""
    odds = pd.to_numeric(g["final_odds"] if "final_odds" in g else g["win_odds"], errors="coerce")
    if "win_odds" in g:
        odds = odds.fillna(pd.to_numeric(g["win_odds"], errors="coerce"))
    inv = (1.0 / odds).where(odds > 0)
    return inv / inv.sum() if inv.notna().any() else inv


def gap_text(rank: float, finish: float, n: int, status=None) -> str:
    """予想の順位と着順のずれ。"""
    if finish != finish:
        return f"{status if isinstance(status, str) and status else '着順なし'}で、比べられない"
    r, f = int(rank), int(finish)
    if r <= 3 and f <= 3:
        return f"予想{r}位で{f}着と、予想どおり好走"
    if r <= 3:
        return f"予想{r}位から{f}着と、予想を下回った"
    if f <= 3:
        return f"予想{r}位から{f}着と、予想を上回った"
    if f - r >= n / 3:
        return f"予想{r}位から{f}着と、予想を下回った"
    if r - f >= n / 3:
        return f"予想{r}位から{f}着と、予想より走った"
    return f"予想{r}位で{f}着と、ほぼ予想どおり"


def flow_gap(fc: dict | None, act: dict | None, grp: str | None) -> str:
    """展開の読みと実際のずれを、その馬の序盤の位置(前・中・後)と合わせて書く。"""
    if act is None:
        return ""
    real = act.get("advantage")
    where = f"序盤は{grp}" if grp else "位置は不明"
    if fc is None:
        return f"展開は{ADVANTAGE[real]}(読みの保存なし)、{where}"
    read = fc.get("advantage")
    if read == real:
        if real is None:
            return f"展開の読み(有利不利なし)どおり、{where}"
        side = "で、有利な側にいた" if grp == real else "で、不利な側にいた" if grp in ("前", "後") else ""
        return f"展開の読み({ADVANTAGE[real]})どおり{side or '、' + where}"
    text = f"展開は{ADVANTAGE[read]}の読みが外れて{ADVANTAGE[real]}"
    if grp is not None and grp == read:
        return text + "で、読みに合う位置取りが裏目"
    if grp is not None and grp == real:
        return text + "で、実際の展開には合う位置だった"
    return text + f"、{where}"


def race_rows(g: pd.DataFrame, flow: tuple | None = None) -> dict:
    """1レースの全頭の回顧。g は weekly.review_day の horses のそのレースの行、flow は weekly._flows の
    (展開予想, 実際の展開, 上がり順位)。{"summary": レースのまとめの文, "rows": [馬ごとの dict(着順の順)]}。"""
    from .race_review import _tag_frame, tags_of, GOOD_TAGS

    g = g.copy()
    n = len(g)
    fc, act = (flow[0], flow[1]) if flow else (None, None)
    for c in ("finish", "last3f", "p_win_ability", "final_pop", "number"):
        if c in g:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    g["_rank"] = g["p_win_ability"].rank(ascending=False, method="first")
    g["_mkt"] = market_probs(g)
    g["_l3"] = g["last3f"].rank(method="min")
    ranks = factor_ranks(g)
    tags = _tag_frame(g.assign(race_id=g["race_id"]))[["flow_fit", "gate_fit", "closer_unlucky"]]
    rows = []
    for i, x in g.sort_values(["finish", "number"], na_position="last").iterrows():
        pa, pm, fin = _num(x["p_win_ability"]), _num(x["_mkt"]), _num(x["finish"])
        ratio = pa / pm if pm == pm and pm > 0 and pa == pa else float("nan")
        strong, weak = strengths(ranks.loc[i], n) if len(ranks.columns) else ([], [])
        grp = position_group(x.get("passing"), n)
        mark = x["mark"] if isinstance(x.get("mark"), str) else ""
        pop = _num(x.get("final_pop"))
        head = f"{mark}{int(x['number'])} {x['horse_name']}" + (f"({int(pop)}番人気)" if pop == pop else "")
        parts = []
        if pa == pa:
            parts.append(f"予想{int(x['_rank'])}位・勝率{pa:.1%}" + (f"(市場の{ratio:.2f}倍)" if ratio == ratio else ""))
        if strong or weak:
            parts.append(" / ".join(filter(None, [f"強み {'、'.join(strong)}" if strong else "",
                                                  f"弱み {'、'.join(weak)}" if weak else ""])))
        elif len(ranks.columns):
            parts.append("要素はどれも中位")
        else:
            parts.append("要素の記録なし")
        res = f"結果 {int(fin)}着" if fin == fin else f"結果 {x.get('status') or '着順なし'}"
        bits = []
        if isinstance(x.get("passing"), str) and x["passing"]:
            bits.append(f"通過 {x['passing']}")
        if x["last3f"] == x["last3f"]:
            bits.append(f"上がり{x['last3f']:.1f}秒({int(x['_l3'])}位)")
        parts.append(res + (f"({'・'.join(bits)})" if bits else ""))
        gap = gap_text(x["_rank"], fin, n, x.get("status")) if pa == pa else ""
        if gap:
            parts.append(gap)
        fg = flow_gap(fc, act, grp) if fin == fin else ""
        if fg:
            parts.append(fg)
        t = tags_of(tags.loc[i]) if fin == fin else []
        good, bad = [s for s in t if s in GOOD_TAGS], [s for s in t if s not in GOOD_TAGS]
        if fin == fin and fin <= 3 and good:
            parts.append(f"勝因の候補 {'・'.join(good)}")
        if fin == fin and fin > 3 and bad:
            parts.append(f"敗因の候補 {'・'.join(bad)}")
        rows.append({"finish": fin, "mark": mark, "number": int(x["number"]), "horse_name": x["horse_name"],
                     "popularity": pop, "pred_rank": _num(x["_rank"]), "p_win": pa, "market_ratio": ratio,
                     "strong": "、".join(strong), "weak": "、".join(weak), "passing": x.get("passing"),
                     "last3f": _num(x["last3f"]), "gap": gap, "flow": fg,
                     "text": f"{head}: " + "。".join(parts) + "。"})
    summary = race_summary(rows, fc, act)
    if "factors_later" in g and (pd.to_numeric(g["factors_later"], errors="coerce") == 1).any():
        summary += "強み・弱みは、保存後に予想し直したときの要素の値(勝率の差は1pt以内)。"
    return {"summary": summary, "rows": rows}


def race_summary(rows: list[dict], fc: dict | None, act: dict | None) -> str:
    """レースの予想と結果のずれのまとめ(1〜2文)。"""
    ran = [r for r in rows if r["finish"] == r["finish"] and r["pred_rank"] == r["pred_rank"]]
    if not ran:
        return ""
    top = [r for r in ran if r["pred_rank"] <= 3]
    hit = sum(r["finish"] <= 3 for r in top)
    up = [f"{r['horse_name']}(予想{int(r['pred_rank'])}位→{int(r['finish'])}着)" for r in ran
          if r["finish"] <= 3 and r["pred_rank"] > 3]
    down = [f"{r['horse_name']}(予想{int(r['pred_rank'])}位→{int(r['finish'])}着)" for r in ran
            if r["pred_rank"] <= 3 and r["finish"] > 3]
    text = f"予想上位3頭のうち3着以内は{hit}頭。"
    if up:
        text += "予想を上回って3着内: " + "、".join(up) + "。"
    if down:
        text += "予想上位で崩れた: " + "、".join(down) + "。"
    if act is not None and fc is not None:
        same = fc.get("advantage") == act.get("advantage")
        text += (f"展開の読み({ADVANTAGE[fc.get('advantage')]})は" + ("当たり。" if same else f"外れ(実際は{ADVANTAGE[act.get('advantage')]})。"))
    return text


def markdown_lines(full: dict, indent: str = "  ") -> list[str]:
    """回顧ファイル用の行。"""
    if not full or not full.get("rows"):
        return []
    out = [f"{indent}- 全頭の振り返り: {full['summary']}"]
    out += [f"{indent}  - {r['text']}" for r in full["rows"]]
    return out


# ---------------------------------------------------------------- 要素の記録がない古い予想の埋め戻し
FACTOR_COLS = list(COLUMN.values())
LATER_TOL = 0.01  # 予想し直した値を使ってよい、能力の勝率の差の上限(レースの全頭で)


def backfill_factors(conn: sqlite3.Connection, cache_dir: str | Path) -> dict[str, int]:
    """要素の列が空の保存済み予想に、予想の保存(model_cache の predict-<日付>-*.pkl)から要素の勝率を入れる。

    保存した予想と能力の勝率が一致する馬は、そのまま入れる。オッズや馬体重を取り直して予想し直したレースは、
    全頭の勝率の差が LATER_TOL 以内なら入れて factors_later=1 にする(回顧で、あとの予想の値だと断る)。
    それより離れたレース(モデルを変える前の予想など)は入れない。{日付: 入れた頭数}。
    """
    out = {}
    days = [r[0] for r in conn.execute("SELECT DISTINCT date FROM predictions WHERE p_win_human IS NULL")]
    for d in days:
        files = glob.glob(str(Path(cache_dir) / f"predict-{d}-*.pkl"))
        if not files:
            continue
        try:
            with open(files[0], "rb") as f:
                t = pickle.load(f)
        except Exception:
            continue
        if not set(FACTOR_COLS) <= set(t.columns):
            continue
        p = pd.read_sql_query("SELECT race_id, number, p_win_ability FROM predictions WHERE date = ? AND p_win_human IS NULL",
                              conn, params=(d,))
        m = p.merge(t[["race_id", "number", "p_win_ability", *FACTOR_COLS]], on=["race_id", "number"], suffixes=("", "_c"))
        diff = (m["p_win_ability"] - m["p_win_ability_c"]).abs()
        m["later"] = (diff >= 1e-9).astype(int)
        race_max = diff.groupby(m["race_id"]).transform("max")
        m = m[(diff < 1e-9) | (race_max <= LATER_TOL)]
        rows = [(*(None if v != v else float(v) for v in r[3:]), int(r[2]), r[0], int(r[1]))
                for r in m[["race_id", "number", "later", *FACTOR_COLS]].itertuples(index=False, name=None)]
        with conn:
            conn.executemany(f"UPDATE predictions SET {', '.join(f'{c} = ?' for c in FACTOR_COLS)}, factors_later = ? "
                             "WHERE race_id = ? AND number = ?", rows)
        if rows:
            out[d] = len(rows)
    return out
