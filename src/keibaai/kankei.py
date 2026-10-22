"""騎手・厩舎・馬主のつながり(相関図)。results の騎乗数から、両方向のシェアと順位を出す。

中心が騎手なら、内側の輪は騎乗の多い厩舎、外側の輪は馬主。中心が調教師なら、内側は騎手。
シェアは数える側で値が変わる。たとえば騎手 A と厩舎 B の組が 50 回なら、
「A の全騎乗のうち B の馬が何%か」と「B の全出走のうち A が乗ったのが何%か」の2つを出す。
"""
from __future__ import annotations

import base64
import datetime as dt
import html
import math
import sqlite3

import pandas as pd

KINDS = {"jockey": ("jockey_id", "jockey", "騎手"), "trainer": ("trainer_id", "trainer", "調教師")}
OTHER = {"jockey": "trainer", "trainer": "jockey"}
MIN_RIDES = 30  # 一覧に出す騎手の、期間中の騎乗数の下限

# 馬主の系統。名前に含まれる語で分ける目安(法人の表記ゆれは含む語で吸収する)
OWNER_GROUPS = [
    ("ノーザンF系", ("サンデーレーシング", "シルクレーシング", "キャロットファーム", "ノーザンファーム",
                  "吉田勝己", "吉田和美")),
    ("社台系", ("社台レースホース", "社台ファーム", "社台グループ", "吉田照哉", "吉田千津", "白老ファーム")),
    ("マイネル系", ("ラフィアン", "ビッグレッドファーム", "コスモヴューファーム", "ウイン")),
]
GROUP_ORDER = [g for g, _ in OWNER_GROUPS] + ["その他"]


def owner_group(name) -> str:
    if not isinstance(name, str):
        return "その他"
    for group, words in OWNER_GROUPS:
        if any(w in name for w in words):
            return group
    return "その他"


def load_runs(conn: sqlite3.Connection, end: str, years: int = 3) -> pd.DataFrame:
    """end までの years 年の出走(取消・除外は除く)。"""
    end_d = dt.date.fromisoformat(end)
    start = end_d.replace(year=end_d.year - years) + dt.timedelta(days=1)
    df = pd.read_sql_query(
        """SELECT c.date, r.race_id, r.finish, r.jockey_id, r.jockey, r.trainer_id, r.trainer, r.owner
           FROM results r JOIN races c ON c.race_id = r.race_id
           WHERE c.date BETWEEN ? AND ? AND COALESCE(r.status, '') NOT IN ('取', '除')""",
        conn, params=(start.isoformat(), end))
    df["owner"] = df["owner"].where(df["owner"].notna() & (df["owner"].astype(str).str.strip() != ""))
    return df


def _names(runs: pd.DataFrame, id_col: str, name_col: str) -> pd.Series:
    """番号ごとの、いちばん新しい名前。"""
    named = runs.dropna(subset=[id_col, name_col]).sort_values("date")
    return named.groupby(id_col)[name_col].last()


def people(runs: pd.DataFrame, kind: str, end: str) -> pd.DataFrame:
    """一覧に出す騎手(期間中 MIN_RIDES 騎乗以上で、直近1年にも騎乗あり)か調教師(直近1年に出走あり)。"""
    id_col, name_col, _ = KINDS[kind]
    recent = (dt.date.fromisoformat(end) - dt.timedelta(days=365)).isoformat()
    g = runs.dropna(subset=[id_col]).groupby(id_col)
    out = pd.DataFrame({"n": g.size(), "last": g["date"].max()})
    out = out[out["last"] >= recent]
    if kind == "jockey":
        out = out[out["n"] >= MIN_RIDES]
    out["name"] = _names(runs, id_col, name_col).reindex(out.index).fillna(out.index.to_series())
    return out.reset_index().rename(columns={id_col: "id"}).sort_values("n", ascending=False)[["id", "name", "n"]]


def _pair_table(runs: pd.DataFrame, center_col: str, partner_col: str, center_id) -> pd.DataFrame:
    """中心と相手の組ごとの回数と、両方向のシェア・順位。"""
    runs = runs.dropna(subset=[center_col, partner_col])
    pair = runs.groupby([partner_col, center_col]).size().rename("n").reset_index()
    # 相手から見た中心の順位(相手の全出走のうち、何番目に多い組か)
    pair["rank_partner"] = pair.groupby(partner_col)["n"].rank(method="min", ascending=False).astype(int)
    partner_total = pair.groupby(partner_col)["n"].sum()
    mine = pair[pair[center_col] == center_id].set_index(partner_col)
    if mine.empty:
        return pd.DataFrame(columns=["n", "share_center", "rank_center", "share_partner", "rank_partner", "top"])
    total = runs[runs[center_col] == center_id].shape[0]
    res = runs[runs[center_col] == center_id].groupby(partner_col)["finish"]
    out = pd.DataFrame({
        "n": mine["n"],
        "share_center": mine["n"] / total,
        "rank_center": mine["n"].rank(method="min", ascending=False).astype(int),
        "share_partner": mine["n"] / partner_total.reindex(mine.index),
        "rank_partner": mine["rank_partner"],
        "wins": res.apply(lambda f: int((f == 1).sum())).reindex(mine.index),
        "top3_rate": res.apply(lambda f: float(f.between(1, 3).mean())).reindex(mine.index),
    })
    out["top"] = out["rank_partner"] == 1  # 相手から見て、中心が1位(相関図の金の輪)
    return out.sort_values(["n", "share_partner"], ascending=False)


def relations(runs: pd.DataFrame, kind: str, center_id, n_inner: int = 20, n_outer: int = 16) -> dict:
    """中心(騎手か調教師)の相関図の中身。

    inner: 内側の輪(騎手なら厩舎、調教師なら騎手)。outer: 外側の輪(馬主)。
    links: 中心の出走のうち、内側の相手と馬主の組ごとの回数(図の細い線)。
    """
    id_col, name_col, _ = KINDS[kind]
    pid_col, pname_col, _ = KINDS[OTHER[kind]]
    mine = runs[runs[id_col] == center_id]
    names = _names(runs, id_col, name_col)
    inner = _pair_table(runs, id_col, pid_col, center_id).head(n_inner)
    inner["name"] = _names(runs, pid_col, pname_col).reindex(inner.index).fillna(inner.index.to_series())
    outer = _pair_table(runs, id_col, "owner", center_id).head(n_outer)
    outer["name"] = outer.index
    outer["group"] = [owner_group(o) for o in outer.index]
    links = (mine[mine[pid_col].isin(inner.index) & mine["owner"].isin(outer.index)]
             .groupby([pid_col, "owner"]).size().rename("n").reset_index()
             .rename(columns={pid_col: "partner"}))
    return {
        "kind": kind, "id": center_id, "name": names.get(center_id, str(center_id)),
        "n": len(mine), "wins": int((mine["finish"] == 1).sum()),
        "top3_rate": float(mine["finish"].between(1, 3).mean()) if len(mine) else float("nan"),
        "inner": inner, "outer": outer, "links": links,
    }


def breakdown(runs: pd.DataFrame, rel: dict, key: tuple[str, str], limit: int = 6) -> pd.DataFrame:
    """選んだ相手との出走の内訳。内側の相手なら馬主別、馬主なら内側の相手別の回数。"""
    id_col = KINDS[rel["kind"]][0]
    pid_col, pname_col, _ = KINDS[OTHER[rel["kind"]]]
    mine = runs[runs[id_col] == rel["id"]]
    ring, value = key
    if ring == "inner":
        sub, by = mine[mine[pid_col] == value], "owner"
        label = sub[by]
    else:
        sub, by = mine[mine["owner"] == value], pid_col
        label = sub[by].map(_names(runs, pid_col, pname_col))
    out = sub.assign(_label=label.fillna("(不明)")).groupby("_label").size().sort_values(ascending=False)
    return out.head(limit).rename_axis("name").reset_index(name="n")


# ---------------------------------------------------------------- 図(SVG)
R_INNER, R_OUTER = 185, 300
LABEL_LEN = 8


def _short(name: str, n: int = LABEL_LEN) -> str:
    name = str(name)
    return name if len(name) <= n else name[:n - 1] + "…"


def _angles(k: int, offset: float = 0.0) -> list[float]:
    return [-math.pi / 2 + offset + 2 * math.pi * i / k for i in range(k)] if k else []


def _owner_order(rel: dict, inner_angle: dict) -> list:
    """馬主を、つながる相手の向きの平均で並べる(線の交差を減らす)。"""
    links = rel["links"]
    keys = []
    for owner in rel["outer"].index:
        l = links[links["owner"] == owner]
        if l.empty:
            keys.append((owner, 99.0))
            continue
        x = sum(n * math.cos(inner_angle[p]) for p, n in zip(l["partner"], l["n"]))
        y = sum(n * math.sin(inner_angle[p]) for p, n in zip(l["partner"], l["n"]))
        keys.append((owner, (math.atan2(y, x) + math.pi / 2) % (2 * math.pi)))
    return [o for o, _ in sorted(keys, key=lambda t: t[1])]


def _label(x: float, y: float, a: float, r_node: float, lines: list[str], cls: str) -> str:
    """ノードの外側に、輪の外向きに名前を置く。"""
    gap = r_node + 6
    lx, ly = x + gap * math.cos(a), y + gap * math.sin(a)
    c = math.cos(a)
    anchor = "middle" if abs(c) < 0.3 else ("start" if c > 0 else "end")
    dy0 = 4 if abs(c) >= 0.3 else (13 if math.sin(a) > 0 else -4 - 15 * (len(lines) - 1))
    tspans = "".join(f'<tspan x="{lx:.1f}" dy="{dy0 if i == 0 else 15}">{html.escape(t)}</tspan>'
                     for i, t in enumerate(lines))
    return f'<text class="{cls}" x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}">{tspans}</text>'


# 図の色。画面の st.html は SVG を通さないので画像として出す。画像の中には画面の CSS が届かないため、テーマごとに色を持つ
THEMES = {
    "light": {"accent": "#1f7a4d", "on_accent": "#ffffff", "text": "#1f2328", "bg": "#ffffff", "inner": "#cfe6d9"},
    "dark": {"accent": "#5fcf8f", "on_accent": "#10241a", "text": "#e6e8eb", "bg": "#0e1117", "inner": "#24493a"},
}
GROUP_COLORS = ["#2f6fd6", "#d0453a", "#d89a1e", "#9aa1ab"]
GOLD = "#d4a72c"


def _svg_style(theme: str) -> str:
    c = THEMES.get(theme, THEMES["light"])
    groups = "".join(f".kb-kk-g{i}{{fill:{col}}}" for i, col in enumerate(GROUP_COLORS))
    return (f"<style>text{{font-family:'Hiragino Sans','Yu Gothic','Meiryo','Noto Sans CJK JP',sans-serif}}"
            f".kb-kk-main{{stroke:{c['accent']};stroke-linecap:round;opacity:.75}}"
            f".kb-kk-sub{{stroke:{c['text']};opacity:.22}}"
            f".kb-kk-center{{fill:{c['accent']}}}"
            f".kb-kk-cname{{fill:{c['on_accent']};font-size:15px;font-weight:700}}"
            f".kb-kk-cn{{fill:{c['on_accent']};font-size:12px}}"
            f".kb-kk-inner{{fill:{c['inner']};stroke:{c['accent']};stroke-width:1.5}}"
            f".kb-kk-owner{{stroke:{c['bg']};stroke-width:1.5}}"
            f".kb-kk-gold{{fill:none;stroke:{GOLD};stroke-width:3}}"
            f".kb-kk-label{{fill:{c['text']};font-size:14px}}"
            f".kb-kk-label tspan+tspan{{opacity:.65;font-size:12px}}"
            f".kb-kk-dim{{opacity:.12}}line.kb-kk-dim{{opacity:.05}}{groups}</style>")


def diagram_img(rel: dict, highlight: tuple[str, str] | None = None, theme: str = "light") -> str:
    """相関図を、画面にそのまま置ける img 要素にする。"""
    svg = diagram_svg(rel, highlight, theme)
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (f'<img class="kb-kk" src="data:image/svg+xml;base64,{data}" '
            f'alt="{html.escape(rel["name"])}の相関図">')


def diagram_svg(rel: dict, highlight: tuple[str, str] | None = None, theme: str = "light") -> str:
    """相関図の SVG。highlight は ("inner", 相手の番号) か ("outer", 馬主名)。そのつながりだけを濃く出す。"""
    inner, outer, links = rel["inner"], rel["outer"], rel["links"]
    ia = dict(zip(inner.index, _angles(len(inner))))
    order = _owner_order(rel, ia)
    oa = dict(zip(order, _angles(len(order), offset=math.pi / max(len(order), 1))))
    pos_i = {p: (R_INNER * math.cos(a), R_INNER * math.sin(a)) for p, a in ia.items()}
    pos_o = {o: (R_OUTER * math.cos(a), R_OUTER * math.sin(a)) for o, a in oa.items()}
    nmax = max([1, *inner["n"].tolist(), *outer["n"].tolist()])

    def on(ring, key, partner=None, owner=None) -> bool:
        if highlight is None:
            return True
        h_ring, h_key = highlight
        if ring == "center":
            return True
        if ring == "link":
            return (h_ring == "inner" and partner == h_key) or (h_ring == "outer" and owner == h_key)
        if ring == h_ring and key == h_key:
            return True
        # 選んだ相手と線でつながる相手も残す
        if h_ring == "inner" and ring == "outer":
            return bool(((links["partner"] == h_key) & (links["owner"] == key)).any())
        if h_ring == "outer" and ring == "inner":
            return bool(((links["owner"] == h_key) & (links["partner"] == key)).any())
        return False

    def dim(flag: bool) -> str:
        return "" if flag else " kb-kk-dim"

    parts = []
    lmax = max([1, *links["n"].tolist()])
    for p, o, n in zip(links["partner"], links["owner"], links["n"]):
        (x1, y1), (x2, y2) = pos_i[p], pos_o[o]
        w = 0.6 + 2.4 * n / lmax
        parts.append(f'<line class="kb-kk-sub{dim(on("link", None, p, o))}" x1="{x1:.1f}" y1="{y1:.1f}" '
                     f'x2="{x2:.1f}" y2="{y2:.1f}" stroke-width="{w:.2f}"/>')
    for p, r in inner.iterrows():
        x, y = pos_i[p]
        w = 1 + 9 * r["share_center"] / max(inner["share_center"].max(), 1e-9)
        parts.append(f'<line class="kb-kk-main{dim(on("inner", p))}" x1="0" y1="0" x2="{x:.1f}" y2="{y:.1f}" '
                     f'stroke-width="{w:.2f}"/>')
    parts.append('<circle class="kb-kk-center" r="46"/>')
    parts.append(f'<text class="kb-kk-cname" y="-2" text-anchor="middle">{html.escape(_short(rel["name"], 7))}</text>'
                 f'<text class="kb-kk-cn" y="17" text-anchor="middle">{rel["n"]:,}回</text>')
    for p, r in inner.iterrows():
        x, y = pos_i[p]
        rad = 6 + 12 * math.sqrt(r["n"] / nmax)
        title = (f'{r["name"]} {int(r["n"])}回 / {rel["name"]}から{r["share_center"]:.1%}({int(r["rank_center"])}位)'
                 f' / 相手から{r["share_partner"]:.1%}({int(r["rank_partner"])}位)')
        gold = f'<circle class="kb-kk-gold" cx="{x:.1f}" cy="{y:.1f}" r="{rad + 4:.1f}"/>' if r["top"] else ""
        parts.append(f'<g class="{dim(on("inner", p)).strip()}"><title>{html.escape(title)}</title>{gold}'
                     f'<circle class="kb-kk-inner" cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}"/>'
                     + _label(x, y, ia[p], rad, [_short(r["name"]), f'{int(r["n"])}'], "kb-kk-label") + "</g>")
    for o in order:
        r = outer.loc[o]
        x, y = pos_o[o]
        rad = 5 + 10 * math.sqrt(r["n"] / nmax)
        group = GROUP_ORDER.index(r["group"])
        title = (f'{o}({r["group"]}) {int(r["n"])}回 / {rel["name"]}から{r["share_center"]:.1%}'
                 f'({int(r["rank_center"])}位) / 馬主から{r["share_partner"]:.1%}({int(r["rank_partner"])}位)')
        gold = f'<circle class="kb-kk-gold" cx="{x:.1f}" cy="{y:.1f}" r="{rad + 4:.1f}"/>' if r["top"] else ""
        parts.append(f'<g class="{dim(on("outer", o)).strip()}"><title>{html.escape(title)}</title>{gold}'
                     f'<circle class="kb-kk-owner kb-kk-g{group}" cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}"/>'
                     + _label(x, y, oa[o], rad, [_short(o), f'{int(r["n"])}'], "kb-kk-label kb-kk-olabel") + "</g>")
    view = R_OUTER + 130
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-view} {-view + 20} {2 * view} {2 * view - 40}" '
            f'role="img" aria-label="{html.escape(rel["name"])}の相関図">' + _svg_style(theme) + "".join(parts) + "</svg>")


def legend_html() -> str:
    items = [('<span class="kb-kk-sw kb-kk-sw-main"></span>', "太い線: 中心とのつながり(太いほど中心から見たシェアが大きい)"),
             ('<span class="kb-kk-sw kb-kk-sw-sub"></span>', "細い線: その馬主の馬に、その相手と組んで出た"),
             ('<span class="kb-kk-sw kb-kk-sw-gold"></span>', "金の輪: 相手から見て、中心が1位")]
    items += [(f'<span class="kb-kk-dot kb-kk-g{i}"></span>', g) for i, g in enumerate(GROUP_ORDER)]
    return '<div class="kb-kk-legend">' + "".join(f"<span>{sw}{html.escape(t)}</span>" for sw, t in items) + "</div>"
