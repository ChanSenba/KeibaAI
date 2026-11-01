"""回顧の「レースの見解」。レース全体の流れ、人気馬の結果と条件の変化、予想の振り返りを、データから文章にする。

1. 流れ: 前半・後半3F とペース、勝ち馬と上位3頭の序盤の位置・上がり、前に付けた馬の結果から「どんな決着だったか」。
2. 人気馬: 1〜3番人気の着順と、前走からの条件の変化(海外・地方帰り、乗り替わりと騎手の成績の差、間隔、距離、馬体重)。
   4着以下に負けた人気馬は、その変化を「敗因の候補」として挙げ、過去の同じ条件の人気馬が、オッズの見込みと比べて
   どれだけ勝ったか(実際の勝ち数 ÷ オッズから出した見込みの勝ち数)を添える。1.0 なら市場の見込みどおりで、
   その条件だけで割り引く根拠は薄い。1レースの結果から原因は断定できないので、候補と過去の傾向を並べるところまでにする。
3. 予想: ◎の着順、勝ち馬を何番手に見ていたか、人気と比べて高く・低く見ていたか。

海外・地方の成績は other_runs(jvother.py)から読む。JRA-VAN の海外の成績は、レースから数週間たってから出るので、
直近の海外遠征がまだ入っていないことがある。
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .flow_review import parse_early_position, position_group

# JRA-VAN の海外の競馬場コード(英字)の国。分からないコードは「海外」とだけ書く
OVERSEAS = {"A4": "アメリカ", "A6": "イギリス", "A8": "フランス", "B2": "アイルランド", "B6": "オーストラリア",
            "C7": "UAE", "G0": "香港", "K6": "サウジアラビア", "N2": "バーレーン", "F0": "韓国"}
JOCKEY_DAYS = 365  # 騎手の成績を見る期間(日)
JOCKEY_GAP = 0.05  # 乗り替わりで、騎手の3着内率がこれ以上違えば「格上げ・格下げ」とする
LAYOFF_DAYS = 90  # 地方・海外も含めた前走からこれ以上空いたら休み明け
DIST_CHANGE = 400  # 前走からこれ以上距離が変われば書く(m)
BODY_CHANGE = 10  # 馬体重がこれ以上増減したら書く(kg)
FAV_POP = 3  # 「人気馬」= 確定の単勝人気がこれ以内
RATE_FROM = "2019-01-01"  # 過去の傾向を数える期間の始まり


def has_other_runs(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='other_runs'").fetchone() is not None


def place_name(region: str | None, code: str | None) -> str:
    if region == "海外":
        return OVERSEAS.get(code or "", "海外")
    return "地方"


# ---------------------------------------------------------------- 過去の傾向(人気馬の、条件ごとの実績 ÷ 市場の見込み)
_RATES: dict = {}


def _history(conn: sqlite3.Connection, until: str) -> pd.DataFrame:
    """RATE_FROM〜until(含まない)の中央の出走に、条件の変化の印を付けた表。"""
    df = pd.read_sql_query(
        "SELECT r.race_id, r.horse_id, r.jockey_id, r.finish, r.win_odds, r.popularity, c.date "
        "FROM results r JOIN races c USING (race_id) WHERE c.date < ? AND r.finish IS NOT NULL", conn, params=(until,))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["horse_id", "date"]).reset_index(drop=True)
    g = df.groupby("horse_id", sort=False)
    df["prev_date"], df["prev_jockey"] = g["date"].shift(1), g["jockey_id"].shift(1)
    # 騎手の前の年の3着内率(その年の成績は使わない)
    df["year"] = df["date"].dt.year
    yr = df.assign(top3=df["finish"] <= 3).groupby(["jockey_id", "year"])["top3"].agg(["mean", "size"]).reset_index()
    yr = yr[yr["size"] >= 30]
    rate = {(j, y + 1): m for j, y, m in zip(yr["jockey_id"], yr["year"], yr["mean"])}
    df["j_now"] = [rate.get((j, y)) for j, y in zip(df["jockey_id"], df["year"])]
    df["j_prev"] = [rate.get((j, y)) for j, y in zip(df["prev_jockey"], df["year"])]
    df["abroad"] = False
    df["nar"] = False
    last_any = df["prev_date"]
    if has_other_runs(conn):
        o = pd.read_sql_query("SELECT horse_id, date AS o_date, region FROM other_runs WHERE date < ?", conn, params=(until,))
        if not o.empty:
            o["o_date"] = pd.to_datetime(o["o_date"])
            q = df[["horse_id", "date"]].reset_index().sort_values("date")
            m = pd.merge_asof(q, o.sort_values("o_date"), left_on="date", right_on="o_date", by="horse_id",
                              allow_exact_matches=False).set_index("index").reindex(df.index)
            after = m["o_date"].notna() & (df["prev_date"].isna() | (m["o_date"] > df["prev_date"]))
            df["abroad"] = after & (m["region"] == "海外")
            df["nar"] = after & (m["region"] == "地方")
            last_any = m["o_date"].where(after, df["prev_date"])
    df["days_any"] = (df["date"] - last_any).dt.days
    df = df[df["date"] >= pd.Timestamp(RATE_FROM)]
    odds = pd.to_numeric(df["win_odds"], errors="coerce")
    inv = (1.0 / odds).where(odds > 0)
    df["p_market"] = inv / inv.groupby(df["race_id"]).transform("sum")
    return df[(df["popularity"] <= FAV_POP) & df["p_market"].notna()]


CONDITIONS = {
    "海外帰り": lambda d: d["abroad"],
    "地方帰り": lambda d: d["nar"],
    "乗り替わり": lambda d: d["prev_jockey"].notna() & (d["prev_jockey"] != d["jockey_id"]),
    "乗り替わり(騎手の格下げ)": lambda d: (d["prev_jockey"] != d["jockey_id"]) & (d["j_now"] - d["j_prev"] <= -JOCKEY_GAP),
    "休み明け": lambda d: d["days_any"] >= LAYOFF_DAYS,
}


def base_rates(conn: sqlite3.Connection, until: str) -> dict:
    """人気馬(1〜FAV_POP番人気)の、条件ごとの {頭数, 勝ち数, 市場の見込みの勝ち数}。until より前の結果だけを使う。
    数えるのに10秒ほどかかるので、同じ DB・同じ until の結果は使い回す。"""
    key = (_db_key(conn), until)
    if key not in _RATES:
        h = _history(conn, until)
        out = {"人気馬全体": (len(h), int((h["finish"] == 1).sum()), float(h["p_market"].sum()))}
        for name, cond in CONDITIONS.items():
            g = h[cond(h).fillna(False).astype(bool)]
            out[name] = (len(g), int((g["finish"] == 1).sum()), float(g["p_market"].sum()))
        _RATES.clear()
        _RATES[key] = out
    return _RATES[key]


def rate_text(name: str, rate: tuple, since: str = RATE_FROM) -> str:
    """「海外帰りの1〜3番人気(2019年〜): 114頭・26勝(勝率23%)。オッズの見込み25.2勝の1.03倍で、市場の見込みどおり」"""
    n, wins, exp = rate
    if n == 0 or exp <= 0:
        return f"{name}の1〜{FAV_POP}番人気は、過去の例がありません。"
    ratio = wins / exp
    half = 1.96 * np.sqrt(exp) / exp  # 勝ち数をポアソンとみた95%の幅
    if ratio - half > 1:
        verdict = "見込みより勝っていて、割り引く根拠にはならない"
    elif ratio + half < 1:
        verdict = "見込みより勝てておらず、割り引く根拠になりうる"
    else:
        verdict = "市場の見込みどおりで、この条件だけで負けたとは言いにくい"
        if n < 100:
            verdict = "例が少なく、良し悪しは言えない"
    return (f"{name}の1〜{FAV_POP}番人気({since[:4]}年〜): {n:,}頭・{wins:,}勝(勝率{wins / n:.0%})。"
            f"オッズからの見込み{exp:,.1f}勝の{ratio:.2f}倍(95%の幅 ±{half:.2f})で、{verdict}。")


# ---------------------------------------------------------------- 馬ごとの条件の変化
_JOCKEYS: dict = {}


def _db_key(conn: sqlite3.Connection) -> tuple:
    """保存した計算を使い回すための、DB の指紋(ファイルと結果の行数)。"""
    file = next((f for _, name, f in conn.execute("PRAGMA database_list") if name == "main"), "") or str(id(conn))
    return file, conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]


def _jockey_rate(conn: sqlite3.Connection, jockey_id, date: str) -> tuple[float, int] | None:
    """騎手の、date より前 JOCKEY_DAYS 日の3着内率と騎乗数。同じ日の騎手はまとめて1回で数える。"""
    if not jockey_id:
        return None
    key = (_db_key(conn), date)
    if key not in _JOCKEYS:
        start = (pd.Timestamp(date) - pd.Timedelta(days=JOCKEY_DAYS)).date().isoformat()
        rows = conn.execute(
            "SELECT r.jockey_id, COUNT(*), SUM(r.finish <= 3) FROM results r JOIN races c USING (race_id) "
            "WHERE c.date >= ? AND c.date < ? AND r.finish IS NOT NULL GROUP BY r.jockey_id", (start, date)).fetchall()
        _JOCKEYS.clear()
        _JOCKEYS[key] = {j: (top3 / n, n) for j, n, top3 in rows if n}
    return _JOCKEYS[key].get(jockey_id)


def _prev_run(conn: sqlite3.Connection, horse_id, date: str) -> dict | None:
    row = conn.execute(
        "SELECT c.date, c.name, c.venue, c.course_type, c.distance, r.finish, r.jockey, r.jockey_id, r.popularity, r.body_weight "
        "FROM results r JOIN races c USING (race_id) WHERE r.horse_id = ? AND +c.date < ? ORDER BY c.date DESC LIMIT 1",
        (horse_id, date)).fetchone()
    keys = ["date", "name", "venue", "course_type", "distance", "finish", "jockey", "jockey_id", "popularity", "body_weight"]
    return dict(zip(keys, row)) if row else None


def _prev_other(conn: sqlite3.Connection, horse_id, date: str) -> dict | None:
    if not has_other_runs(conn):
        return None
    cols = [r[1] for r in conn.execute("PRAGMA table_info(other_runs)")]
    name = "name" if "name" in cols else "NULL"
    row = conn.execute(
        f"SELECT date, region, venue_code, course_type, distance, finish, {name} FROM other_runs "
        "WHERE horse_id = ? AND date < ? ORDER BY date DESC LIMIT 1", (horse_id, date)).fetchone()
    keys = ["date", "region", "venue_code", "course_type", "distance", "finish", "name"]
    return dict(zip(keys, row)) if row else None


def _weeks(a: str, b: str) -> int:
    return int((pd.Timestamp(b) - pd.Timestamp(a)).days // 7)


def horse_changes(conn: sqlite3.Connection, h: dict, race: dict) -> list[tuple[tuple, str, bool]]:
    """1頭の前走からの条件の変化。[(種類の組, 説明, 負けの理由になりうるか)]。種類は CONDITIONS の名前(と「距離」「馬体重」)。
    海外帰りで間隔も空いていれば ("海外帰り", "休み明け") のように2つ持つ。騎手の格上げの乗り替わりは、負けの理由に数えない。"""
    date = race["date"]
    out = []
    jra = _prev_run(conn, h.get("horse_id"), date)
    other = _prev_other(conn, h.get("horse_id"), date)
    if other and (jra is None or other["date"] > jra["date"]):
        where = place_name(other["region"], other["venue_code"])
        what = f"{other['name']}" if isinstance(other.get("name"), str) and other["name"].strip() else \
            f"{other['course_type'] or ''}{other['distance'] or ''}m".strip()
        fin = f"{int(other['finish'])}着" if other["finish"] is not None else "着順不明"
        kind = "海外帰り" if other["region"] == "海外" else "地方帰り"
        rest = (pd.Timestamp(date) - pd.Timestamp(other["date"])).days >= LAYOFF_DAYS
        out.append(((kind, "休み明け") if rest else (kind,),
                    f"前走は{where}({other['date']} {what} {fin})から中{_weeks(other['date'], date)}週"
                    + ("の休み明け" if rest else ""), True))
    elif jra and (pd.Timestamp(date) - pd.Timestamp(jra["date"])).days >= LAYOFF_DAYS:
        prev = f"(前走 {jra['name']} {int(jra['finish'])}着)" if jra["finish"] else ""
        out.append((("休み明け",), f"{_weeks(jra['date'], date)}週ぶり{prev}", True))
    if jra and jra.get("jockey_id") and h.get("jockey_id") and jra["jockey_id"] != h["jockey_id"]:
        now, prev = _jockey_rate(conn, h["jockey_id"], date), _jockey_rate(conn, jra["jockey_id"], date)
        text = f"乗り替わり({jra['jockey']} → {h.get('jockey') or '不明'}"
        kind, bad = "乗り替わり", True
        if now and prev and min(now[1], prev[1]) >= 30:
            text += f"。直近1年の3着内率 {prev[0]:.0%} → {now[0]:.0%}"
            if now[0] - prev[0] <= -JOCKEY_GAP:
                kind = "乗り替わり(騎手の格下げ)"
                text += "、格下げ"
            elif now[0] - prev[0] >= JOCKEY_GAP:
                text += "、格上げ"
                bad = False
        if other and other["date"] > jra["date"]:
            text += "。海外・地方での騎手は不明なので、前の中央のレースと比べた"
        out.append(((kind,), text + ")", bad))
    if jra and jra.get("distance") and race.get("distance") and abs(race["distance"] - jra["distance"]) >= DIST_CHANGE \
            and not (other and other["date"] > jra["date"]):
        d = race["distance"] - jra["distance"]
        out.append((("距離",), f"前走から{abs(d)}m{'延長' if d > 0 else '短縮'}({jra['distance']}m → {race['distance']}m)", True))
    bwd = h.get("body_weight_diff")
    if bwd is not None and bwd == bwd and abs(bwd) >= BODY_CHANGE:
        out.append((("馬体重",), f"馬体重 {int(bwd):+d}kg", True))
    return out


# ---------------------------------------------------------------- レースの見解
def _early(passing) -> float:
    return parse_early_position(passing)


def flow_lines(race: dict, runners: pd.DataFrame) -> tuple[list[str], str | None]:
    """流れの文。(文の並び, 決着の形 '前' '後' '混' か None)。runners は全出走馬(finish・passing・last3f・horse_name・popularity)。"""
    lines = []
    a, b = race.get("pace_first3f"), race.get("pace_last3f")
    if a and b and a == a and b == b:
        d = a - b
        pace = "スロー" if d >= 1.0 else "ハイ" if d <= -1.0 else "平均"
        lines.append(f"前半3F {a:.1f}秒・後半3F {b:.1f}秒の{pace}ペース(差 {d:+.1f}秒)。")
    r = runners[runners["finish"].notna()].copy()
    n = len(runners)
    if r.empty:
        return lines, None
    r["early"] = [_early(p) for p in r["passing"]]
    r["grp"] = [position_group(p, n) for p in r["passing"]]
    r["l3_rank"] = r["last3f"].rank(method="min")
    r = r.sort_values("finish")
    w = r.iloc[0]
    pop = f"{int(w['popularity'])}番人気" if w["popularity"] == w["popularity"] else "人気不明"
    text = f"勝った{w['horse_name']}({pop})は"
    if w["early"] == w["early"]:
        text += f"序盤{int(w['early'])}番手から、"
    if w["last3f"] == w["last3f"]:
        text += f"上がり{w['last3f']:.1f}秒({int(w['l3_rank'])}位)。"
    lines.append(text.rstrip("、") + ("" if text.endswith("。") else "。"))
    top3 = r[r["finish"] <= 3]
    known = top3["grp"].dropna()
    shape = None
    if len(known) == len(top3) and len(top3) >= 3:
        pos = "・".join(f"{int(e)}" for e in top3["early"])
        if (known == "後").sum() + (known == "中").sum() == len(known) and (known == "後").any():
            shape = "後"
            lines.append(f"1〜3着の序盤の位置は{pos}番手で、差し・追い込みが決まった。")
        elif (known == "前").sum() >= 2:
            shape = "前"
            lines.append(f"1〜3着の序盤の位置は{pos}番手で、前に付けた馬が残った。")
        else:
            shape = "混"
            lines.append(f"1〜3着の序盤の位置は{pos}番手で、前と後ろが入り混じった決着。")
    front = r[r["early"] <= 3]
    if len(front) and shape == "後":
        lines.append(f"序盤3番手以内の{len(front)}頭の最高は{int(front['finish'].min())}着。")
    fast = r[r["l3_rank"] <= 3]
    if len(fast):
        k = int((fast["finish"] <= 3).sum())
        lines.append(f"上がり3F上位{len(fast)}頭のうち{k}頭が3着以内。")
    return lines, shape


def _place(fin, status=None) -> str:
    return f"{int(fin)}着" if fin == fin and fin is not None else (status or "着順なし")


def favorite_lines(conn: sqlite3.Connection, race: dict, runners: pd.DataFrame, shape: str | None) -> tuple[list[str], set]:
    """1〜3番人気の結果と条件の変化。(文の並び, 敗因の候補に挙がった条件の名前)。"""
    lines, kinds = [], set()
    n = len(runners)
    favs = runners[runners["popularity"] <= FAV_POP].sort_values("popularity")
    held = favs[favs["finish"] <= 3]
    if len(favs):
        lines.append(f"1〜{FAV_POP}番人気のうち3着以内は{len(held)}頭"
                     + (f"({'・'.join(f'{int(p)}番人気' for p in held['popularity'])})" if len(held) else "") + "。")
    l3 = runners["last3f"].rank(method="min")
    for i, x in favs.iterrows():
        h = x.to_dict()
        head = f"{int(x['popularity'])}番人気 {x['horse_name']}: {_place(x['finish'], x.get('status'))}"
        bits = []
        e = _early(x["passing"])
        if e == e:
            bits.append(f"序盤{int(e)}番手")
        if l3[i] == l3[i]:
            bits.append(f"上がり{int(l3[i])}位")
        head += f"({'・'.join(bits)})" if bits else ""
        changes = horse_changes(conn, h, race)
        lost = x["finish"] != x["finish"] or x["finish"] > 3
        grp = position_group(x["passing"], n)
        rank = f"上がり{int(l3[i])}位" if l3[i] == l3[i] else "上がり不明"
        if lost and shape == "後" and grp == "前":
            changes.append((("展開",), "前に付けて、差し・追い込みの決着に巻き込まれた", True))
        elif lost and shape == "前" and grp == "後":
            changes.append((("展開",), "後ろからで、前残りの決着に届かなかった", True))
        elif lost and shape == "後" and grp == "後":
            changes.append((("伸び",), f"差し・追い込みが決まる流れで後ろにいながら{rank}で、展開のせいとは言いにくい", True))
        elif lost and shape == "前" and grp == "前":
            changes.append((("伸び",), "前が残る流れで前にいながら粘れず、展開のせいとは言いにくい", True))
        bad = [c for c in changes if c[2]] if lost else []
        rest = [c for c in changes if c not in bad]
        if bad:
            head += "。敗因の候補: " + "、".join(t for _, t, _ in bad)
            kinds |= {k for ks, _, _ in bad for k in ks if k in CONDITIONS}
        elif lost:
            head += "。敗因の候補になる条件の変化は、データからは見えない"
        if rest:
            head += f"。{'ほかの変化' if bad else '条件の変化'}: " + "、".join(t for _, t, _ in rest)
        if lost and any("休み明け" in ks and not {"海外帰り", "地方帰り"} & set(ks) for ks, _, _ in changes):
            kinds.add("_遅れ")
        lines.append(head + "。")
    return lines, kinds


def prediction_lines(g: pd.DataFrame) -> list[str]:
    """予想の振り返り。g は1レースの予想と結果(mark・p_win_ability・final_pop・finish・horse_name)。"""
    if g.empty or g["p_win_ability"].isna().all():
        return []
    g = g.assign(rank=g["p_win_ability"].rank(ascending=False, method="first"))
    out = []
    top = g[g["mark"] == "◎"]
    if len(top):
        t = top.iloc[0]
        out.append(f"◎{t['horse_name']}は{_place(t['finish'], t.get('status'))}"
                   + (f"({int(t['final_pop'])}番人気)" if t["final_pop"] == t["final_pop"] else "") + "。")
    win = g[g["finish"] == 1]
    if len(win):
        w = win.iloc[0]
        text = f"勝った{w['horse_name']}は予想{int(w['rank'])}位"
        if isinstance(w.get("mark"), str) and w["mark"]:
            text += f"({w['mark']})"
        pop = w["final_pop"]
        if pop == pop and pop is not None:
            diff = int(pop) - int(w["rank"])
            text += (f"で、人気({int(pop)}番人気)より{diff}つ上に見ていた" if diff > 0
                     else f"で、人気({int(pop)}番人気)より{-diff}つ下に見ていた" if diff < 0 else "で、人気と同じ順")
        out.append(text + "。")
    return out


# ---------------------------------------------------------------- 各馬の勝因・敗因の印(features.add_result_tags と同じ計算)
TAG_EDGE = 0.2  # 展開・枠の向き(3着内率の差)がこれ以上なら「恵まれた」、−これ以下なら「不利」
TAGS = {"展開に恵まれた": lambda d: d["flow_fit"] >= TAG_EDGE, "展開が不利": lambda d: d["flow_fit"] <= -TAG_EDGE,
        "枠に恵まれた": lambda d: d["gate_fit"] >= TAG_EDGE, "枠が不利": lambda d: d["gate_fit"] <= -TAG_EDGE,
        "上がり上位で届かず": lambda d: d["closer_unlucky"] == 1}
GOOD_TAGS = {"展開に恵まれた", "枠に恵まれた"}


def _tag_frame(runners: pd.DataFrame) -> pd.DataFrame:
    """出走馬の表(race_id・horse_id・number・finish・passing・last3f)に flow_fit・gate_fit・closer_unlucky を付ける。"""
    from .features import add_result_tags

    d = runners.copy()
    d["finish"] = pd.to_numeric(d["finish"], errors="coerce")
    d["field_size"] = d.groupby("race_id")["number"].transform("count")
    d["early_pos"] = d["passing"].map(parse_early_position) / d["field_size"]
    d["top3"] = (d["finish"] <= 3).astype(int)
    d["l3f_rank_rel"] = pd.to_numeric(d["last3f"], errors="coerce").groupby(d["race_id"]).rank(method="min") / d["field_size"]
    return add_result_tags(d)


def tags_of(row) -> list[str]:
    return [name for name, cond in TAGS.items() if bool(cond(row))]


def horse_lines(runners: pd.DataFrame) -> list[str]:
    """各馬の勝因・敗因。1〜3着は全頭、4着以下は印の付いた馬だけ(着順の順)。"""
    d = _tag_frame(runners.assign(race_id=runners.get("race_id", "r")))
    out = []
    for _, x in d[d["finish"].notna()].sort_values("finish").iterrows():
        tags = tags_of(x)
        won = x["finish"] <= 3
        if not tags and not won:
            continue
        good, bad = [t for t in tags if t in GOOD_TAGS], [t for t in tags if t not in GOOD_TAGS]
        pop = f"・{int(x['popularity'])}番人気" if x.get("popularity") == x.get("popularity") and x.get("popularity") else ""
        text = f"{int(x['finish'])}着 {x['horse_name']}({pop.lstrip('・')}): " if pop else f"{int(x['finish'])}着 {x['horse_name']}: "
        if won:
            parts = [f"勝因の候補 {'・'.join(good)}" if good else "展開・枠の恵みはなく、力で来た"]
            if bad:
                parts.append(f"{'・'.join(bad)}をはね返した")
        else:
            parts = [f"敗因の候補 {'・'.join(bad)}"] if bad else []
            if good:
                parts.append(f"{'・'.join(good)}のに負けた")
        out.append(text + "、".join(parts) + "。")
    return out


_NEXT: dict = {}


def next_run_rates(conn: sqlite3.Connection, until: str) -> dict:
    """前走で各印が付いた馬が、次走でオッズの見込みと比べてどれだけ勝ったか。{印: (頭数, 勝ち数, 見込みの勝ち数)}。
    「負け方を学習に生かせるか」の確かめ。1.0 から離れていれば、市場が見落としている(モデルに足す価値がある)。"""
    key = (_db_key(conn), until)
    if key not in _NEXT:
        df = pd.read_sql_query(
            "SELECT r.race_id, r.number, r.horse_id, r.finish, r.passing, r.last3f, r.win_odds, c.date "
            "FROM results r JOIN races c USING (race_id) WHERE c.date < ? AND r.finish IS NOT NULL", conn, params=(until,))
        df = _tag_frame(df.sort_values(["date", "race_id", "number"]).reset_index(drop=True))
        g = df.groupby("horse_id", sort=False)
        prev = pd.DataFrame({c: g[c].shift(1) for c in ("flow_fit", "gate_fit", "closer_unlucky")})
        odds = pd.to_numeric(df["win_odds"], errors="coerce")
        inv = (1.0 / odds).where(odds > 0)
        df["p_market"] = inv / inv.groupby(df["race_id"]).transform("sum")
        ok = (df["date"] >= RATE_FROM) & df["p_market"].notna()
        out = {"全体": (int(ok.sum()), int(((df["finish"] == 1) & ok).sum()), float(df.loc[ok, "p_market"].sum()))}
        for name, cond in TAGS.items():
            m = ok & cond(prev).fillna(False).astype(bool)
            out[name] = (int(m.sum()), int(((df["finish"] == 1) & m).sum()), float(df.loc[m, "p_market"].sum()))
        _NEXT.clear()
        _NEXT[key] = out
    return _NEXT[key]


def next_run_table(rates: dict) -> pd.DataFrame:
    rows = []
    for name, (n, wins, exp) in rates.items():
        if name == "全体" or not n:
            continue
        half = 1.96 * np.sqrt(exp) / exp if exp > 0 else float("nan")
        rows.append({"前走の印": name, "次走の頭数": n, "勝ち": wins, "見込み": round(exp, 1),
                     "実績÷見込み": round(wins / exp, 3) if exp else float("nan"), "95%の幅": round(half, 3)})
    return pd.DataFrame(rows)


def race_view(conn: sqlite3.Connection, race_id: str, g: pd.DataFrame | None = None) -> dict | None:
    """1レースの見解。{"flow": [...], "favorites": [...], "trends": [...], "prediction": [...]}。結果がなければ None。

    g は回顧の表(weekly.review_day の horses)のそのレースの行。なければ予想の振り返りは空。
    """
    row = conn.execute("SELECT date, distance, course_type, venue, pace_first3f, pace_last3f FROM races WHERE race_id = ?",
                       (race_id,)).fetchone()
    if row is None:
        return None
    race = dict(zip(["date", "distance", "course_type", "venue", "pace_first3f", "pace_last3f"], row))
    runners = pd.read_sql_query(
        "SELECT number, horse_id, horse_name, jockey, jockey_id, finish, status, passing, last3f, popularity, "
        "body_weight_diff FROM results WHERE race_id = ? ORDER BY number", conn, params=(race_id,))
    if runners["finish"].isna().all():
        return None
    for c in ("finish", "last3f", "popularity", "body_weight_diff"):
        runners[c] = pd.to_numeric(runners[c], errors="coerce")
    flow, shape = flow_lines(race, runners)
    favs, kinds = favorite_lines(conn, race, runners, shape)
    trends = []
    if kinds - {"_遅れ"}:
        rates = base_rates(conn, race["date"][:8] + "01")  # その月の前月までの結果(月ごとに1回だけ数える)
        trends = [rate_text(k, rates[k]) for k in CONDITIONS if k in kinds]
    if "_遅れ" in kinds:
        trends.append("JRA-VAN の海外の成績はレースから数週間たって出るので、休み明けに見える馬が、直近に海外を使っている"
                      "ことがある(出たあとで回顧を開き直すと反映される)。")
    return {"flow": flow, "favorites": favs, "horses": horse_lines(runners.assign(race_id=race_id)), "trends": trends,
            "prediction": prediction_lines(g) if g is not None else []}


SECTIONS = {"flow": "流れ", "favorites": "人気馬", "horses": "各馬の勝因・敗因", "trends": "過去の傾向", "prediction": "予想"}


def view_markdown(view: dict, indent: str = "  ") -> list[str]:
    """見解を Markdown の行にする(weekly の回顧ファイル用)。"""
    out = []
    for key, title in SECTIONS.items():
        lines = view.get(key) or []
        if not lines:
            continue
        if key in ("favorites", "horses", "trends"):
            out.append(f"{indent}- {title}:")
            out += [f"{indent}  - {t}" for t in lines]
        else:
            out.append(f"{indent}- {title}: " + "".join(lines))
    return out
