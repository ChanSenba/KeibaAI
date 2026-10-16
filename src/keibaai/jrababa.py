"""JRA の馬場情報アーカイブ(含水率・クッション値の PDF)を読んで、data/jra_baba.db に入れる。

PDF は https://www.jra.go.jp/keiba/baba/archive/ の年ごとのページから、開催(例: 2026pdf/tokyo03.pdf)ごとに1つ。
含水率は 2018-07 から、クッション値は 2020-09 から。PDF の文字を読むのに pdfplumber を使う(読むときだけ)。

PDF の形は2通り(どちらも文字として読める):
  週ごとの表(含水率だけの年): 「2018年11月30日から12月2日の含水率」「場所 金曜日 土曜日 日曜日」のあとに
      「芝コース含水率 ゴール前 8.9 10.3 9.6」「（パーセント） ４コーナー …」「ダートコース含水率 ゴール前 …」「… ４コーナー …」
  1日1行の表: 「第 1日 6月 6日 土曜日 C 07:00 9.9 05:00 16.2 15.7 7.7 8.9」
      (開催日次, 月日, 曜日, 使用コース, クッション値の測定時刻, クッション値, 含水率の測定時刻,
       芝ゴール前, 芝4コーナー, ダートゴール前, ダート4コーナー)。金曜日の測定は開催日次が空。
      クッション値のない年は、使用コース・クッション値の欄がない。
"""
from __future__ import annotations

import datetime as dt
import re
import sqlite3
import unicodedata
from pathlib import Path

VENUE_FILES = {"sapporo": "札幌", "hakodate": "函館", "fukushima": "福島", "niigata": "新潟", "tokyo": "東京",
               "nakayama": "中山", "chukyo": "中京", "kyoto": "京都", "hanshin": "阪神", "kokura": "小倉"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS baba (
    venue TEXT NOT NULL, date TEXT NOT NULL, day_no INTEGER, weekday TEXT, course TEXT,
    cushion_time TEXT, cushion REAL, moisture_time TEXT,
    turf_goal REAL, turf_4c REAL, dirt_goal REAL, dirt_4c REAL, pdf TEXT,
    PRIMARY KEY (venue, date)
);
-- 縦に並べた形(芝/ダート・場所ごと)。クッション値は芝のゴール前の行にだけ入れる
CREATE VIEW IF NOT EXISTS baba_long AS
    SELECT venue, date, day_no, moisture_time AS measured_time, '芝' AS surface, 'ゴール前' AS place, turf_goal AS moisture,
           cushion, cushion_time FROM baba
    UNION ALL SELECT venue, date, day_no, moisture_time, '芝', '4コーナー', turf_4c, NULL, NULL FROM baba
    UNION ALL SELECT venue, date, day_no, moisture_time, 'ダート', 'ゴール前', dirt_goal, NULL, NULL FROM baba
    UNION ALL SELECT venue, date, day_no, moisture_time, 'ダート', '4コーナー', dirt_4c, NULL, NULL FROM baba;
"""

WEEKDAYS = "月火水木金土日"
_NUM = r"(\d+(?:\.\d+)?|[-－―])"


def _norm(text: str) -> str:
    """全角の数字・記号を半角に。"""
    return unicodedata.normalize("NFKC", text)


def _val(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def parse_rows(text: str, year: int) -> list[dict]:
    """1日1行の表を読む。"""
    out = []
    pat = re.compile(r"^(?:第\s*(\d+)\s*日\s+)?(\d{1,2})月\s*(\d{1,2})日\s+(.)曜日\s+(.*)$")
    for line in _norm(text).splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        day_no, mon, day, wd, rest = m.groups()
        toks = rest.split()
        times = [i for i, t in enumerate(toks) if re.fullmatch(r"\d{1,2}:\d{2}", t)]
        row = {"date": dt.date(year, int(mon), int(day)).isoformat(), "day_no": int(day_no) if day_no else None,
               "weekday": wd, "course": None, "cushion_time": None, "cushion": None, "moisture_time": None}
        if len(times) == 2:  # クッション値あり: コース 時刻 値 時刻 含水率4つ
            a, b = times
            row["course"] = " ".join(toks[:a]) or None
            row["cushion_time"], row["cushion"] = toks[a], _val(toks[a + 1]) if a + 1 < b else None
            row["moisture_time"], vals = toks[b], toks[b + 1:]
        elif len(times) == 1:
            row["moisture_time"], vals = toks[times[0]], toks[times[0] + 1:]
        else:
            vals = toks
        vals = [_val(v) for v in vals if re.fullmatch(_NUM, v)]
        if len(vals) != 4:
            row["_bad"] = line
            vals = (vals + [None] * 4)[:4]
        row["turf_goal"], row["turf_4c"], row["dirt_goal"], row["dirt_4c"] = vals
        out.append(row)
    return out


def parse_weeks(text: str) -> list[dict]:
    """週ごとの表を読む。週の始まりは日付の行(どちらかの形):
      「2018年11月30日から12月2日の含水率」「2018年7月27日(金)から29日(日)」
      「第1日・第2日(2020年10月9日~11日)」
    そのあとに曜日の行(「場所 金曜日 土曜日 日曜日」または「金曜日 土曜日 日曜日」)と、
    「芝コース含水率 ゴール前 …」「(パーセント) 4コーナー …」「ダートコース含水率 ゴール前 …」「… 4コーナー …」、
    クッション値のある年は「芝コースクッション値 …」(含水率の前にも後にも来る)。
    """
    lines = [l.strip() for l in _norm(text).splitlines()]
    starts = []
    for i, line in enumerate(lines):
        m = (re.match(r"^(\d{4})年(\d{1,2})月(\d{1,2})日(?:\(.\))?から", line)
             or re.search(r"\((\d{4})年(\d{1,2})月(\d{1,2})日[~〜～]", line))
        if m:
            starts.append((i, dt.date(int(m[1]), int(m[2]), int(m[3]))))
    out = []
    for b, (i, start) in enumerate(starts):
        block = lines[i + 1:starts[b + 1][0] if b + 1 < len(starts) else len(lines)]
        days, vals, cur = None, {}, None
        for l in block:
            if days is None and re.search(r"曜日", l) and not re.search(r"\d", l):
                days = re.findall(r"(.)曜日", l)
                continue
            if "クッション値" in l:
                vals["cushion"] = [_val(n) for n in re.findall(_NUM, l.split("クッション値", 1)[1])]
                continue
            if "芝コース含水率" in l:
                cur = "turf"
            elif "ダートコース含水率" in l:
                cur = "dirt"
            place = "goal" if "ゴール前" in l else "4c" if "4コーナー" in l else None
            if place is None or cur is None:
                continue
            nums = re.findall(_NUM, l.split("ゴール前" if place == "goal" else "4コーナー", 1)[1])
            vals[f"{cur}_{place}"] = [_val(n) for n in nums]
        if not days:
            continue
        d = start - dt.timedelta(days=1)
        for j, wd in enumerate(days):
            # 列の日付は、前の列の次の日から、曜日が合う日(日曜の開催が火曜に延びた週もある)
            d = next((d + dt.timedelta(days=k) for k in range(1, 8) if WEEKDAYS[(d + dt.timedelta(days=k)).weekday()] == wd),
                     d + dt.timedelta(days=1))
            row = {"date": d.isoformat(), "day_no": None, "weekday": wd, "course": None, "cushion_time": None,
                   "cushion": None, "moisture_time": None}
            if WEEKDAYS[d.weekday()] != wd:
                row["_bad"] = f"曜日が合わない {d} {wd}"
            if "cushion" in vals:
                c = vals["cushion"]
                row["cushion"] = c[j] if j < len(c) else None
                if j >= len(c):
                    row["_bad"] = f"クッション値が足りない {d}"
            for k in ("turf_goal", "turf_4c", "dirt_goal", "dirt_4c"):
                v = vals.get(k, [])
                row[k] = v[j] if j < len(v) else None
                if j >= len(v):
                    row["_bad"] = f"{k} の値が足りない {d}"
            out.append(row)
    return out


def parse_text(text: str, year: int) -> list[dict]:
    rows = parse_rows(text, year)
    return rows if rows else parse_weeks(text)


def pdf_text(path: Path) -> str:
    import pdfplumber  # 読むときだけ要る

    with pdfplumber.open(path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)


def venue_year_of(path: Path) -> tuple[str, int] | None:
    m = re.search(r"(\d{4})pdf_([a-z]+)\d*\.pdf$", path.name)
    if not m or m[2] not in VENUE_FILES:
        return None
    return VENUE_FILES[m[2]], int(m[1])


def import_pdfs(conn: sqlite3.Connection, pdf_dir: str | Path) -> dict:
    """フォルダの PDF をすべて読んで baba に入れる(同じ競馬場・日付は入れ替え)。読めた行数と、読めなかったものを返す。"""
    conn.executescript(SCHEMA)
    stats = {"pdfs": 0, "rows": 0, "bad_rows": [], "empty_pdfs": []}
    for path in sorted(Path(pdf_dir).glob("*.pdf")):
        vy = venue_year_of(path)
        if not vy:
            continue
        venue, year = vy
        rows = parse_text(pdf_text(path), year)
        stats["pdfs"] += 1
        if not rows:
            stats["empty_pdfs"].append(path.name)
            continue
        for r in rows:
            if "_bad" in r:
                stats["bad_rows"].append((path.name, r.pop("_bad")))
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO baba VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(venue, r["date"], r["day_no"], r["weekday"], r["course"], r["cushion_time"], r["cushion"],
                  r["moisture_time"], r["turf_goal"], r["turf_4c"], r["dirt_goal"], r["dirt_4c"], path.name)
                 for r in rows])
        stats["rows"] += len(rows)
    return stats
