"""調教(追い切り)。race.netkeiba.com/race/oikiri.html?race_id=<ID>&type=1 の表を読む。

1頭ぶんは、見出し行(短評)1行 + 調教1本ごとに1行。新しい調教が上。
タイムの列(li)は5個固定で、ウッドなどは左から 6F・5F・4F・3F・1F、坂路は 4F・3F・2F・1F(先頭が '-')。
未ログインで全頭ぶん出るのは評価(短評とA〜D)までで、タイムは一部の馬だけのことがある。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup


@dataclass
class TrainingRun:
    race_id: str
    number: int  # 馬番
    seq: int  # 0 が最終追い切り(いちばん新しい調教)
    horse_id: str | None = None
    date: str | None = None  # YYYY-MM-DD
    course: str | None = None  # ＣＷ・栗坂 など
    slope: int = 0  # 坂路なら 1
    ground: str | None = None
    rider: str | None = None
    f6: float | None = None
    f5: float | None = None
    f4: float | None = None
    f3: float | None = None
    f2: float | None = None  # 坂路だけ
    f1: float | None = None
    position: int | None = None  # 位置(コースの内外)
    load: str | None = None  # 脚色: 馬也・強め・一杯 など
    partner: str | None = None  # 併せ馬(先着・遅れ など)
    critic: str | None = None  # 短評: 気配上昇 など
    rank: str | None = None  # A〜D
    review: str | None = None  # 馬ごとの見出しの寸評(seq=0 の行にだけ入れる)


def _text(el) -> str | None:
    if el is None:
        return None
    t = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
    return t or None


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    m = re.match(r"\d+(?:\.\d+)?", s.strip())
    return float(m.group(0)) if m else None


def _times(td, slope: bool) -> dict[str, float | None]:
    vals = []
    for li in td.select("ul.TrainingTimeDataList > li") if td is not None else []:
        lap = li.find("span", class_="RapTime")
        if lap is not None:
            lap.extract()  # 括弧の区間ラップは除き、通過タイムだけ読む
        vals.append(_num(li.get_text(strip=True)))
    vals = (vals + [None] * 5)[:5]
    keys = (None, "f4", "f3", "f2", "f1") if slope else ("f6", "f5", "f4", "f3", "f1")
    return {k: v for k, v in zip(keys, vals) if k}


def _date(s: str | None) -> str | None:
    m = re.search(r"(\d{4})/(\d{2})/(\d{2})", s or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def parse_oikiri(html: str, race_id: str) -> list[TrainingRun]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.OikiriTable")
    if table is None:
        raise ValueError(f"調教の表が見つかりません: {race_id}")
    out: list[TrainingRun] = []
    number = horse_id = review = None
    seq = 0
    for tr in table.select("tr.HorseList"):
        if (td := tr.find("td", class_="Umaban")) is not None:
            number = int(_num(_text(td)) or 0) or None
            a = tr.select_one(".Horse_Name a[href]")
            m = re.search(r"/horse/(\w+)/?$", a["href"]) if a else None
            horse_id = m.group(1) if m else None
            review = _text(tr.find("td", class_="TrainingReview_Cell"))
            seq = 0
        if number is None or tr.find("td", class_="Training_Day") is None and tr.find(
                "td", class_="Training_Critic") is None:
            continue  # 見出し行だけ
        cells = tr.find_all("td", recursive=False)
        day = tr.find("td", class_="Training_Day")
        i = cells.index(day) if day in cells else -1
        course = _text(cells[i + 1]) if i >= 0 and len(cells) > i + 3 else None
        slope = bool(course and "坂" in course)
        time_td = tr.find("td", class_="TrainingTimeData")
        pos_td = time_td.find_next_sibling("td") if time_td is not None else None
        rank_td = tr.find("td", class_=re.compile(r"^Rank_"))
        run = TrainingRun(
            race_id=race_id, number=number, seq=seq, horse_id=horse_id, date=_date(_text(day)),
            course=course, slope=int(slope),
            ground=_text(cells[i + 2]) if course else None, rider=_text(cells[i + 3]) if course else None,
            position=int(_num(_text(pos_td))) if _num(_text(pos_td)) is not None else None,
            load=_text(tr.find("td", class_="TrainingLoad")),
            partner=" / ".join(_text(p) for p in time_td.select(".Comment_Cell p")) or None
            if time_td is not None else None,
            critic=_text(tr.find("td", class_="Training_Critic")), rank=_text(rank_td),
            review=review if seq == 0 else None,
            **_times(time_td, slope),
        )
        out.append(run)
        seq += 1
    if not out:
        raise ValueError(f"調教の行がありません: {race_id}")
    return out
