"""これから走るレースの出馬表(race.netkeiba.com/race/shutuba.html)と単勝オッズの解析。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from .parse import VENUES, _id_from_href, _num, _squash


@dataclass
class EntryRace:
    race_id: str
    name: str | None = None
    date: str | None = None  # YYYY-MM-DD
    venue: str | None = None
    round: int | None = None
    course_type: str | None = None
    direction: str | None = None
    distance: int | None = None
    weather: str | None = None
    ground: str | None = None
    start_time: str | None = None


@dataclass
class Entry:
    race_id: str
    number: int
    bracket: int | None = None
    horse_id: str | None = None
    horse_name: str | None = None
    sex_age: str | None = None
    weight_carried: float | None = None
    jockey: str | None = None
    jockey_id: str | None = None
    trainer: str | None = None
    trainer_id: str | None = None
    body_weight: int | None = None
    body_weight_diff: int | None = None
    win_odds: float | None = None
    popularity: int | None = None


@dataclass
class ParsedEntries:
    race: EntryRace
    entries: list[Entry] = field(default_factory=list)


_HEADERS = {
    "枠": "bracket", "馬番": "number", "馬名": "horse_name", "性齢": "sex_age", "斤量": "weight_carried",
    "騎手": "jockey", "厩舎": "trainer", "馬体重(増減)": "body_weight", "馬体重": "body_weight", "人気": "popularity",
}


def _parse_body_weight(text: str | None) -> tuple[int | None, int | None]:
    """'496(+6)' -> (496, 6)、'496(前計不)' -> (496, None)、'計不' -> (None, None)"""
    m = re.match(r"\s*(\d{3})\s*(?:\(([+-]?\d+)\))?", text or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2)) if m.group(2) else None


def _parse_race_info(soup: BeautifulSoup, race_id: str) -> EntryRace:
    race = EntryRace(race_id=race_id, venue=VENUES.get(race_id[4:6]), round=_num(race_id[10:12], int))
    text = _squash(soup.get_text(" "))
    title = soup.title.get_text() if soup.title else ""

    name_el = soup.select_one(".RaceName")
    name = name_el.get_text(strip=True) if name_el else None
    if m := re.search(r"\((G[123]|GI{1,3}|L)\)", title):  # クラスはタイトル側にだけ付いている
        name = (name or "") + m.group(0)
    race.name = name or None

    if m := re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title + " " + text):
        race.date = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    if m := re.search(r"(障|芝|ダ)(?:芝|ダート|ダ)*(\d{3,4})m", text):
        race.course_type, race.distance = m.group(1), int(m.group(2))
        if d := re.search(r"\((左|右|直線)", text[m.end():m.end() + 12]):
            race.direction = d.group(1)
    if m := re.search(r"天候[:：]?(晴|曇|小雨|雨|小雪|雪)", text):
        race.weather = m.group(1)
    if m := re.search(r"馬場[:：]?(良|稍重|重|不良)", text):
        race.ground = m.group(1)
    if m := re.search(r"(\d{1,2}:\d{2})発走", text):
        race.start_time = m.group(1)
    return race


def parse_shutuba(html: str, race_id: str) -> ParsedEntries:
    soup = BeautifulSoup(html, "lxml")
    out = ParsedEntries(race=_parse_race_info(soup, race_id))

    table = None
    for t in soup.find_all("table"):
        first = t.find("tr")
        heads = [_squash(c.get_text()) for c in first.find_all(["th", "td"])] if first else []
        if "馬番" in heads and "馬名" in heads:
            table, headers = t, heads
            break
    if table is None:
        raise ValueError(f"出馬表が見つかりません: {race_id}")
    col = {name: i for i, h in enumerate(headers) if (name := _HEADERS.get(h))}

    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < len(headers):
            continue

        def cell(name):
            i = col.get(name)
            return tds[i] if i is not None else None

        def text(name):
            c = cell(name)
            return c.get_text(" ", strip=True) if c else None

        number = _num(text("number"), int)
        if number is None:
            continue
        e = Entry(race_id=race_id, number=number)
        e.bracket = _num(text("bracket"), int)
        e.horse_name = text("horse_name")
        e.horse_id = _id_from_href(cell("horse_name").find("a", href=re.compile("/horse/")) if cell("horse_name") else None, "horse")
        e.sex_age = text("sex_age")
        e.weight_carried = _num(text("weight_carried"))
        e.jockey = text("jockey")
        e.jockey_id = _id_from_href(cell("jockey").find("a", href=re.compile("/jockey/")) if cell("jockey") else None, "jockey")
        e.trainer = text("trainer")
        e.trainer_id = _id_from_href(cell("trainer").find("a", href=re.compile("/trainer/")) if cell("trainer") else None, "trainer")
        e.body_weight, e.body_weight_diff = _parse_body_weight(text("body_weight"))
        e.popularity = _num(text("popularity"), int)  # 発売前は「**」で None
        out.entries.append(e)
    return out


def parse_odds_json(text: str) -> dict[int, tuple[float | None, int | None]]:
    """api_get_jra_odds.html?type=1 の返り値から {馬番: (単勝オッズ, 人気)} を作る。

    data.odds["1"]["03"] = ["2.3", "0.0", "1"] の形(先頭が単勝オッズ、末尾が人気)。
    発売前などで取れないときは空の辞書を返す。
    """
    try:
        win = json.loads(text)["data"]["odds"]["1"]
    except (ValueError, KeyError, TypeError):
        return {}
    out = {}
    for k, v in win.items():
        try:
            odds = float(v[0])
        except (ValueError, TypeError, IndexError):
            continue
        pop = _num(v[2], int) if len(v) > 2 else None
        out[int(k)] = (odds if odds > 0 else None, pop)
    return out


def apply_odds(parsed: ParsedEntries, odds: dict[int, tuple[float | None, int | None]]) -> None:
    for e in parsed.entries:
        if e.number in odds:
            e.win_odds, pop = odds[e.number]
            if pop is not None:
                e.popularity = pop
