"""netkeiba のレース結果ページ(db.netkeiba.com/race/<race_id>/)を解析する。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

# race_id = 年(4) + 場(2) + 回(2) + 日(2) + R(2)。場コード 01-10 が JRA。
JRA_PLACE_CODES = {f"{i:02d}" for i in range(1, 11)}
VENUES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


@dataclass
class Race:
    race_id: str
    name: str | None = None
    date: str | None = None  # YYYY-MM-DD
    venue: str | None = None
    round: int | None = None
    course_type: str | None = None  # 芝 / ダ / 障
    direction: str | None = None  # 左 / 右 / 直線
    distance: int | None = None
    weather: str | None = None
    ground: str | None = None  # 良 / 稍重 / 重 / 不良
    start_time: str | None = None
    lap_times: str | None = None  # 200m ごとのラップ('12.7 - 11.4 - ...')
    pace_first3f: float | None = None  # 前半3ハロン
    pace_last3f: float | None = None  # 後半3ハロン
    course: str | None = None  # 内回り・外回りの別(外 / 内 / 外-内 / 内2周 など)。別のない競馬場は None
    rail: str | None = None  # 芝の A〜E コース(仮柵の位置)。JRA-VAN から入る


@dataclass
class Result:
    race_id: str
    number: int
    finish: int | None = None
    status: str | None = None  # 着順が数字でないとき(中止・除外・取消など)
    bracket: int | None = None
    horse_id: str | None = None
    horse_name: str | None = None
    sex_age: str | None = None
    weight_carried: float | None = None
    jockey: str | None = None
    jockey_id: str | None = None
    time_sec: float | None = None
    margin: str | None = None
    passing: str | None = None
    last3f: float | None = None
    win_odds: float | None = None
    popularity: int | None = None
    body_weight: int | None = None
    body_weight_diff: int | None = None
    trainer: str | None = None
    trainer_id: str | None = None
    owner: str | None = None
    prize: float | None = None


@dataclass
class ParsedRace:
    race: Race
    results: list[Result] = field(default_factory=list)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _num(text: str | None, cast=float):
    if text is None:
        return None
    t = text.replace(",", "").strip()
    try:
        return cast(t)
    except ValueError:
        return None


def parse_time(text: str | None) -> float | None:
    """'2:25.5' -> 145.5"""
    m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", (text or "").strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + float(m.group(2))


def parse_body_weight(text: str | None) -> tuple[int | None, int | None]:
    """'510(+6)' -> (510, 6)"""
    m = re.fullmatch(r"(\d+)\(([+-]?\d+)\)", _squash(text))
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d+)", _squash(text))
    return (int(m.group(1)), None) if m else (None, None)


def _id_from_href(a, kind: str) -> str | None:
    if a is None:
        return None
    m = re.search(rf"/{kind}/(?:result/recent/)?(\w+)/?", a.get("href", ""))
    return m.group(1) if m else None


def _parse_race_info(soup: BeautifulSoup, race_id: str) -> Race:
    race = Race(race_id=race_id)
    race.venue = VENUES.get(race_id[4:6])
    race.round = _num(race_id[10:12], int)

    h1 = soup.select_one("dl.racedata dd h1")
    if h1:
        race.name = h1.get_text(strip=True)

    info = soup.select_one("dl.racedata dd p")
    info_text = _squash(info.get_text(" ") if info else "")
    m = re.search(r"(障|芝|ダ)(?:芝|ダート|ダ)*(左|右|直線)?\s*((?:外|内)(?:-(?:外|内))?(?:\d周)?)?(\d{3,4})m", info_text)
    if m:
        race.course_type, race.direction, race.course, race.distance = m.group(1), m.group(2), m.group(3), int(m.group(4))
    if m := re.search(r"天候:(\S+?)(?:/|$)", info_text):
        race.weather = m.group(1)
    if m := re.search(r"(?:芝|ダート|ダ):(良|稍重|重|不良)", info_text):
        race.ground = m.group(1)
    if m := re.search(r"発走:(\d{1,2}:\d{2})", info_text):
        race.start_time = m.group(1)

    small = soup.select_one("p.smalltxt")
    if small and (m := re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", small.get_text())):
        race.date = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    _parse_lap(soup, race)
    return race


def _parse_lap(soup: BeautifulSoup, race: Race) -> None:
    """ラップタイムの表(summary='ラップタイム')から、ラップと前後半3ハロンを取る。無いレースもある。"""
    table = soup.find("table", attrs={"summary": "ラップタイム"})
    if table is None:
        return
    for tr in table.find_all("tr"):
        th, td = tr.find("th"), tr.find("td")
        if th is None or td is None:
            continue
        label, text = _squash(th.get_text()), td.get_text(" ", strip=True).replace("\xa0", " ")
        if label == "ラップ":
            race.lap_times = _squash(text) or None
        elif label == "ペース" and (m := re.search(r"\((\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\)", text)):
            race.pace_first3f, race.pace_last3f = float(m.group(1)), float(m.group(2))
    laps = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", race.lap_times or "")]
    if race.pace_first3f is None and len(laps) >= 6:  # 括弧の表記がないときは、ラップから足す
        race.pace_first3f, race.pace_last3f = round(sum(laps[:3]), 1), round(sum(laps[-3:]), 1)


# ヘッダー(空白除去後) -> Result の属性。プレミアム限定列などは無視する。
_HEADERS = {
    "着順": "finish", "枠番": "bracket", "馬番": "number", "馬名": "horse_name",
    "性齢": "sex_age", "斤量": "weight_carried", "騎手": "jockey", "タイム": "time",
    "着差": "margin", "通過": "passing", "上り": "last3f", "単勝": "win_odds",
    "人気": "popularity", "馬体重": "body_weight", "調教師": "trainer",
    "馬主": "owner", "賞金(万円)": "prize",
}


def parse_race(html: str, race_id: str) -> ParsedRace:
    soup = BeautifulSoup(html, "lxml")
    race = _parse_race_info(soup, race_id)
    table = soup.select_one("table.race_table_01")
    if table is None:
        raise ValueError(f"結果表が見つかりません: {race_id}")

    rows = table.find_all("tr")
    headers = [_squash(th.get_text()) for th in rows[0].find_all(["th", "td"])]
    col = {name: i for i, h in enumerate(headers) if (name := _HEADERS.get(h))}
    if "number" not in col:
        raise ValueError(f"馬番の列が見つかりません: {race_id} {headers}")

    out = ParsedRace(race=race)
    for tr in rows[1:]:
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
        r = Result(race_id=race_id, number=number)
        finish_text = text("finish")
        r.finish = _num(finish_text, int)
        if r.finish is None:
            r.status = finish_text or None
        r.bracket = _num(text("bracket"), int)
        r.horse_name = text("horse_name")
        r.horse_id = _id_from_href(cell("horse_name").find("a") if cell("horse_name") else None, "horse")
        r.sex_age = text("sex_age")
        r.weight_carried = _num(text("weight_carried"))
        r.jockey = text("jockey")
        r.jockey_id = _id_from_href(cell("jockey").find("a") if cell("jockey") else None, "jockey")
        r.time_sec = parse_time(text("time"))
        r.margin = text("margin") or None
        r.passing = text("passing") or None
        r.last3f = _num(text("last3f"))
        r.win_odds = _num(text("win_odds"))
        r.popularity = _num(text("popularity"), int)
        r.body_weight, r.body_weight_diff = parse_body_weight(text("body_weight"))
        r.trainer = re.sub(r"^\[[東西地外]\]", "", _squash(text("trainer") or "")) or None  # 先頭の所属 [東] だけを取る
        r.trainer_id = _id_from_href(cell("trainer").find("a") if cell("trainer") else None, "trainer")
        r.owner = text("owner")
        r.prize = _num(text("prize"))
        out.results.append(r)
    return out
