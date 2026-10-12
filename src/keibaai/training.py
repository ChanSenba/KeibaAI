"""調教(追い切り)。race.netkeiba.com/race/oikiri.html?race_id=<ID>&type=1 の表を読む。

1頭ぶんは、見出し行(短評)1行 + 調教1本ごとに1行。新しい調教が上。
タイムの列(li)は5個固定で、ウッドなどは左から 6F・5F・4F・3F・1F、坂路は 4F・3F・2F・1F(先頭が '-')。
未ログインで全頭ぶん出るのは、type を付けないページの評価(短評とA〜D)まで。そのページは1頭1行で、
日付・コース・タイムはない。type=1 のタイムは未ログインだと3頭ぶんしか出ないので、いまは使わない。
ランクのセルの class は 'Rank_B' のことも 'Rank_気配上昇' のこともあるので、A〜D はセルの文字で読む。
"""
from __future__ import annotations

import datetime as dt
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from . import db
from .fetch import Fetcher, PageUnavailable

MAX_CONSECUTIVE_FAILURES = 5


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
            critic=_text(tr.find("td", class_="Training_Critic")),
            rank=r if (r := _text(rank_td)) in ("A", "B", "C", "D", "E") else None,
            review=review if seq == 0 else None,
            **_times(time_td, slope),
        )
        out.append(run)
        seq += 1
    if not out:
        raise ValueError(f"調教の行がありません: {race_id}")
    return out


def save_training(conn: sqlite3.Connection, race_id: str, runs: list[TrainingRun]) -> None:
    """1レースぶんを置き換えて保存する。runs が空でも「読んだ」印は残す(表のないレースを取り直さないため)。"""
    with conn:
        conn.execute("DELETE FROM training WHERE race_id = ?", (race_id,))
        for r in runs:
            db._upsert(conn, "training", TrainingRun, r)
        conn.execute("INSERT OR REPLACE INTO training_fetched VALUES (?, ?, ?)",
                     (race_id, len({r.number for r in runs}), dt.datetime.now().isoformat(timespec="seconds")))


def pending_race_ids(conn: sqlite3.Connection) -> list[str]:
    """調教をまだ読んでいない、結果のあるレース。新しい順(予想に近い時期のデータから効かせる)。"""
    rows = conn.execute(
        "SELECT race_id FROM races WHERE race_id NOT IN (SELECT race_id FROM training_fetched) "
        "ORDER BY date DESC, race_id").fetchall()
    return [r[0] for r in rows]


def collect_training(conn: sqlite3.Connection, fetcher: Fetcher, limit: int = 0, log=print) -> int:
    """終わったレースの調教評価を取って保存する。保存したレース数を返す。途中で止めても続きから再開できる。"""
    ids = pending_race_ids(conn)
    if limit:
        ids = ids[:limit]
    failures = n = 0
    for race_id in ids:
        try:
            runs = parse_oikiri(fetcher.oikiri_html(race_id), race_id)
        except PageUnavailable as e:  # 表そのものがない(古いレースなど)。読んだ印だけ付けて先へ
            save_training(conn, race_id, [])
            failures += 1
            log(f"skip {race_id}: {e}")
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{MAX_CONSECUTIVE_FAILURES}件連続で調教の表がありませんでした。"
                    "アクセス制限の可能性があるので、時間をおいて再実行してください") from e
            continue
        except ValueError as e:  # 表はあるが読めない。壊れたキャッシュを残さず、印も付けない
            fetcher.discard_oikiri(race_id)
            log(f"skip {race_id}: {e}")
            continue
        failures = 0
        save_training(conn, race_id, runs)
        n += 1
        if n % 50 == 0:
            log(f"saved {n}/{len(ids)}")
    return n


def fetch_entry_training(conn: sqlite3.Connection, fetcher: Fetcher, date: str, log=print) -> int:
    """出馬表を取った日のレースの調教評価を取り直す(発走前は更新されるのでキャッシュしない)。"""
    ids = [r[0] for r in conn.execute("SELECT race_id FROM entry_races WHERE date = ? ORDER BY race_id", (date,))]
    n = 0
    for race_id in ids:
        try:
            save_training(conn, race_id, parse_oikiri(fetcher.oikiri_html(race_id, cache=False), race_id))
            n += 1
        except ValueError as e:
            log(f"skip {race_id}: {e}")
    return n


def reparse_training(conn: sqlite3.Connection, cache_dir: str | Path) -> tuple[int, int]:
    ok = bad = 0
    for path in sorted((Path(cache_dir) / "oikiri").glob("*.html")):
        try:
            save_training(conn, path.stem, parse_oikiri(path.read_text(encoding="utf-8"), path.stem))
            ok += 1
        except ValueError as e:
            bad += 1
            print(f"skip {path.stem}: {e}")
    return ok, bad
