"""血統。db.netkeiba.com/horse/ped/<馬ID>/ の5代血統表から、父・母・母の父・父の父を取り出す。"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from . import db
from .fetch import Fetcher, PageUnavailable

MAX_CONSECUTIVE_FAILURES = 5


@dataclass
class Pedigree:
    horse_id: str
    sire_id: str | None = None
    sire: str | None = None
    dam_id: str | None = None
    dam: str | None = None
    damsire_id: str | None = None  # 母の父
    damsire: str | None = None
    sire_sire_id: str | None = None  # 父の父
    sire_sire: str | None = None
    sire_line: str | None = None  # 父の系統('Halo系' など)


def _cell(td) -> tuple[str | None, str | None]:
    a = td.find("a", href=re.compile(r"/horse/(?!ped/|sire/|mare/)\w+/?$"))
    if a is None:
        return None, None
    m = re.search(r"/horse/(\w+)/?$", a["href"])
    return (m.group(1) if m else None), re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip() or None


def parse_pedigree(html: str, horse_id: str) -> Pedigree:
    """5代血統表(table.blood_table)を読む。

    1列目の rowspan=16 のセルが父(1つ目)と母(2つ目)。rowspan=8 のセルが祖父母で、
    父と母の間にある最初のものが父の父、母のあとにある最初のものが母の父。
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.blood_table")
    if table is None:
        raise ValueError(f"血統表が見つかりません: {horse_id}")
    cells = [(td.get("rowspan"), td) for td in table.find_all("td")]
    parents = [i for i, (rs, _) in enumerate(cells) if rs == "16"]
    if len(parents) < 2:
        raise ValueError(f"父母のセルが見つかりません: {horse_id}")
    sire_i, dam_i = parents[0], parents[1]

    def first_grand(lo: int, hi: int):
        return next((td for rs, td in cells[lo:hi] if rs == "8"), None)

    out = Pedigree(horse_id=horse_id)
    out.sire_id, out.sire = _cell(cells[sire_i][1])
    out.dam_id, out.dam = _cell(cells[dam_i][1])
    if m := re.search(r"(\S+系)", cells[sire_i][1].get_text(" ", strip=True)):
        out.sire_line = m.group(1)
    if (td := first_grand(sire_i + 1, dam_i)) is not None:
        out.sire_sire_id, out.sire_sire = _cell(td)
    if (td := first_grand(dam_i + 1, len(cells))) is not None:
        out.damsire_id, out.damsire = _cell(td)
    if out.sire_id is None and out.dam_id is None:
        raise ValueError(f"父母が読み取れません: {horse_id}")
    return out


def save_pedigree(conn: sqlite3.Connection, p: Pedigree) -> None:
    with conn:
        db._upsert(conn, "pedigree", Pedigree, p)


def pending_horse_ids(conn: sqlite3.Connection) -> list[str]:
    """血統をまだ取っていない馬。出馬表の馬(予想に要る)を先に、残りは最近走った順に並べる。"""
    rows = conn.execute(
        """
        SELECT horse_id, MAX(prio) AS prio, MAX(d) AS d FROM (
            SELECT e.horse_id AS horse_id, 1 AS prio, c.date AS d FROM entries e JOIN entry_races c USING (race_id)
            UNION ALL
            SELECT r.horse_id, 0, c.date FROM results r JOIN races c USING (race_id)
        ) WHERE horse_id IS NOT NULL AND horse_id NOT IN (SELECT horse_id FROM pedigree)
        GROUP BY horse_id ORDER BY prio DESC, d DESC, horse_id
        """
    ).fetchall()
    return [r[0] for r in rows]


def collect_pedigrees(conn: sqlite3.Connection, fetcher: Fetcher, limit: int = 0, log=print) -> int:
    """血統を取って保存する(取得済みはキャッシュから読む)。保存した頭数を返す。途中で止めても続きから再開できる。"""
    ids = pending_horse_ids(conn)
    if limit:
        ids = ids[:limit]
    failures = n = 0
    for horse_id in ids:
        try:
            save_pedigree(conn, parse_pedigree(fetcher.pedigree_html(horse_id), horse_id))
        except ValueError as e:  # PageUnavailable も含む
            fetcher.discard_pedigree(horse_id)
            failures += 1
            log(f"skip {horse_id}: {e}")
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{MAX_CONSECUTIVE_FAILURES}件連続で取得に失敗しました。"
                    "アクセス制限の可能性があるので、時間をおいて再実行してください") from e
            continue
        failures = 0
        n += 1
        if n % 50 == 0:
            log(f"saved {n}/{len(ids)}")
    return n


def reparse_pedigrees(conn: sqlite3.Connection, cache_dir: str | Path) -> tuple[int, int]:
    ok = bad = 0
    for path in sorted((Path(cache_dir) / "ped").glob("*.html")):
        try:
            save_pedigree(conn, parse_pedigree(path.read_text(encoding="utf-8"), path.stem))
            ok += 1
        except ValueError as e:
            bad += 1
            print(f"skip {path.stem}: {e}")
    return ok, bad
