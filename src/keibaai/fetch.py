"""netkeiba からの取得。個人利用・低頻度・キャッシュ済みページは再取得しない。"""
from __future__ import annotations

import re
import time
from pathlib import Path

import requests

from .parse import JRA_PLACE_CODES

RACE_URL = "https://db.netkeiba.com/race/{race_id}/"
RACE_LIST_URL = "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date}"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) keibaai-personal-use"
MIN_INTERVAL_SEC = 3.0  # 規約上、サーバーに負荷をかけないための下限


class Fetcher:
    def __init__(self, cache_dir: str | Path = "data/cache", min_interval: float = MIN_INTERVAL_SEC,
                 session: requests.Session | None = None, sleep=time.sleep, clock=time.monotonic):
        self.cache_dir = Path(cache_dir)
        self.min_interval = max(min_interval, MIN_INTERVAL_SEC)
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT  # requests 既定の UA は 400 を返される
        self._sleep, self._clock = sleep, clock
        self._last = None

    def _get(self, url: str, encoding: str | None = None) -> str:
        for attempt in range(4):
            if self._last is not None:
                wait = self.min_interval - (self._clock() - self._last)
                if wait > 0:
                    self._sleep(wait)
            resp = self.session.get(url, timeout=30)
            self._last = self._clock()
            if resp.status_code in (429, 503):
                self._sleep(30 * (attempt + 1))  # 取得を間引いて待つ
                continue
            resp.raise_for_status()
            if encoding:
                resp.encoding = encoding
            return resp.text
        raise RuntimeError(f"取得に失敗しました(混雑・制限の可能性): {url}")

    def race_html(self, race_id: str) -> str:
        path = self.cache_dir / "race" / f"{race_id}.html"
        if path.exists():
            return path.read_text(encoding="utf-8")
        html = self._get(RACE_URL.format(race_id=race_id), encoding="euc-jp")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return html

    def race_ids_on(self, date: str) -> list[str]:
        """date: YYYYMMDD。その日の JRA のレース ID を返す。"""
        html = self._get(RACE_LIST_URL.format(date=date), encoding="utf-8")
        return extract_jra_race_ids(html)


def extract_jra_race_ids(html: str) -> list[str]:
    ids = dict.fromkeys(re.findall(r"race_id=(\d{12})", html))
    return [i for i in ids if i[4:6] in JRA_PLACE_CODES]
