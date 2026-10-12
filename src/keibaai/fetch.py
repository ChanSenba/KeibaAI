"""netkeiba からの取得。個人利用・低頻度・キャッシュ済みページは再取得しない。"""
from __future__ import annotations

import datetime as dt
import os
import random
import re
import time
from pathlib import Path

import requests

from .parse import JRA_PLACE_CODES

RACE_URL = "https://db.netkeiba.com/race/{race_id}/"
RACE_LIST_URL = "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date}"
PEDIGREE_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"
SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
ODDS_URL = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type={odds_type}&action=init"
MIN_INTERVAL_SEC = 3.0  # サーバーに負荷をかけないための下限
JITTER_SEC = 2.0  # 3〜5秒の間でばらつかせ、アクセスが一定の拍で集中しないようにする


def build_user_agent(contact: str | None = None) -> str:
    """ブラウザを装わず、何のプログラムかが分かる UA。連絡先は環境変数 KEIBAAI_CONTACT で任意に付けられる。"""
    contact = contact if contact is not None else os.environ.get("KEIBAAI_CONTACT", "")
    base = "keibaai/0.1 (personal-use research script"
    return f"{base}; contact: {contact})" if contact else f"{base})"


USER_AGENT = build_user_agent()


def is_race_time(now: dt.datetime) -> bool:
    """土日の日中(JRA の開催時間帯)。一般の利用者が集中するので、収集は避けたい。"""
    return now.weekday() >= 5 and 9 <= now.hour < 17


class PageUnavailable(ValueError):
    """結果表のないページ(中止・未開催・制限ページなど)。キャッシュしない。"""


class ResultNotPublished(PageUnavailable):
    """結果がまだ掲載されていない(日付が1970年の空ひな形)。アクセス制限とは区別する。"""


class Fetcher:
    def __init__(self, cache_dir: str | Path = "data/cache", min_interval: float = MIN_INTERVAL_SEC,
                 session: requests.Session | None = None, sleep=time.sleep, clock=time.monotonic,
                 jitter: float = JITTER_SEC, rng: random.Random | None = None):
        self.cache_dir = Path(cache_dir)
        self.min_interval = max(min_interval, MIN_INTERVAL_SEC)
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT  # requests 既定の UA は 400 を返される
        self._jitter, self._rng = max(jitter, 0.0), rng or random.Random()
        self._sleep, self._clock = sleep, clock
        self._last = None

    def _get(self, url: str, encoding: str | None = None) -> str:
        for attempt in range(4):
            if self._last is not None:
                wait = self.min_interval + self._rng.uniform(0, self._jitter) - (self._clock() - self._last)
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
        if "race_table_01" not in html:
            if "1970年" in html:
                raise ResultNotPublished(f"結果が未掲載です: {race_id}")
            raise PageUnavailable(f"結果表のないページです: {race_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return html

    def pedigree_html(self, horse_id: str) -> str:
        """5代血統表のページ。血統は変わらないので、一度取ったらキャッシュから読む。"""
        path = self.cache_dir / "ped" / f"{horse_id}.html"
        if path.exists():
            return path.read_text(encoding="utf-8")
        html = self._get(PEDIGREE_URL.format(horse_id=horse_id), encoding="euc-jp")
        if "blood_table" not in html:
            raise PageUnavailable(f"血統表のないページです: {horse_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return html

    def discard_pedigree(self, horse_id: str) -> None:
        (self.cache_dir / "ped" / f"{horse_id}.html").unlink(missing_ok=True)

    def shutuba_html(self, race_id: str) -> str:
        """出馬表。発走前に内容が変わる(取消・馬体重など)ので、キャッシュしない。"""
        html = self._get(SHUTUBA_URL.format(race_id=race_id), encoding="utf-8")
        if "馬番" not in html:
            raise PageUnavailable(f"出馬表のないページです: {race_id}")
        return html

    def odds_json(self, race_id: str, odds_type: int = 1) -> str:
        """オッズ(JSON)。type=1 は単勝・複勝、4 馬連、5 ワイド、6 馬単、7 三連複、8 三連単。
        発売前は空に近い応答になる。これもキャッシュしない。"""
        return self._get(ODDS_URL.format(race_id=race_id, odds_type=odds_type), encoding="utf-8")

    def discard(self, race_id: str) -> None:
        (self.cache_dir / "race" / f"{race_id}.html").unlink(missing_ok=True)

    def race_ids_on(self, date: str) -> list[str]:
        """date: YYYYMMDD。その日の JRA のレース ID を返す。"""
        html = self._get(RACE_LIST_URL.format(date=date), encoding="utf-8")
        return extract_jra_race_ids(html)


def extract_jra_race_ids(html: str) -> list[str]:
    ids = dict.fromkeys(re.findall(r"race_id=(\d{12})", html))
    return [i for i in ids if i[4:6] in JRA_PLACE_CODES]
