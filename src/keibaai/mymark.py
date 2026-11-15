"""netkeiba のマイ印を付けるためのリンク。

印は netkeiba の出馬表のアドレスの # 以降に載せる(例: ``#keibaai=1-1,7-2``。馬番-印の番号)。
# 以降はサーバーへ送られない。ブラウザの拡張 Tampermonkey に入れた ``scripts/netkeiba-mymark.user.js`` がそれを読み、
手で付けるときと同じプルダウン操作をする。認証情報は KeibaAI 側で扱わない。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

# netkeiba の印プルダウンの値(option の value の "_" より後)。★ に当たる印はないので付けない。
MARK_IDS = {"◎": 1, "○": 2, "◯": 2, "▲": 3, "△": 4, "☆": 5}


def mark_url(race_id: str, marks: Mapping[int, str] | Iterable[tuple[int, str]]) -> str:
    """出馬表のアドレスに印を載せて返す。印のない馬・netkeiba にない印は飛ばす。"""
    items = marks.items() if isinstance(marks, Mapping) else marks
    parts = [f"{int(n)}-{MARK_IDS[m]}" for n, m in items if m in MARK_IDS]
    url = SHUTUBA_URL.format(race_id=race_id)
    return f"{url}#keibaai={','.join(parts)}" if parts else url
