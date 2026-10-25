"""予想印の定義。印の種類と並びは、ここだけで決める。"""
from __future__ import annotations

TOP_MARKS = ["◎", "○", "▲"]  # 勝率(や総合点)の高い順に付ける印。続く馬に △ を付ける
MARK_ORDER = TOP_MARKS + ["△", "☆"]  # 印の並び。表示・集計・買い方はこの順
MARK_RANK = {m: i for i, m in enumerate(MARK_ORDER)}


def top_marks(n_runners: int) -> list[str]:
    """1レースの上位に付ける印を、頭数に合わせて返す。◎ ○ ▲ のあと、10頭立て以上は △ を2頭、9頭立て以下は1頭。"""
    return (TOP_MARKS + ["△", "△"])[:len(TOP_MARKS) + (2 if n_runners >= 10 else 1)]
