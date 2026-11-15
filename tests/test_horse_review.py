import pickle
import sqlite3

import pandas as pd

from keibaai import db, horse_review
from keibaai.factors import COLUMN


def _race(n=6):
    rows = []
    for i in range(1, n + 1):
        row = {"race_id": "r1", "number": i, "horse_id": f"h{i}", "horse_name": f"馬{i}", "mark": "◎" if i == 1 else "",
               "p_win_ability": (n + 1 - i) / 21, "finish": i if i != 1 else 5, "status": None,
               "passing": f"{i}-{i}", "last3f": 34.0 + i / 10, "final_pop": i, "final_odds": 2.0 * i, "win_odds": 2.0 * i}
        for k, col in enumerate(COLUMN.values()):
            row[col] = (i if k % 2 else n + 1 - i) / 10  # 偶数番目の要素は馬1が最上位、奇数番目は最下位
        rows.append(row)
    rows[-1]["finish"] = None
    rows[-1]["status"] = "取消"
    return pd.DataFrame(rows)


def test_race_rows_texts():
    fc = {"advantage": "前"}
    act = {"advantage": "後"}
    out = horse_review.race_rows(_race(), (fc, act, None))
    rows = {r["number"]: r for r in out["rows"]}
    t = rows[1]["text"]
    assert t.startswith("◎1 馬1(1番人気): 予想1位")
    assert "強み 馬の質1位" in t and "弱み 適性6位" in t
    assert "予想1位から5着と、予想を下回った" in t
    assert "前有利の読みが外れて後ろ有利" in t and "読みに合う位置取りが裏目" in t
    assert "取消" in rows[6]["text"]
    assert out["rows"][-1]["number"] == 6  # 着順なしは最後
    assert "予想上位3頭のうち3着以内は2頭" in out["summary"]
    assert "外れ(実際は後ろ有利)" in out["summary"]


def test_race_rows_without_factors():
    g = _race().drop(columns=list(COLUMN.values()))
    out = horse_review.race_rows(g, None)
    assert all("要素の記録なし" in r["text"] for r in out["rows"])
    assert "展開" not in out["rows"][0]["text"]


def test_backfill_factors(tmp_path):
    conn = db.connect(tmp_path / "k.db")
    conn.executemany("INSERT INTO predictions (race_id, number, date, p_win_ability) VALUES (?, ?, ?, ?)",
                     [("a", 1, "2026-11-15", 0.5), ("a", 2, "2026-11-15", 0.5), ("b", 1, "2026-11-15", 0.6),
                      ("b", 2, "2026-11-15", 0.4)])
    conn.commit()
    cache = pd.DataFrame({"race_id": ["a", "a", "b", "b"], "number": [1, 2, 1, 2],
                          "p_win_ability": [0.5, 0.5, 0.3, 0.7], **{c: [0.1, 0.2, 0.3, 0.4] for c in COLUMN.values()}})
    with open(tmp_path / "predict-2026-11-15-x.pkl", "wb") as f:
        pickle.dump(cache, f)
    assert horse_review.backfill_factors(conn, tmp_path) == {"2026-11-15": 2}
    got = dict(conn.execute("SELECT race_id || number, p_win_human FROM predictions").fetchall())
    assert got == {"a1": 0.1, "a2": 0.2, "b1": None, "b2": None}  # b は勝率が離れているので入れない


def test_backfill_later_within_tolerance(tmp_path):
    conn = db.connect(tmp_path / "k.db")
    conn.execute("INSERT INTO predictions (race_id, number, date, p_win_ability) VALUES ('a', 1, '2026-11-15', 0.5)")
    conn.commit()
    cache = pd.DataFrame({"race_id": ["a"], "number": [1], "p_win_ability": [0.505],
                          **{c: [0.1] for c in COLUMN.values()}})
    with open(tmp_path / "predict-2026-11-15-x.pkl", "wb") as f:
        pickle.dump(cache, f)
    horse_review.backfill_factors(conn, tmp_path)
    assert conn.execute("SELECT factors_later, p_win_human FROM predictions").fetchone() == (1, 0.1)


def test_save_predictions_keeps_factors(tmp_path):
    import datetime as dt

    from keibaai import weekly

    conn = db.connect(tmp_path / "k.db")
    t = _race().assign(date="2026-11-21", venue="京都", round=11, race_name="R", jockey="j", popularity=1,
                       p_win_market=0.1, ev_ability=1.0, ev_market=1.0)
    weekly.save_predictions(conn, t, dt.datetime(2026, 11, 20, 18))
    assert conn.execute("SELECT COUNT(*) FROM predictions WHERE p_win_human IS NOT NULL").fetchone()[0] == 6
    weekly.save_predictions(conn, t.drop(columns=list(COLUMN.values())), dt.datetime(2026, 11, 20, 19))
    assert isinstance(conn, sqlite3.Connection)
