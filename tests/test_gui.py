import datetime as dt

from keibaai import gui
from keibaai.parse import ParsedRace, Race, Result

from test_model import synthetic_db


def test_summary_and_tables():
    conn = synthetic_db(n_days=3, races_per_day=2)
    s = gui.summary(conn)
    assert s["races"] == 6 and s["results"] == 60 and 20 <= s["horses"] <= 60
    assert s["first"] == "2025-01-04" and s["last"] == "2025-01-18"
    assert gui.races_by_month(conn)["races"].sum() == 6
    assert gui.recent_races(conn, limit=3)["runners"].tolist() == [10, 10, 10]
    assert gui.date_bounds(conn) == (dt.date(2025, 1, 4), dt.date(2025, 1, 18))


def test_summary_empty():
    from keibaai import db
    conn = db.connect(":memory:")
    assert gui.summary(conn)["races"] == 0
    assert gui.date_bounds(conn) == (None, None)


def test_race_days_match_cli():
    # 2024-11-22(金)〜11-25(月): 土日月の3日
    days = gui.race_days(dt.date(2024, 11, 22), dt.date(2024, 11, 25))
    assert [d.isoformat() for d in days] == ["2024-11-23", "2024-11-24", "2024-11-25"]


class FakeFetcher:
    def race_ids_on(self, date):
        return [f"{date}01"]

    def race_html(self, race_id):
        return race_id


def test_collect_reports_progress(monkeypatch):
    from keibaai import cli, db
    monkeypatch.setattr(cli, "parse_race", lambda html, race_id: ParsedRace(
        Race(race_id=race_id, date="2024-11-23"), [Result(race_id=race_id, number=1, finish=1)]))
    conn = db.connect(":memory:")
    seen = []
    n = gui.collect(conn, FakeFetcher(), dt.date(2024, 11, 23), dt.date(2024, 11, 24),
                    lambda i, total, d: seen.append((i, total)))
    assert n == 2
    assert seen == [(0, 2), (1, 2), (2, 2)]


def test_train_reports_table():
    conn = synthetic_db()
    report = gui.train_reports(conn, "2026-07-01")
    assert report["モデル"].tolist()[0] .startswith("基準")
    assert len(report) == 3


def test_preset_range():
    mon = dt.date(2024, 11, 25)
    assert gui.preset_range("先週末", mon) == (dt.date(2024, 11, 23), dt.date(2024, 11, 25))
    wed = dt.date(2024, 11, 27)
    assert gui.preset_range("先週末", wed) == (dt.date(2024, 11, 23), dt.date(2024, 11, 25))
    sat = dt.date(2024, 11, 30)  # 当日はまだ結果が出ていないので前の週末
    assert gui.preset_range("先週末", sat) == (dt.date(2024, 11, 23), dt.date(2024, 11, 25))
    assert gui.preset_range("今年", wed) == (dt.date(2024, 1, 1), wed)


def test_next_race_day():
    assert gui.next_race_day(dt.date(2026, 10, 12)) == dt.date(2026, 10, 12)  # 月曜(祝日開催がある)
    assert gui.next_race_day(dt.date(2026, 10, 14)) == dt.date(2026, 10, 17)  # 水曜 → 次の土曜
    assert gui.next_race_day(dt.date(2026, 10, 17)) == dt.date(2026, 10, 17)


def test_entries_summary_and_predict_table():
    from test_entries import _entries_for_future

    conn = synthetic_db(n_days=40)
    assert gui.entries_summary(conn, "2030-01-05")["races"] == 0
    _entries_for_future(conn, "2030-01-05")
    s = gui.entries_summary(conn, "2030-01-05")
    assert (s["races"], s["horses"], s["with_odds"]) == (1, 10, 10)


def test_day_races_and_race_entries():
    from test_entries import _entries_for_future

    conn = synthetic_db(n_days=40)
    _entries_for_future(conn, "2030-01-05")
    races = gui.day_races(conn, "2030-01-05")
    assert races[["venue", "round", "runners"]].values.tolist() == [["東京", 1, 10]]
    one = gui.race_entries(conn, "202609990101")
    assert one["number"].tolist() == list(range(1, 11)) and "p_win_market" not in one
    pred = gui.predict_table(conn, dt.date(2030, 1, 5))
    with_pred = gui.race_entries(conn, "202609990101", pred)
    assert len(with_pred) == 10 and with_pred["ev_market"].notna().all()


def test_race_choices_merge_fetched_and_listed():
    from test_entries import _entries_for_future

    conn = synthetic_db(n_days=3)
    _entries_for_future(conn, "2030-01-05")
    ch = gui.race_choices(conn, "2030-01-05", ["202608010111", "202609990101", "202601010102"])
    # 場コード順(01 札幌 → 08 京都 → 09 阪神)、その中はレース番号順
    assert ch[["venue", "round", "fetched"]].values.tolist() == [
        ["札幌", 2, False], ["京都", 11, False], ["阪神", 1, True]]
    assert ch.set_index("race_id").loc["202609990101", "fetched"]
    assert gui.race_choices(conn, "2030-01-06").empty


def test_add_marks_by_win_probability():
    import pandas as pd

    t = pd.DataFrame({
        "race_id": ["A"] * 10 + ["B"] * 5,
        "number": list(range(1, 11)) + list(range(1, 6)),
        "p_win_market": [0.05, 0.3, 0.2, 0.1, 0.08, 0.07, 0.06, 0.06, 0.04, 0.04] + [None] * 5,
        "p_win_ability": [0.1] * 10 + [0.1, 0.4, 0.3, 0.15, 0.05],
        "ev_ability": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.5, 0.9] + [0.2, 0.5, 0.5, 0.5, 0.4],
        "ev_market": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.2, 0.9] + [None] * 5,  # ★はオッズ込みの期待値で付く
        "win_odds": [10.0] * 8 + [25.0, 60.0] + [5.0] * 5,
    })
    star = gui.add_marks(t).set_index(["race_id", "number"])["star"]
    assert star["A", 9] == "★" and star["A", 10] == ""  # 期待値0.9は基準未満
    assert (star["B"] == "").all()  # オッズ込みの値がないレースには付かない
    t.loc[8, "win_odds"] = 40.0
    assert gui.add_marks(t).set_index(["race_id", "number"])["star"]["A", 9] == ""  # 30倍を超える
    t.loc[8, "win_odds"] = 25.0
    m = gui.add_marks(t).set_index(["race_id", "number"])["mark"]
    assert [m["A", i] for i in (2, 3, 4, 5, 6)] == ["◎", "○", "▲", "△", "△"]
    assert m["A", 9] == "☆" and m["A", 1] == ""
    # オッズのないレースは能力の勝率で。9 頭以下は △ が 1 頭、期待値 1 以上がいなければ ☆ なし
    assert [m["B", i] for i in (2, 3, 4, 1)] == ["◎", "○", "▲", "△"] and m["B", 5] == ""


def test_backtest_run_returns_table_and_curves():
    conn = synthetic_db(n_days=60, races_per_day=4)
    n_races, table, settled = gui.backtest_run(conn, "2025-09-01", False, 1.0, None, 100)
    assert n_races > 0 and len(table) == 4 and set(settled) == set(table["rule"])
    assert settled[table["rule"].iloc[1]]["balance"].iloc[-1] == table["profit"].iloc[1]


def test_race_head_html_survives_missing_fields():
    import pandas as pd

    full = pd.Series({"venue": "京都", "round": 11, "name": "スワンS", "course_type": "芝", "direction": "右",
                      "distance": 1400, "ground": "良", "start_time": "15:45", "runners": 18})
    out = gui.race_head_html(full)
    assert "京都11R" in out and "スワンS" in out and "芝・右回り 1400m ・ 良 ・ 15:45 発走 ・ 18頭" in out
    assert gui.course_text(pd.Series({"course_type": "芝", "direction": "直線", "distance": 1000})) == "芝・直線 1000m"
    assert gui.course_text(pd.Series({"course_type": "障", "direction": None, "distance": 2910})) == "障 2910m"
    empty = pd.Series({"venue": None, "round": 1, "name": float("nan"), "course_type": None, "distance": float("nan"),
                       "ground": float("nan"), "start_time": float("nan"), "runners": float("nan")})
    out = gui.race_head_html(empty)
    assert "1R" in out and "nan" not in out and "None" not in out


def test_eval_start_bounds():
    # 長い期間: 最初の1年は選ばせず、初期値は最後の1年前
    assert gui.eval_start_bounds(dt.date(2019, 1, 5), dt.date(2026, 10, 12)) == (dt.date(2020, 1, 5), dt.date(2025, 10, 12))
    # 2年に満たない期間: 前半は選ばせず、初期値は最も早い日
    assert gui.eval_start_bounds(dt.date(2025, 1, 4), dt.date(2025, 1, 18)) == (dt.date(2025, 1, 11), dt.date(2025, 1, 11))


def test_course_text_names_turn_inner_outer_and_rail():
    import pandas as pd

    s = lambda **kw: pd.Series({"venue": "京都", "course_type": "芝", "distance": 1600, **kw})
    assert gui.course_text(s(direction="右", course="外", rail="A")) == "芝・右回り・外回り・Aコース 1600m"
    assert gui.course_text(s(direction="右", course=None, rail=None)) == "芝・右回り 1600m"
    assert gui.course_text(s(course_type="障", distance=3170, direction=None, course="外-内")) == "障・外回り→内回り 3170m"
    assert gui.course_text(s(direction="右", course="内2周", rail="C")) == "芝・右回り・内回り2周・Cコース 1600m"
    out = gui.race_head_html(s(round=11, direction="右", course="外", rail="B", ground="良"))
    assert "京都11R" in out and "芝・右回り・外回り・Bコース 1600m ・ 良" in out