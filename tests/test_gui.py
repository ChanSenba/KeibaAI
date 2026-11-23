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


def test_load_evals_reads_sunday_reports_newest_first(tmp_path):
    import json

    for d in ("2026-11-08", "2026-11-15"):
        (tmp_path / f"{d}.json").write_text(json.dumps({"date": d, "test_from": "2025-11-15", "reports": [
            {"name": "基準:オッズ/1番人気", "n_races": 3000, "win_logloss": 0.21, "win_auc": 0.78, "top3_auc": None,
             "top1_hit_rate": 0.33, "top1_roi": 80.0},
            {"name": "オッズなし(能力のみ)", "n_races": 3000, "win_logloss": 0.22 if d.endswith("08") else 0.215,
             "win_auc": 0.80, "top3_auc": 0.78, "top1_hit_rate": 0.30, "top1_roi": 78.0}]}), encoding="utf-8")
    (tmp_path / "2026-11-01.json").write_text("{", encoding="utf-8")  # 書きかけは飛ばす
    (tmp_path / "2026-10-25.md").write_text("# 古い形式", encoding="utf-8")
    evals = gui.load_evals(tmp_path)
    assert [e["date"] for e in evals] == ["2026-11-15", "2026-11-08"]
    assert list(evals[0]["report"].columns[:2]) == ["モデル", "レース数"]
    hist = gui.eval_history(evals)
    assert hist["日付"].tolist() == ["2026-11-08", "2026-11-08", "2026-11-15", "2026-11-15"]
    assert gui.load_evals(tmp_path / "none") == [] and gui.eval_history([]).empty


def test_live_totals_weights_roi_by_bets():
    import pandas as pd

    trend = pd.DataFrame({"日付": ["2026-11-08", "2026-11-09"], "レース数": [24, 12], "◎1着": [6, 2],
                          "◎単勝回収率(%)": [100.0, 40.0], "期待値≧1の頭数": [30, 0], "期待値≧1の回収率(%)": [90.0, float("nan")]})
    t = gui.live_totals(trend)
    assert t["races"] == 36 and t["honmei_wins"] == 8 and t["days"] == 2
    assert abs(t["honmei_roi"] - 80.0) < 1e-9  # (24×100 + 12×40) / 36
    assert t["ev_bets"] == 30 and abs(t["ev_roi"] - 90.0) < 1e-9
    assert gui.live_totals(trend.iloc[:0])["races"] == 0


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
    # 期待値の高い馬の表のコース列は、予想の表ではなく出馬表から引く
    courses = gui.day_courses(conn, "2030-01-05")
    assert pred["race_id"].map(courses).str.endswith("m").all()


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
    # 印は能力のみの勝率の順(オッズ込みの勝率では並べない)。レース A は能力が全馬同じなので馬番順
    assert [m["A", i] for i in (1, 2, 3, 4, 5)] == ["◎", "○", "▲", "△", "△"]
    assert m["A", 9] == "☆" and m["A", 6] == ""  # ☆はオッズ込みの期待値で
    # 9 頭以下は △ が 1 頭、期待値 1 以上がいなければ ☆ なし
    assert [m["B", i] for i in (2, 3, 4, 1)] == ["◎", "○", "▲", "△"] and m["B", 5] == ""
    # 古い予想の表に p_win_mix(オッズを混ぜた勝率)があっても、印には使わない
    t["p_win_mix"] = [0.3, 0.05, 0.2, 0.1, 0.08, 0.07, 0.06, 0.06, 0.04, 0.04] + [None] * 5
    t.loc[:9, "p_win_ability"] = [0.05, 0.3, 0.2, 0.1, 0.08, 0.07, 0.06, 0.06, 0.04, 0.04]
    out = gui.add_marks(t).set_index(["race_id", "number"])
    assert [out["mark"]["A", i] for i in (2, 3, 4)] == ["◎", "○", "▲"] and out["star"]["A", 9] == "★"


def test_add_marks_confidence_mark_and_records():
    import pandas as pd

    t = pd.DataFrame({
        "race_id": ["A"] * 5 + ["B"] * 5,
        "number": list(range(1, 6)) * 2,
        "p_win_ability": [0.45, 0.25, 0.15, 0.1, 0.05] + [0.39, 0.3, 0.2, 0.06, 0.05],
        "p_win_market": [None] * 10,
        "ev_ability": [0.5] * 10,
        "ev_market": [None] * 10,
        "win_odds": [3.0] * 10,
    })
    out = gui.add_marks(t).set_index(["race_id", "number"])
    # ◆は◎のうち勝率が基準(0.40)以上の馬だけ。A の◎(45%)に付き、B の◎(39%)には付かない
    assert out.loc[("A", 1), "conf_mark"] == "◆"
    assert (out.loc["A", "conf_mark"].iloc[1:] == "").all()  # ◎以外には付けない
    assert (out.loc["B", "conf_mark"] == "").all()
    # margin(◎と2位の勝率の差)と頭数は、レースの各行に記録する
    assert abs(out.loc[("A", 3), "margin"] - 0.20) < 1e-9
    assert abs(out.loc[("B", 1), "margin"] - 0.09) < 1e-9
    assert (out["n_runners"] == 5).all()
    assert (out["conf_version"] == gui.CONF_VERSION).all() and (out["star_version"] == gui.STAR_VERSION).all()


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
    # 取得済みが TRAIN_FROM(2019-01-01)より前からでも、学習に使うのは TRAIN_FROM 以後なので、その1年後から
    assert gui.train_start(dt.date(2014, 1, 5), dt.date(2026, 11, 1)) == dt.date(2019, 1, 1)
    assert gui.eval_start_bounds(dt.date(2014, 1, 5), dt.date(2026, 11, 1)) == (dt.date(2020, 1, 1), dt.date(2025, 11, 1))


def test_course_text_names_turn_inner_outer_and_rail():
    import pandas as pd

    s = lambda **kw: pd.Series({"venue": "京都", "course_type": "芝", "distance": 1600, **kw})
    assert gui.course_text(s(direction="右", course="外", rail="A")) == "芝・右回り・外回り・Aコース 1600m"
    assert gui.course_text(s(direction="右", course=None, rail=None)) == "芝・右回り 1600m"
    assert gui.course_text(s(course_type="障", distance=3170, direction=None, course="外-内")) == "障・外回り→内回り 3170m"
    assert gui.course_text(s(direction="右", course="内2周", rail="C")) == "芝・右回り・内回り2周・Cコース 1600m"
    out = gui.race_head_html(s(round=11, direction="右", course="外", rail="B", ground="良"))
    assert "京都11R" in out and "芝・右回り・外回り・Bコース 1600m ・ 良" in out

def test_predict_table_drops_attrs(monkeypatch):
    import pandas as pd

    from keibaai import gui, predict

    table = pd.DataFrame({"venue": ["東京"], "round": [1.0]})
    table.attrs["flow"] = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(predict, "predict_day", lambda conn, date, rounds=300: table)
    out = gui.predict_table(None, dt.date(2026, 10, 24))
    assert out.attrs == {}
    out.astype({"round": int})  # attrs に表が残っていると ValueError
    pd.concat([out, out.astype({"round": int})], axis=1)


def test_saved_prediction_falls_back_to_the_saved_table():
    import datetime as dt

    from keibaai import gui
    from test_model import synthetic_db

    conn = synthetic_db(n_days=3)
    assert gui.saved_prediction(conn, dt.date(2030, 1, 5)) is None  # 予想がない日
    conn.execute("INSERT INTO predictions (race_id, number, saved_at, date, venue, round, horse_name, win_odds, "
                 "mark, star, p_win_ability, p_win_market, ev_ability, ev_market) VALUES "
                 "('203001050101', 1, '2030-01-04T21:00:00', '2030-01-05', '東京', 1, '馬A', 3.0, '◎', NULL, 0.3, 0.3, 0.9, 0.9)")
    t = gui.saved_prediction(conn, dt.date(2030, 1, 5))  # 計算の保存はない(メモリの DB)ので、保存した予想を出す
    assert list(t["mark"]) == ["◎"] and list(t["star"]) == [""]
    one = gui.race_entries(conn, "203001050101", t)  # 画面の表に足せる形
    assert "mark" in one
