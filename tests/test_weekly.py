import datetime as dt
from pathlib import Path

import pandas as pd

from keibaai import db, flow_review, weekly
from keibaai.parse import ParsedRace, Race, Result

from test_model import synthetic_db


def pred_table(race_id="202505010101", date="2025-03-01"):
    rows = []
    for i in range(1, 6):
        rows.append({"race_id": race_id, "date": pd.Timestamp(date), "venue": "東京", "round": 1, "race_name": "テスト特別",
                     "number": i, "horse_name": f"馬{i}", "jockey": "騎手", "win_odds": 2.0 * i, "popularity": i,
                     "p_win_ability": [0.4, 0.25, 0.2, 0.1, 0.05][i - 1], "p_win_market": None,
                     "ev_ability": [0.8, 1.25, 1.2, 0.8, 0.5][i - 1], "ev_market": None})
    return pd.DataFrame(rows)


def add_result(conn, race_id, date, order):
    race = Race(race_id=race_id, date=date, venue="東京", round=1, course_type="芝", distance=1600, ground="良")
    results = [Result(race_id=race_id, number=n, finish=f, win_odds=2.0 * n, popularity=n, horse_id=f"h{n}")
               for f, n in enumerate(order, start=1)]
    db.save_race(conn, ParsedRace(race=race, results=results))


def test_predict_and_save_makes_the_day_reviewable(monkeypatch):
    from keibaai import gui

    conn = db.connect(":memory:")
    monkeypatch.setattr(gui, "predict_with_flow", lambda conn, date: (pred_table(date="2030-01-05"), None))
    assert gui.review_dates(conn) == []
    table = gui.predict_and_save(conn, dt.date(2030, 1, 5))
    assert "mark" in table and len(table) == 5
    assert gui.review_dates(conn) == ["2030-01-05"]
    gui.predict_and_save(conn, dt.date(2030, 1, 5))  # 取り直しても、同じ馬が重ならない
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 5


def test_predict_and_save_keeps_flow_forecast(monkeypatch):
    from keibaai import gui, predict

    conn = db.connect(":memory:")
    table = pred_table(date="2030-01-05")
    table.attrs["flow"] = pd.DataFrame([{"race_id": "202505010101", "n_front": 3, "front_share": 0.2, "course_edge": 0.1,
                                         "course_pace_diff": -0.5, "pace": "H", "advantage": "差", "basis": "テスト"}])
    monkeypatch.setattr(predict, "predict_day", lambda conn, date, rounds=300: table)
    out = gui.predict_and_save(conn, dt.date(2030, 1, 5))
    assert out.attrs == {}  # 画面に渡す表は attrs を持たない
    assert conn.execute("SELECT race_id, pace FROM flow_forecast").fetchall() == [("202505010101", "H")]


def test_save_is_idempotent_and_frozen_after_race_day_morning():
    conn = db.connect(":memory:")
    assert weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 27, 20)) == 5
    t = pred_table(); t["win_odds"] = 99.0
    weekly.save_predictions(conn, t, dt.datetime(2025, 3, 1, 8))  # 当日朝: 更新できる
    assert conn.execute("SELECT win_odds FROM predictions LIMIT 1").fetchone()[0] == 99.0
    t["win_odds"] = 1.0
    assert weekly.save_predictions(conn, t, dt.datetime(2025, 3, 1, 13)) == 0  # 当日昼: 凍結
    assert weekly.save_predictions(conn, t, dt.datetime(2025, 3, 2, 8)) == 0  # 翌日以降も凍結
    assert conn.execute("SELECT win_odds FROM predictions LIMIT 1").fetchone()[0] == 99.0


def test_no_save_when_result_exists():
    conn = db.connect(":memory:")
    add_result(conn, "202505010101", "2025-03-01", [1, 2, 3, 4, 5])
    assert weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 3, 1, 7)) == 0


def test_review_matches_predictions_with_results():
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    assert weekly.review_day(conn, "2025-03-01").summary["races"] == 0  # 結果が出る前
    add_result(conn, "202505010101", "2025-03-01", [3, 1, 2, 5, 4])  # 3番が勝ち、◎(1番)は2着
    r = weekly.review_day(conn, "2025-03-01")
    s = r.summary
    assert s["races"] == 1 and s["honmei_wins"] == 0 and s["top3_hits"] == 3  # ◎○▲(1,2,3番)が全員3着以内
    assert s["ev_bets"] == 2 and s["ev_wins"] == 1 and s["ev_roi"] == 100 * 6.0 / 2  # 3番(6倍)が勝ち
    x = r.races.iloc[0]
    assert x["winner"] == "馬3" and x["winner_rank"] == 3 and x["honmei"] == "馬1" and x["honmei_finish"] == 2
    md = weekly.render_markdown(r)
    assert "2025-03-01 回顧" in md and "勝ち馬は馬3(予想3位" in md


def test_unreviewed_dates_and_upcoming_weekend(tmp_path):
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    assert weekly.unreviewed_dates(conn, dt.date(2025, 3, 3), tmp_path) == ["2025-03-01"]
    (tmp_path / "2025-03-01.md").write_text("x")
    assert weekly.unreviewed_dates(conn, dt.date(2025, 3, 3), tmp_path) == []
    assert weekly.unreviewed_dates(conn, dt.date(2025, 3, 1), tmp_path) == []  # 当日はまだ
    up = weekly._upcoming_weekend
    assert up(dt.date(2026, 10, 14)) == [dt.date(2026, 10, 17), dt.date(2026, 10, 18)]
    assert up(dt.date(2026, 10, 17)) == [dt.date(2026, 10, 17), dt.date(2026, 10, 18)]
    assert up(dt.date(2026, 10, 18)) == [dt.date(2026, 10, 18)]


class FakeFetcher:
    def __init__(self, conn=None):
        self.conn = conn

    def race_ids_on(self, date):
        return []


def test_run_weekly_reviews_then_forecasts(tmp_path, monkeypatch):
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    add_result(conn, "202505010101", "2025-03-01", [3, 1, 2, 5, 4])
    monkeypatch.setattr(weekly, "forecast_day", lambda conn, f, d, now, jv=None: 0)
    out = weekly.run_weekly(conn, FakeFetcher(), dt.date(2025, 3, 3), tmp_path / "r", now=dt.datetime(2025, 3, 3, 21))
    assert out["reviewed"] == ["2025-03-01"] and (tmp_path / "r" / "2025-03-01.md").exists()
    assert list(out["forecast"]) == ["2025-03-08", "2025-03-09"]
    out = weekly.run_weekly(conn, FakeFetcher(), dt.date(2025, 3, 4), tmp_path / "r", now=dt.datetime(2025, 3, 4, 21))
    assert out["reviewed"] == []  # 済んだ日は飛ばす


def test_weekly_command_runs_run_weekly(tmp_path, monkeypatch, capsys):
    from keibaai import cli

    seen = {}

    def fake(conn, fetcher, today, reviews_dir, jv=None, after_races=False, evals_dir=None):
        seen.update(today=today, reviews=str(reviews_dir), jv=jv)
        return {"reviewed": ["2025-03-01"], "forecast": {"2025-03-08": 0, "2025-03-09": 12}, "evaluated": None}

    monkeypatch.setattr(weekly, "run_weekly", fake)
    cli.main(["weekly", "--db", str(tmp_path / "k.db"), "--cache", str(tmp_path / "c"),
              "--reviews", str(tmp_path / "r"), "--today", "2025-03-03", "--source", "jv", "--raw", "rawdir", "--python32", "py32"])
    out = capsys.readouterr().out
    assert seen["today"] == dt.date(2025, 3, 3) and "reviewed: 2025-03-01" in out and seen["jv"] == ("rawdir", "py32")
    assert "forecast 2025-03-08: 出馬表がまだありません" in out and "forecast 2025-03-09: 12頭" in out


def test_weekly_with_jv_takes_results_and_entries_from_jravan(tmp_path, monkeypatch):
    from keibaai import jvlink

    calls = []
    monkeypatch.setattr(weekly, "unreviewed_dates", lambda conn, today, d, include_today=False: ["2025-03-01"])
    monkeypatch.setattr(weekly, "review_day", lambda conn, d: weekly.Review(d, None, None, {"races": 0}))
    monkeypatch.setattr(jvlink, "fetch_results", lambda conn, s, e, raw, py, today=None: calls.append(("results", s)))
    monkeypatch.setattr(jvlink, "fetch_entries", lambda conn, d, odds, raw, py, now=None: calls.append(("entries", d)) or 0)
    conn = db.connect(":memory:")
    out = weekly.run_weekly(conn, None, dt.date(2025, 3, 4), tmp_path / "r", now=dt.datetime(2025, 3, 4, 21),
                            jv=("raw", "py32"))
    assert calls == [("results", dt.date(2025, 3, 1)), ("entries", "2025-03-08"), ("entries", "2025-03-09")]
    assert out["forecast"] == {"2025-03-08": 0, "2025-03-09": 0}


def test_after_races_reviews_today_and_writes_evaluation(tmp_path, monkeypatch):
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    add_result(conn, "202505010101", "2025-03-01", [3, 1, 2, 5, 4])
    today = dt.date(2025, 3, 1)
    assert weekly.unreviewed_dates(conn, today, tmp_path, include_today=True) == ["2025-03-01"]
    monkeypatch.setattr(weekly, "forecast_day", lambda *a, **k: 1 / 0)  # 日曜の夜は予想を作らない
    monkeypatch.setattr(weekly, "write_evaluation", lambda conn, t, d: Path(d) / f"{t}.md")
    out = weekly.run_weekly(conn, FakeFetcher(), today, tmp_path / "r", now=dt.datetime(2025, 3, 1, 18, 30),
                            after_races=True, evals_dir=tmp_path / "e")
    assert out["reviewed"] == ["2025-03-01"] and out["forecast"] == {}
    assert out["evaluated"] == str(tmp_path / "e" / "2025-03-01.md")


def test_write_evaluation_has_report_and_live_trend(tmp_path):
    conn = synthetic_db()
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    add_result(conn, "202505010101", "2025-03-01", [3, 1, 2, 5, 4])
    path = weekly.write_evaluation(conn, dt.date(2026, 6, 1), tmp_path / "e")
    text = path.read_text(encoding="utf-8")
    assert "2026-06-01 学習・評価" in text and "勝AUC" in text and "実運用の推移" in text and "2025-03-01" in text


def test_review_lists_every_marked_horse_and_top3_rate_per_mark():
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    add_result(conn, "202505010101", "2025-03-01", [3, 1, 2, 5, 4])  # 着順: 3番1着、1番2着、2番3着、5番4着、4番5着
    r = weekly.review_day(conn, "2025-03-01")
    m = r.marks.set_index("mark")
    assert m.loc["◎", "top3"] == 1 and m.loc["◎", "top3_rate"] == 1.0  # ◎(1番)は2着
    assert list(r.marks["mark"]) == ["◎", "○", "▲", "△"]
    assert all(m.loc[k, "n"] == 1 for k in ("◎", "○", "▲"))
    md = weekly.render_markdown(r)
    assert "## 印ごとの成績" in md and "| ◎ | 1 | 0 | 1 | 100% |" in md
    line = [x for x in md.splitlines() if "印の着順" in x][0]
    assert line.startswith("  - 印の着順: ◎馬1 2着 / ○馬2 3着 / ▲馬3 1着") and "△馬4 5着" in line


def add_flow_result(conn, order, passing, first3f=34.0, last3f=36.0):
    """着順 order(馬番の並び)。passing は馬番ごとの通過順。"""
    race = Race(race_id="202505010101", date="2025-03-01", venue="東京", round=1, course_type="芝", distance=1600,
                ground="良", pace_first3f=first3f, pace_last3f=last3f)
    results = [Result(race_id="202505010101", number=n, finish=f, win_odds=2.0 * n, popularity=n, horse_id=f"h{n}",
                      passing=passing[n]) for f, n in enumerate(order, start=1)]
    db.save_race(conn, ParsedRace(race=race, results=results))


def forecast_table(share, edge, race_id="202505010101"):
    t = pd.DataFrame({"race_id": [race_id] * 5, "front_share": share, "n_front_runners": 1,
                      "course_front_edge_5": edge, "course_pace_diff_5": 0.5})
    return flow_review.forecast_flow(t)


def test_forecast_flow_rules():
    f = forecast_table(0.5, 0.0).iloc[0]
    assert f["pace"] == "ハイ" and f["advantage"] == "後"  # 先行馬が多く、直近の傾きもなければペースから
    f = forecast_table(0.3, 0.3).iloc[0]
    assert f["pace"] == "平均" and f["advantage"] == "前" and "直近" not in f["basis"] and "3着内率" in f["basis"]
    assert forecast_table(0.3, 0.0).iloc[0]["advantage"] is None


def test_review_compares_forecast_with_actual_flow():
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    flow_review.save_forecast(conn, forecast_table(0.1, 0.0), {"202505010101"}, dt.datetime(2025, 2, 28, 20))  # スロー→前有利の予想
    # 前に付けた1番手(1/5)が勝ち、後ろの馬が着外。ペースは後傾(スロー)
    add_flow_result(conn, [1, 2, 3, 4, 5], {1: "1-1", 2: "3-3", 3: "3-3", 4: "4-4", 5: "5-5"}, 36.5, 34.5)
    r = weekly.review_day(conn, "2025-03-01")
    x = r.races.iloc[0]
    assert x["pace_hit"] and x["adv_hit"] and r.summary["flow_adv_hits"] == 1 and r.summary["flow_races"] == 1
    md = weekly.render_markdown(r)
    assert "展開の予想: 前有利(ペーススロー" in md and "実際の展開: 前有利(前に付けた馬の3着内率が他より" in md
    assert "有利不利(的中)" in md and "展開の予想: 前後の有利不利が1/1レース的中、ペースが1/1レース的中" in md


def test_review_without_forecast_still_shows_actual_flow():
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    add_flow_result(conn, [1, 2, 3, 4, 5], {n: f"{n}-{n}" for n in range(1, 6)})
    r = weekly.review_day(conn, "2025-03-01")
    assert r.races.iloc[0]["adv_hit"] is None and r.summary["flow_races"] == 0
    md = weekly.render_markdown(r)
    assert "実際の展開:" in md and "展開の予想" not in md


def test_forecast_day_saves_flow_forecast_with_predictions(monkeypatch):
    from keibaai import predict

    conn = db.connect(":memory:")
    pred = pred_table()
    pred.attrs["flow"] = forecast_table(0.5, 0.0)
    monkeypatch.setattr(predict, "fetch_entries", lambda conn, f, d, now=None: 5)
    monkeypatch.setattr(predict, "predict_day", lambda conn, d: pred)
    now = dt.datetime(2025, 2, 28, 20)
    assert weekly.forecast_day(conn, FakeFetcher(), dt.date(2025, 3, 1), now) == 5
    row = conn.execute("SELECT pace, advantage FROM flow_forecast").fetchone()
    assert row == ("ハイ", "後")


def test_review_gives_fit_for_each_horse_from_flow_gate_and_ground():
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    # 過去: 馬4(h4)は道悪で2走して3着内0回。今回は道悪(稍重)の、前残り・内枠有利のレース
    for i, (rid, order) in enumerate([("202504010101", [1, 2, 3, 5, 4]), ("202504010201", [1, 2, 3, 5, 4])]):
        race = Race(race_id=rid, date=f"2025-02-0{i + 1}", venue="東京", round=1, course_type="芝", distance=1600, ground="重")
        db.save_race(conn, ParsedRace(race=race, results=[
            Result(race_id=rid, number=n, finish=f, horse_id=f"h{n}") for f, n in enumerate(order, start=1)]))
    race = Race(race_id="202505010101", date="2025-03-01", venue="東京", round=1, course_type="芝", distance=1600,
                ground="稍重", pace_first3f=36.5, pace_last3f=34.5)
    passing = {1: "1-1", 2: "3-3", 3: "3-3", 4: "4-4", 5: "5-5"}
    db.save_race(conn, ParsedRace(race=race, results=[
        Result(race_id="202505010101", number=n, finish=f, win_odds=2.0 * n, popularity=n, horse_id=f"h{n}",
               passing=passing[n], last3f=35.0 + n / 10) for f, n in enumerate([1, 2, 3, 4, 5], start=1)]))
    r = weekly.review_day(conn, "2025-03-01")
    rv = r.reviews["202505010101"]
    assert "稍重" in rv["track"] and "内枠有利" in rv["track"]
    by = {t["number"]: t for t in rv["horses"]}
    assert [t["number"] for t in rv["horses"][:4]] == [1, 2, 3, 4]  # 印のある馬が先
    one = {x["name"]: x for x in by[1]["rows"]}
    assert one["展開"]["grade"] == "向いた" and one["枠"]["grade"] == "向いた" and by[1]["fit"] == "向いた条件"
    assert "序盤1番手(前)" in one["展開"]["text"] and "1番(内枠)" in one["枠"]["text"]
    four = {x["name"]: x for x in by[4]["rows"]}
    assert four["馬場"]["grade"] == "向かなかった" and "道悪は過去2走で3着内0回" in four["馬場"]["text"]
    assert one["コース"]["grade"] == "向いた" and "東京の芝は過去2走で3着内2回(100%)" in one["コース"]["text"]
    assert one["距離"]["grade"] == "向いた" and "芝1600m前後は過去2走で3着内2回" in one["距離"]["text"]
    assert four["コース"]["grade"] == "向かなかった" and four["距離"]["grade"] == "向かなかった"
    assert "上がり" in by[4]["summary"] and by[4]["sub"].startswith("4着 ・ 4番人気")
    md = weekly.render_markdown(r)
    assert "  - 馬場・枠: 稍重・内枠有利" in md and "◎馬1(1着 ・ 1番人気 ・ 向いた条件): 展開向いた / 枠向いた / 馬場向いた / コース向いた / 距離向いた" in md
