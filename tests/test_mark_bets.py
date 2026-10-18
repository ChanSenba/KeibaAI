import numpy as np
import pandas as pd
import pytest

from keibaai import betting, db, gui, weekly

from test_weekly import add_result, pred_table


def test_mark_combos_build_formations_around_honmei():
    marks = {3: "◎", 5: "○", 1: "▲", 7: "△", 2: "☆"}
    c = betting.mark_combos(marks)
    assert c["単勝"] == [(3,)] and c["複勝"] == [(3,)]
    assert c["馬連"] == [(3, 5), (1, 3), (3, 7), (2, 3)]  # ◎から ○▲△☆ へ流し(組は小さい順)
    assert c["馬単"] == [(3, 5), (3, 1), (3, 7), (3, 2)]
    assert len(c["三連単"]) == 2 * 3  # ◎ → ○▲ → 残り3頭
    assert (3, 5, 1) in c["三連単"] and (3, 1, 5) in c["三連単"] and (3, 7, 2) not in c["三連単"]
    assert c["三連複"] == sorted(set(c["三連複"])) and (1, 3, 5) in c["三連複"] and (2, 3, 7) not in c["三連複"]


def test_mark_combos_need_one_honmei():
    with pytest.raises(ValueError):
        betting.mark_combos({1: "○"})
    with pytest.raises(ValueError):
        betting.mark_combos({1: "◎", 2: "◎"})
    assert set(betting.mark_combos({1: "◎"})) == {"単勝", "複勝"}  # 相手がいなければ単複だけ


def odds_for(n=8):
    return {"単勝": {(i,): 2.0 * i for i in range(1, n + 1)},
            "複勝": {(i,): 1.0 + 0.3 * i for i in range(1, n + 1)},
            "馬連": {(a, b): 3.0 * a * b for a in range(1, n + 1) for b in range(a + 1, n + 1)},
            "ワイド": {(a, b): 1.2 * a * b for a in range(1, n + 1) for b in range(a + 1, n + 1)}}


def test_mark_summaries_and_equal_payout():
    numbers = list(range(1, 9))
    p = 1 / np.array([2.0 * i for i in numbers])
    marks = {1: "◎", 2: "○", 3: "▲"}
    sums, prob = betting.mark_summaries(numbers, p, odds_for(), marks, no_torigami=False)
    by = {s.kind: s for s in sums}
    assert set(by) == {"単勝", "複勝", "馬連", "ワイド"}  # オッズのない券種は出さない
    assert [s.expected_return for s in sums] == sorted((s.expected_return for s in sums), reverse=True)
    w = by["馬連"]
    assert len(w.bets) == 2 and w.synthetic == pytest.approx(1 / sum(1 / b.odds for b in w.bets))
    assert w.expected_return == pytest.approx(w.synthetic * sum(b.prob for b in w.bets))
    assert 0 < w.hit_prob <= sum(b.prob for b in w.bets) + 1e-9

    plan = betting.equal_payout_plan([w], prob, 3000)
    assert plan.total <= 3000 and all(b.stake % 100 == 0 and b.stake >= 100 for b in plan.bets)
    pays = [b.stake * b.odds for b in plan.bets]
    assert max(pays) - min(pays) <= max(b.odds for b in plan.bets) * 100  # 払戻はほぼそろう


def test_no_torigami_drops_low_odds_until_synthetic_reaches_one():
    numbers = list(range(1, 9))
    p = np.full(8, 1 / 8)
    odds = {"ワイド": {(1, 2): 1.5, (1, 3): 2.0, (1, 4): 6.0, (1, 5): 9.0}}
    marks = {1: "◎", 2: "○", 3: "▲", 4: "△", 5: "△"}
    keep, _ = betting.mark_summaries(numbers, p, odds, marks, no_torigami=False)
    trim, _ = betting.mark_summaries(numbers, p, odds, marks, no_torigami=True)
    assert keep[0].synthetic < 1 and len(keep[0].bets) == 4
    assert trim[0].synthetic >= 1 and trim[0].dropped == 4 - len(trim[0].bets)
    assert min(b.odds for b in trim[0].bets) > 1.5  # オッズの低い方から外す


def test_equal_payout_plan_with_small_budget_keeps_likely_bets():
    numbers = list(range(1, 9))
    p = 1 / np.array([2.0 * i for i in numbers])
    sums, prob = betting.mark_summaries(numbers, p, odds_for(), {1: "◎", 2: "○", 3: "▲", 4: "△"}, no_torigami=False)
    w = next(s for s in sums if s.kind == "馬連")
    plan = betting.equal_payout_plan([w], prob, 200)
    assert len(plan.bets) == 2 and plan.total == 200 and "2点だけ" in plan.note


def test_market_probs_and_user_marks_roundtrip():
    t = gui.market_probs({"単勝": {(1,): 2.0, (2,): 4.0, (3,): 4.0}})
    assert t["p_win"].tolist() == pytest.approx([0.5, 0.25, 0.25])
    with pytest.raises(ValueError):
        gui.market_probs({"単勝": {(1,): 2.0}})
    conn = db.connect(":memory:")
    gui.save_user_marks(conn, "r1", {1: "◎", 4: "△", 5: ""})
    assert gui.load_user_marks(conn, "r1") == {1: "◎", 4: "△"}
    gui.save_user_marks(conn, "r1", {})
    assert gui.load_user_marks(conn, "r1") == {}


def test_review_mark_stats_and_marked_finishes():
    import datetime as dt
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    add_result(conn, "202505010101", "2025-03-01", [3, 1, 2, 5, 4])  # 3番が勝ち、◎(1番)は2着
    r = weekly.review_day(conn, "2025-03-01")
    m = r.marks.set_index("印")
    assert m.loc["◎", "頭数"] == 1 and m.loc["◎", "勝率"] == 0 and m.loc["◎", "連対率"] == 1
    assert m.loc["▲", "1着"] == 1 and m.loc["▲", "馬券内率"] == 1 and m.loc["▲", "単勝回収率"] == pytest.approx(6.0)
    got = weekly.marked_finishes(r.horses, "202505010101")
    assert [(x[0], x[3]) for x in got][:3] == [("◎", "2着"), ("○", "3着"), ("▲", "1着")]
    assert "印の馬: ◎馬1 2着" in weekly.render_markdown(r) and "馬券内率" in weekly.render_markdown(r)


def test_recommend_skips_long_shots():
    mk = lambda kind, hit, ret: betting.KindSummary(kind, [], np.zeros((0, 1), bool), hit, 1.0, ret)
    sums = [mk("三連単", 0.01, 0.95), mk("ワイド", 0.3, 0.9), mk("単勝", 0.15, 0.8)]
    assert betting.recommend(sums) == 1
    assert betting.recommend([mk("三連単", 0.01, 0.95), mk("馬単", 0.02, 0.9)]) == 0
    assert betting.recommend([]) is None
