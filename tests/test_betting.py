import itertools

import numpy as np
import pytest

from keibaai import betting as bt

P = np.array([0.4, 0.3, 0.2, 0.1])
NUMS = [1, 2, 3, 4]


def prob_of(kind, key, p=P, nums=NUMS):
    order, prob = bt.harville_outcomes(p)
    idx = {n: i for i, n in enumerate(nums)}
    return prob[bt.hits(order, kind, tuple(idx[k] for k in key), len(nums))].sum()


def test_harville_sums_to_one_and_matches_hand_calc():
    order, prob = bt.harville_outcomes(P)
    assert len(prob) == 24 and np.isclose(prob.sum(), 1)
    assert np.isclose(prob_of("三連単", (1, 2, 3)), 0.4 * 0.3 / 0.6 * 0.2 / 0.3)
    assert np.isclose(prob_of("単勝", (2,)), 0.3)
    assert np.isclose(prob_of("馬単", (1, 2)), 0.4 * 0.3 / 0.6)


def test_unordered_bets_are_sums_of_ordered_ones():
    assert np.isclose(prob_of("馬連", (1, 2)), prob_of("馬単", (1, 2)) + prob_of("馬単", (2, 1)))
    perms = sum(prob_of("三連単", t) for t in itertools.permutations((1, 2, 3)))
    assert np.isclose(prob_of("三連複", (1, 2, 3)), perms)
    # ワイド = 2頭がともに3着以内 = 三連複(その2頭 + 残り1頭)の合計
    assert np.isclose(prob_of("ワイド", (1, 2)), prob_of("三連複", (1, 2, 3)) + prob_of("三連複", (1, 2, 4)))


def test_place_pays_to_third_with_eight_or_more_horses_else_second():
    assert [bt.place_slots(n) for n in (4, 5, 7, 8, 18)] == [0, 2, 2, 3, 3]
    p8 = np.full(8, 1 / 8)
    nums = list(range(1, 9))
    assert np.isclose(prob_of("複勝", (1,), p8, nums), 3 / 8)
    p6 = np.full(6, 1 / 6)
    assert np.isclose(prob_of("複勝", (1,), p6, list(range(1, 7))), 2 / 6)


def test_candidates_keep_only_positive_ev():
    odds = {"単勝": {(1,): 3.0, (2,): 3.0, (3,): 4.0, (4,): 8.0}}  # EV: 1.2, 0.9, 0.8, 0.8
    cands, *_ = bt.candidates(NUMS, P, odds)
    assert [(c.kind, c.numbers) for c in cands] == [("単勝", (1,))] and np.isclose(cands[0].ev, 1.2)


def test_single_bet_matches_closed_form_kelly():
    odds = {"単勝": {(1,): 3.0}}
    cands, order, prob, hit = bt.candidates(NUMS, P, odds)
    f = bt.kelly_fractions(prob, hit, np.array([3.0]))
    assert np.isclose(f[0], (0.4 * 3 - 1) / (3 - 1), atol=1e-4)  # (p·o − 1) / (o − 1)


def test_plan_skips_when_no_edge():
    odds = {"単勝": {(1,): 2.0, (2,): 3.0}, "馬連": {(1, 2): 2.0}}
    plan = bt.make_plan(NUMS, P, odds)
    assert plan.bets == [] and "見送り" in plan.note and plan.top_ev


def test_plan_stakes_are_units_within_cap_and_summary_consistent():
    odds = {"単勝": {(1,): 3.0, (2,): 5.0}, "馬連": {(1, 2): 6.0}}  # 単勝1:1.2, 単勝2:1.5, 馬連: 0.7×6=4.2 → 全部有利
    plan = bt.make_plan(NUMS, P, odds, bankroll=100_000, cap=10_000)
    assert plan.bets and all(b.stake % 100 == 0 and b.stake >= 100 for b in plan.bets)
    assert plan.total == sum(b.stake for b in plan.bets) <= 10_000
    assert plan.worst == -plan.total
    assert np.isclose(plan.expected_return, sum(b.stake * b.ev for b in plan.bets) / plan.total)
    assert np.isclose(plan.expected_profit, sum(b.stake * (b.ev - 1) for b in plan.bets))
    assert plan.expected_return > 1 and 0 < plan.hit_prob <= 1
    best = max(plan.bets, key=lambda b: b.ev)
    assert best.kind == "馬連"  # 期待値がいちばん高い買い目に多く配る
    assert plan.bets[0].stake >= plan.bets[-1].stake


def test_cap_shrinks_stakes_in_proportion():
    odds = {"単勝": {(1,): 3.0}, "馬連": {(1, 2): 6.0}}
    big = bt.make_plan(NUMS, P, odds, bankroll=1_000_000, cap=1_000_000)
    small = bt.make_plan(NUMS, P, odds, bankroll=1_000_000, cap=5_000)
    assert big.total > 5_000 and small.total <= 5_000


def test_unknown_numbers_and_missing_odds_are_ignored():
    odds = {"単勝": {(9,): 3.0, (1,): None}, "複勝": {(1,): 1.5}}  # 4頭立てには複勝がない
    cands, *_ = bt.candidates(NUMS, P, odds, ev_min=0.0)
    assert cands == []


def test_henery_moves_place_probability_from_favourites_to_longshots():
    order, prob = bt.harville_outcomes(P, bt.GAMMA_2ND, bt.GAMMA_3RD)
    _, plain = bt.harville_outcomes(P)
    assert np.isclose(prob.sum(), 1) and (prob > 0).all()
    favs = bt.hits(order, "三連複", (0, 1, 2), 4)
    longshot = bt.hits(order, "三連複", (0, 1, 3), 4)
    assert prob[favs].sum() < plain[favs].sum() and prob[longshot].sum() > plain[longshot].sum()
    win = bt.hits(order, "単勝", (0,), 4)
    assert np.isclose(prob[win].sum(), 0.4)  # 1着の確率は勝率のまま


def test_gamma_one_is_plain_harville():
    a = bt.harville_outcomes(P)[1]
    b = bt.harville_outcomes(P, 1.0, 1.0)[1]
    assert np.allclose(a, b)
