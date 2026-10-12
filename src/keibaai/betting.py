"""1レースの買い方を決める: 各馬の勝率と当日オッズから、期待値の高い券種・買い目と、その金額配分を出す。

1. 勝率から、1〜3着の並び(3頭の順列)の確率を、ハーヴィルの式(人気馬の2〜3着を少し抑えたヘネリー版)で作る。
2. 並びごとに、的中する買い目(単勝・複勝・馬連・ワイド・馬単・三連複・三連単)が決まる。
3. 買い目の期待値 = 的中確率 × 当日オッズ。1を超えるものだけを候補にする。
4. 候補の中から、資金が長い目で最も増える配分(ケリー基準)を求め、半分に抑えて100円単位にする。

複勝とワイドのオッズは幅(最小〜最大)で出るので、安全側に最小のほうを使う。
期待値が1を超える買い目がなければ、買わない(見送り)が答え。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

KINDS = ["単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"]
UNIT = 100  # 馬券の最小単位(円)


def place_slots(n: int) -> int:
    """複勝が何着まで払うか: 8頭以上は3着まで、5〜7頭は2着まで、4頭以下は発売なし。"""
    return 3 if n >= 8 else 2 if n >= 5 else 0


# 2着・3着を決めるときの勝率のべき乗(ヘネリー)。1ならハーヴィルそのまま。
# ハーヴィルは、人気馬が2〜3着に来る確率を多く、人気薄を少なく見積もる傾向があると言われる。
# 1より小さくすると、人気馬の2〜3着が減り、人気薄の2〜3着が増える。
# 値は文献のもので、JRA の払戻で確かめたものではない。
GAMMA_2ND, GAMMA_3RD = 0.81, 0.65


def harville_outcomes(p: np.ndarray, g2: float = 1.0, g3: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """勝率 p(合計1)から、1〜3着の並び(a, b, c)ごとの確率を返す。

    1着は勝率どおり。2着は残りの馬から p**g2 に比例、3着は残りから p**g3 に比例して決まる(g=1 でハーヴィル)。
    (並びの配列 shape=(m,3), 確率 shape=(m,))。
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n < 3:
        raise ValueError("3頭以上のレースだけ扱えます")
    a, b, c = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    ok = (a != b) & (b != c) & (a != c)
    a, b, c = a[ok], b[ok], c[ok]
    q2, q3 = p ** g2, p ** g3
    prob = p[a] * q2[b] / (q2.sum() - q2[a]) * q3[c] / (q3.sum() - q3[a] - q3[b])
    return np.stack([a, b, c], axis=1), prob


def hits(order: np.ndarray, kind: str, combo: tuple[int, ...], n: int) -> np.ndarray:
    """並び(m,3)のうち、この買い目が的中するもの(真偽の配列)。combo は 0 始まりの馬の番号。"""
    a, b, c = order[:, 0], order[:, 1], order[:, 2]
    if kind == "単勝":
        return a == combo[0]
    if kind == "複勝":
        k = place_slots(n)
        return (a == combo[0]) | (b == combo[0]) | ((c == combo[0]) if k == 3 else False)
    x, y = combo[0], combo[1] if len(combo) > 1 else -1
    if kind == "馬連":
        return ((a == x) & (b == y)) | ((a == y) & (b == x))
    if kind == "馬単":
        return (a == x) & (b == y)
    if kind == "ワイド":
        top3 = [a, b, c]
        return np.any([t == x for t in top3], axis=0) & np.any([t == y for t in top3], axis=0)
    if kind == "三連複":
        z = combo[2]
        return np.any([a == v for v in combo], axis=0) & np.any([b == v for v in combo], axis=0) & \
            np.any([c == v for v in combo], axis=0) & (len({x, y, z}) == 3)
    if kind == "三連単":
        return (a == combo[0]) & (b == combo[1]) & (c == combo[2])
    raise ValueError(kind)


@dataclass
class Bet:
    kind: str
    numbers: tuple[int, ...]  # 馬番(1始まり)。馬連・ワイド・三連複は小さい順、馬単・三連単は着順
    odds: float  # 当日オッズ(倍)
    prob: float  # 的中確率
    ev: float  # 期待値 = prob × odds
    stake: int = 0

    @property
    def label(self) -> str:
        sep = "→" if self.kind in ("馬単", "三連単") else "-"
        return sep.join(str(x) for x in self.numbers)


def _combos(kind: str, n: int):
    if kind in ("単勝", "複勝"):
        return [(i,) for i in range(n)]
    if kind in ("馬連", "ワイド"):
        return list(itertools.combinations(range(n), 2))
    if kind == "馬単":
        return list(itertools.permutations(range(n), 2))
    if kind == "三連複":
        return list(itertools.combinations(range(n), 3))
    return list(itertools.permutations(range(n), 3))


# 的中確率がこれ未満の買い目は候補にしない。確率が小さいほど推定の誤差が大きく、大穴の期待値は過大に出やすい
MIN_HIT = 0.03


def candidates(numbers: list[int], p: np.ndarray, odds: dict[str, dict[tuple[int, ...], float]],
               ev_min: float = 1.0, g2: float = 1.0, g3: float = 1.0,
               min_hit: float = 0.0) -> tuple[list[Bet], np.ndarray, np.ndarray, np.ndarray]:
    """オッズのある全買い目の期待値を出し、ev_min 以上で的中確率が min_hit 以上のものを候補にして返す。

    戻り値: (候補, 並び(m,3), 並びの確率(m,), 候補×並びの的中表(候補数, m))。
    numbers は馬番、p はその勝率(合計1)。odds[券種][馬番の組] = 倍率。
    """
    n = len(numbers)
    order, prob = harville_outcomes(p, g2, g3)
    idx = {num: i for i, num in enumerate(numbers)}
    out, rows = [], []
    for kind in KINDS:
        for key, o in odds.get(kind, {}).items():
            if not o or o <= 0 or any(k not in idx for k in key):
                continue
            combo = tuple(idx[k] for k in key)
            h = hits(order, kind, combo, n)
            pr = float(prob[h].sum())
            if kind == "複勝" and place_slots(n) == 0:
                continue
            if pr * o >= ev_min and pr >= min_hit:
                out.append(Bet(kind, tuple(key), float(o), pr, pr * o))
                rows.append(h)
    hit = np.array(rows, dtype=bool) if rows else np.zeros((0, len(prob)), dtype=bool)
    return out, order, prob, hit


def kelly_fractions(prob: np.ndarray, hit: np.ndarray, odds: np.ndarray, iters: int = 2000) -> np.ndarray:
    """資金に対する各買い目の割合(現金を残す分を除く)。長い目での資金の増え方(対数)が最大になる配分。

    Cover の反復: 資産(現金と各買い目)の割合 f を、f_j ← f_j × E[x_j / W] で更新する。
    現金は常に1、買い目は的中したら odds 倍。購入は現金の範囲内(借りない)。
    """
    k = len(odds)
    if k == 0:
        return np.zeros(0)
    x = np.concatenate([np.ones((1, len(prob))), hit * odds[:, None]], axis=0).T  # (outcomes, 1+k)
    f = np.full(k + 1, 1.0 / (k + 1))
    for _ in range(iters):
        w = x @ f
        new = f * (prob @ (x / w[:, None]))
        if np.abs(new - f).max() < 1e-10:
            f = new
            break
        f = new
    return f[1:]


@dataclass
class Plan:
    bets: list[Bet]
    total: int = 0
    hit_prob: float = 0.0  # どれか1つでも当たる確率
    expected_return: float = 0.0  # 期待回収率(倍)
    expected_profit: float = 0.0
    worst: int = 0  # 全部外れたときの収支
    note: str = ""
    top_ev: list[Bet] = field(default_factory=list)  # 見送りのとき、期待値の高かった買い目(参考)


def make_plan(numbers: list[int], p: np.ndarray, odds: dict[str, dict[tuple[int, ...], float]],
              bankroll: int = 100_000, cap: int = 10_000, kelly_scale: float = 0.5, ev_min: float = 1.0,
              max_candidates: int = 300, g2: float = GAMMA_2ND, g3: float = GAMMA_3RD,
              min_hit: float = MIN_HIT, equal_stake: int = 0) -> Plan:
    """1レースの買い方。bankroll は使える資金、cap はこのレースに使う上限、kelly_scale は配分を抑える割合。

    equal_stake が正のときは配分をせず、候補を期待値の高い順に、1点 equal_stake 円ずつ上限まで買う。
    """
    cap = min(cap, bankroll)
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    bets, order, prob, hit = candidates(numbers, p, odds, ev_min, g2, g3, min_hit)
    if not bets:
        all_bets, *_ = candidates(numbers, p, odds, 0.0, g2, g3)
        top = sorted(all_bets, key=lambda b: -b.ev)[:5]
        return Plan([], note=f"見送り: 期待値が1を超え、的中確率が{min_hit:.0%}以上の買い目がありません", top_ev=top)
    keep = np.argsort([-b.ev for b in bets])[:max_candidates]
    bets, hit = [bets[i] for i in keep], hit[keep]
    if equal_stake > 0:
        unit = max(UNIT, equal_stake // UNIT * UNIT)
        stakes = np.zeros(len(bets), dtype=int)
        stakes[:min(len(bets), cap // unit)] = unit  # bets は期待値の高い順
    else:
        f = kelly_fractions(prob, hit, np.array([b.odds for b in bets])) * kelly_scale
        stakes = np.floor(f * bankroll / UNIT).astype(int) * UNIT
    if stakes.sum() > cap:  # 上限を超えるときは、比率を保って縮める
        stakes = (np.floor(stakes * (cap / stakes.sum()) / UNIT) * UNIT).astype(int)
    chosen = [i for i, s in enumerate(stakes) if s >= UNIT]
    if not chosen:
        return Plan([], note="見送り: 期待値は1を超えますが、優位が小さいか資金が少なく、100円に満たない配分でした",
                    top_ev=sorted(bets, key=lambda b: -b.ev)[:5])
    for i in chosen:
        bets[i].stake = int(stakes[i])
    final = sorted((bets[i] for i in chosen), key=lambda b: -b.stake)
    total = sum(b.stake for b in final)
    hit_any = prob[np.any(hit[chosen], axis=0)].sum()
    exp_ret = sum(b.stake * b.ev for b in final) / total
    return Plan(final, total=total, hit_prob=float(hit_any), expected_return=float(exp_ret),
                expected_profit=float(sum(b.stake * (b.ev - 1) for b in final)), worst=-total)
