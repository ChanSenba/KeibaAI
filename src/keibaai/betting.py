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
# 資金が少なくて100円ずつにするときの歯止め: 1レースに使う資金の割合と、買い目の期待値の下限
SMALL_SHARE, SMALL_EV_MIN = 0.3, 1.05


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
    note = ""
    if not chosen and equal_stake <= 0 and cap >= UNIT:
        # 資金が少なく、ケリーの配分がどれも100円に満たないときは、期待値の高い買い目から100円ずつ、上限まで買う
        # ただし 1レースに使うのは資金の3割まで(最低1点)、期待値が1.05に満たない買い目は入れない
        worth = sum(1 for b in bets if b.ev >= SMALL_EV_MIN)
        n = max(1, min(worth, cap // UNIT, int(bankroll * SMALL_SHARE) // UNIT))
        stakes = np.zeros(len(bets), dtype=int)
        stakes[:n] = UNIT  # bets は期待値の高い順
        chosen = list(range(n))
        note = "資金が少ないため、ケリー基準の配分は100円に満たず、期待値の高い買い目から100円ずつにしています(配分より多めに張っています。資金の3割まで)"
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
                expected_profit=float(sum(b.stake * (b.ev - 1) for b in final)), worst=-total, note=note)


# ---------------------------------------------------------------- 印からの買い方
MARKS = ["◎", "○", "▲", "△", "☆"]


def mark_combos(marks: dict[int, str]) -> dict[str, list[tuple[int, ...]]]:
    """ユーザーの印から、券種ごとの買い目(馬番の組)を組む。◎が軸。

    単勝・複勝は ◎。馬連・ワイドは ◎ から印の馬への流し、馬単は ◎ → 印の馬。
    三連複は ◎ − ○▲ − 印の馬、三連単は ◎ → ○▲ → 印の馬のフォーメーション(○▲がなければ印の馬どうし)。
    """
    honmei = [n for n, m in marks.items() if m == "◎"]
    if len(honmei) != 1:
        raise ValueError("◎ を1頭だけ付けてください")
    a = honmei[0]
    others = sorted((n for n, m in marks.items() if m in MARKS[1:]), key=lambda n: (MARKS.index(marks[n]), n))
    second = [n for n in others if marks[n] in ("○", "▲")] or others
    out: dict[str, list[tuple[int, ...]]] = {"単勝": [(a,)], "複勝": [(a,)]}
    if others:
        out["馬連"] = [tuple(sorted((a, x))) for x in others]
        out["ワイド"] = list(out["馬連"])
        out["馬単"] = [(a, x) for x in others]
    if len(others) >= 2:
        out["三連複"] = sorted({tuple(sorted((a, x, y))) for x in second for y in others if y != x})
        out["三連単"] = [(a, x, y) for x in second for y in others if y != x]
    return out


@dataclass
class KindSummary:
    """印から組んだ、1つの券種の買い目のまとめ(どれが当たっても同じ払戻になるように買ったとき)。"""
    kind: str
    bets: list[Bet]  # オッズのある買い目(トリガミを外すときは外したあと)
    hit: np.ndarray  # 買い目 × 並びの的中表
    hit_prob: float  # どれか1つでも当たる確率
    synthetic: float  # 合成オッズ = 1 / Σ(1/オッズ)。1 未満なら、当たっても払戻が買った金額に届かない(トリガミ)
    expected_return: float  # 期待回収率 = 合成オッズ × 的中確率の合計
    dropped: int = 0  # トリガミを避けるために外した点数


def mark_summaries(numbers: list[int], p: np.ndarray, odds: dict[str, dict[tuple[int, ...], float]],
                   marks: dict[int, str], no_torigami: bool = True, g2: float = GAMMA_2ND,
                   g3: float = GAMMA_3RD) -> tuple[list[KindSummary], np.ndarray]:
    """印の買い目を券種ごとにまとめる。期待回収率の高い順。(まとめ, 並びの確率) を返す。

    numbers は出走する全馬の馬番、p はその勝率。no_torigami なら、合成オッズが1以上になるまでオッズの低い買い目を外す。
    """
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    n = len(numbers)
    order, prob = harville_outcomes(p, g2, g3)
    idx = {num: i for i, num in enumerate(numbers)}
    out = []
    for kind, combos in mark_combos(marks).items():
        if kind == "複勝" and place_slots(n) == 0:
            continue
        table = odds.get(kind, {})
        bets, rows = [], []
        for key in combos:
            o = table.get(key)
            if not o or o <= 0 or any(k not in idx for k in key):
                continue
            h = hits(order, kind, tuple(idx[k] for k in key), n)
            pr = float(prob[h].sum())
            bets.append(Bet(kind, key, float(o), pr, pr * o))
            rows.append(h)
        if not bets:
            continue
        keep = sorted(range(len(bets)), key=lambda i: -bets[i].odds)
        dropped = 0
        while no_torigami and len(keep) > 1 and sum(1 / bets[i].odds for i in keep) > 1:
            keep.pop()  # いちばんオッズの低い買い目を外す
            dropped += 1
        keep.sort(key=lambda i: -bets[i].prob)
        bs, hit = [bets[i] for i in keep], np.array([rows[i] for i in keep], dtype=bool)
        syn = 1 / sum(1 / b.odds for b in bs)
        out.append(KindSummary(kind, bs, hit, float(prob[hit.any(axis=0)].sum()), syn,
                               syn * sum(b.prob for b in bs), dropped))
    out.sort(key=lambda s: -s.expected_return)
    return out, prob


# おすすめの券種は、どれか当たる確率がこれ以上のものから選ぶ(三連単の数点のような、まず当たらない券種を推さない)
RECOMMEND_MIN_HIT = 0.10


def recommend(sums: list[KindSummary]) -> int | None:
    """おすすめの券種の位置。的中確率が RECOMMEND_MIN_HIT 以上の中で期待回収率がいちばん高いもの(なければ全体で)。"""
    if not sums:
        return None
    ok = [i for i, s in enumerate(sums) if s.hit_prob >= RECOMMEND_MIN_HIT] or list(range(len(sums)))
    return max(ok, key=lambda i: sums[i].expected_return)


def equal_payout_stakes(odds: list[float], budget: int) -> list[int]:
    """どれが当たっても払戻がなるべく同じになる、100円単位の金額。まず全部に100円、残りは払戻のいちばん少ない買い目に足す。"""
    stakes = [UNIT] * len(odds)
    total = UNIT * len(odds)
    while total + UNIT <= budget:
        i = min(range(len(odds)), key=lambda j: stakes[j] * odds[j])
        stakes[i] += UNIT
        total += UNIT
    return stakes


def equal_payout_plan(chosen: list[KindSummary], prob: np.ndarray, budget: int,
                      amounts: list[int] | None = None) -> Plan:
    """選んだ券種を、券種ごとに、どれが当たっても同じくらいの払戻になるように買う。

    amounts を渡すと券種ごとにその金額(100円未満は切り捨て)、なければ予算を券種の数で等分する。
    """
    if not chosen:
        return Plan([], note="券種を選んでください")
    if amounts is None:
        amounts = [budget // len(chosen)] * len(chosen)
    bets, rows, notes = [], [], []
    for s, amount in zip(chosen, amounts):
        share = int(amount) // UNIT * UNIT
        k = min(len(s.bets), share // UNIT)
        if k == 0:
            notes.append(f"{s.kind}は予算が足りず買えません")
            continue
        if k < len(s.bets):  # 予算が足りないときは、当たりやすい買い目から(bets は的中確率の高い順)
            notes.append(f"{s.kind}は予算が足りず、当たりやすい{k}点だけにしています")
        stakes = equal_payout_stakes([b.odds for b in s.bets[:k]], share)
        for b, st, h in zip(s.bets[:k], stakes, s.hit[:k]):
            bets.append(Bet(b.kind, b.numbers, b.odds, b.prob, b.ev, st))
            rows.append(h)
    if not bets:
        return Plan([], note="金額が足りません。金額を増やすか、券種を減らしてください")
    total = sum(b.stake for b in bets)
    hit_any = float(prob[np.array(rows).any(axis=0)].sum())
    exp = sum(b.stake * b.ev for b in bets)
    return Plan(bets, total=total, hit_prob=hit_any, expected_return=exp / total, expected_profit=exp - total,
                worst=-total, note="。".join(notes))
