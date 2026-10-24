"""能力のみの勝率を、レース単位のニューラルネットで出す(要素別タワー型)。

1レースを丸ごと1つの入力にして、出走馬の点数をレース内のソフトマックスで確率にし、1着 → 残りの中の2着 → 3着を
順に当てる形(Plackett-Luce)で学ぶ。点数は7つの要素(factors.FACTORS)ごとの小さい網の点数の合計なので、
要素の点数がそのまま総合の内訳になる。

2024・2025・2026年をそれぞれその年より前(2019年〜)で学習して評価し、1レースあたりの勝ち馬の対数尤度は
LightGBM(1頭ずつ勝ち負けを学び、あとでレース内で合計1)の −2.0670 → −2.0257、◎の的中は 29.8% → 30.3%。
レース内で比べるだけ(1着だけを学ぶ)だと −2.0404、1〜3着の順まで学ぶと −2.0288 で、要素別に分けても落ちなかった。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import FACTORS
from .features import ABILITY_MODEL_FEATURES, CATEGORICAL
from .model import BODY_WEIGHT_FEATURES, BODY_WEIGHT_MASK_RATE, TRAIN_FROM

MAX_HORSES = 18
CATS = [c for c in CATEGORICAL if c in ABILITY_MODEL_FEATURES]
NUMS = [c for c in ABILITY_MODEL_FEATURES if c not in CATS]
OTHER = "その他"  # どの要素にも入らない列(性別など)の網。表示には出さない
PLACE_WEIGHTS = (1.0, 0.5, 0.25)  # 1着・2着・3着を当てる項の重み
EPOCHS, PATIENCE, BATCH = 40, 4, 256


class _Prep:
    """数値の列は1〜99%で切って標準化し、レース内の平均との差と、空欄の印を足す。カテゴリは番号にする。"""

    def fit(self, train: pd.DataFrame) -> "_Prep":
        x = train[NUMS].astype(float)
        self.lo, self.hi = x.quantile(0.01), x.quantile(0.99)
        xc = x.clip(self.lo, self.hi, axis=1)
        self.mu, self.sd = xc.mean().fillna(0.0), xc.std().replace(0, 1).fillna(1.0)
        self.levels = {c: {v: i + 1 for i, v in enumerate(train[c].astype(str).unique())} for c in CATS}
        return self

    def transform(self, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        z = (d[NUMS].astype(float).clip(self.lo, self.hi, axis=1) - self.mu) / self.sd
        miss = z.isna().to_numpy(np.float32)
        z = z.fillna(0.0)
        # ソフトマックスはレース内の差しか見ないので、出走馬の平均との差を直接渡す
        rel = (z - z.groupby(d["race_id"].to_numpy()).transform("mean")).to_numpy()
        num = np.concatenate([z.to_numpy(), rel, miss], axis=1).astype(np.float32)
        cat = np.stack([d[c].astype(str).map(self.levels[c]).fillna(0).to_numpy() for c in CATS], axis=1).astype(np.int64)
        return num, cat


def _groups() -> dict[str, list[int]]:
    """要素ごとの入力の列番号(値・レース内の差・空欄の印の3つずつ)。"""
    n = len(NUMS)
    out, used = {}, set()
    for name, feats in {**FACTORS, OTHER: NUMS}.items():
        ii = [NUMS.index(c) for c in feats if c in NUMS and (name != OTHER or NUMS.index(c) not in used)]
        if not ii:
            continue
        used |= set(ii)
        out[name] = ii + [i + n for i in ii] + [i + 2 * n for i in ii]
    return out


def _pack(d: pd.DataFrame, num: np.ndarray, cat: np.ndarray):
    """馬の行を (レース数, MAX_HORSES, 列) の箱に詰める。"""
    import torch

    race = pd.factorize(d["race_id"])[0]
    pos = d.groupby("race_id", sort=False).cumcount().to_numpy()
    keep = pos < MAX_HORSES
    n_race = race.max() + 1 if len(race) else 0
    X = np.zeros((n_race, MAX_HORSES, num.shape[1]), np.float32)
    C = np.zeros((n_race, MAX_HORSES, cat.shape[1]), np.int64)
    M = np.zeros((n_race, MAX_HORSES), bool)
    F = np.full((n_race, MAX_HORSES), 99.0, np.float32)
    r, p = race[keep], pos[keep]
    X[r, p], C[r, p], M[r, p] = num[keep], cat[keep], True
    F[r, p] = pd.to_numeric(d["finish"], errors="coerce").fillna(99).to_numpy()[keep]
    return dict(X=torch.tensor(X), C=torch.tensor(C), M=torch.tensor(M), F=torch.tensor(F), race=race, pos=pos, keep=keep)


def _net(groups: dict[str, list[int]], cat_sizes: list[int]):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(s + 1, 4) for s in cat_sizes])
            self.idx = [torch.tensor(g) for g in groups.values()]
            n_cat = 4 * len(cat_sizes)
            self.towers = nn.ModuleList([
                nn.Sequential(nn.Linear(len(g) + n_cat, 64), nn.GELU(), nn.Dropout(0.1),
                              nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)) for g in groups.values()])

        def forward(self, X, C, M):
            """要素ごとの点数 (レース, 馬, 要素)。出走していない枠は大きな負の値。"""
            e = torch.cat([m(C[..., i]) for i, m in enumerate(self.emb)], -1)
            s = torch.cat([t(torch.cat([X[..., g], e], -1)) for t, g in zip(self.towers, self.idx)], -1)
            return s.masked_fill(~M[..., None], -1e9)

    return Net()


def _pl_loss(z, F, M):
    import torch

    loss, avail = 0.0, M.clone()
    order = torch.argsort(F.masked_fill(~M, 1e9), dim=1)
    for j, w in enumerate(PLACE_WEIGHTS):
        idx = order[:, j]
        ok = torch.gather(F, 1, idx[:, None]).squeeze(1) == j + 1  # その着順の馬がいるレースだけ
        lp = torch.log_softmax(z.masked_fill(~avail, -1e9), 1).gather(1, idx[:, None]).squeeze(1)
        loss = loss - w * (lp * ok).sum() / ok.sum().clamp(min=1)
        avail = avail.clone()
        avail[torch.arange(len(idx)), idx] = False
    return loss


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """train(結果のある行)で学び、target の各馬の勝率(レース内で合計1)と、要素ごとの勝率の表を返す。

    学習は TRAIN_FROM 以後のレース。最後の1割のレースで成績を見て、良くならなくなったら止める。
    馬体重は、学習の行の BODY_WEIGHT_MASK_RATE の割合で空にする(朝の予想では馬体重が出ていないため)。
    """
    import torch

    torch.manual_seed(seed)
    tr = train[train["date"] >= pd.Timestamp(TRAIN_FROM)].sort_values(["date", "race_id", "number"])
    order = target.index
    target = target.sort_values(["race_id", "number"])
    races = tr["race_id"].unique()
    n_val = len(races) // 10
    fit_r, val_r = (races[:-n_val], races[-n_val:]) if n_val else (races, races[:0])
    prep = _Prep().fit(tr[tr["race_id"].isin(fit_r)])
    groups = _groups()
    net = _net(groups, [len(prep.levels[c]) for c in CATS])
    D = _pack(tr[tr["race_id"].isin(fit_r)], *prep.transform(tr[tr["race_id"].isin(fit_r)]))
    V = _pack(tr[tr["race_id"].isin(val_r)], *prep.transform(tr[tr["race_id"].isin(val_r)])) if n_val else None
    n = len(NUMS)
    bw = [NUMS.index(c) for c in BODY_WEIGHT_FEATURES if c in NUMS]
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = -np.inf, None, 0
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(D["X"]))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            X = D["X"][b].clone()
            hide = torch.rand(X.shape[:2]) < BODY_WEIGHT_MASK_RATE  # 馬体重が空の場合も学ばせる
            for j in bw:
                X[..., j] = X[..., j].masked_fill(hide, 0.0)
                X[..., j + n] = X[..., j + n].masked_fill(hide, 0.0)
                X[..., j + 2 * n] = torch.where(hide, torch.ones_like(X[..., j]), X[..., j + 2 * n])
            loss = _pl_loss(net(X, D["C"][b], D["M"][b]).sum(-1), D["F"][b], D["M"][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if V is None:
            continue
        net.eval()
        with torch.no_grad():
            lp = torch.log_softmax(net(V["X"], V["C"], V["M"]).sum(-1), 1)
        won = (V["F"] == 1) & V["M"]
        ll = float((lp * won).sum() / won.any(1).sum().clamp(min=1))
        if ll > best + 1e-5:
            best, bad, best_state = ll, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    T = _pack(target, *prep.transform(target))
    net.eval()
    with torch.no_grad():
        s = net(T["X"], T["C"], T["M"])
        total = torch.softmax(s.sum(-1), 1).numpy()
        each = torch.softmax(s, 1).numpy()  # 要素ごとの点数だけで出した勝率
    k = T["keep"]
    r, p = T["race"][k], T["pos"][k]
    p_win = pd.Series(np.nan, index=target.index)
    p_win[target.index[k]] = total[r, p]
    parts = pd.DataFrame(np.nan, index=target.index, columns=list(FACTORS))
    for j, name in enumerate(groups):
        if name in FACTORS:
            parts.loc[target.index[k], name] = each[r, p, j]
    return p_win.reindex(order), parts.reindex(order)
