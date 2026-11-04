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
from .features import CATEGORICAL, NN_FEATURES
from .model import BODY_WEIGHT_FEATURES, BODY_WEIGHT_MASK_RATE, TRAIN_FROM

MAX_HORSES = 18
CATS = [c for c in CATEGORICAL if c in NN_FEATURES]
NUMS = [c for c in NN_FEATURES if c not in CATS]
OTHER = "その他"  # どの要素にも入らない列(性別など)の網。表示には出さない
PLACE_WEIGHTS = (1.0, 0.5, 0.25)  # 1着・2着・3着を当てる項の重み
EPOCHS, PATIENCE, BATCH = 40, 4, 256
N_SEEDS = 3  # 乱数を変えて学ぶ回数(平均を使う)
# 過去走の塔: 直近 PAST_RUNS 走を1走ずつ見て、どの走りを重く見るかも学ぶ。レースレベル入りのモデルに足すと
# 2025-01〜2026-10 の 6,369 レースで勝ち馬の対数尤度 +10.9/1000(95%区間 +7.6〜+14.0。数字は docs/spec.md 8-12)
PAST_RUNS = 10
PAST_TOWER = "馬の質"  # 過去走の塔の点数を足す要素(要素ごとの勝率の表示はこの要素に含める)
GROUND_CODE = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
PAST_RUN_COLUMNS = ["race_id", "horse_id", "date", "number", "finish", "time_sec", "last3f", "distance", "early_pos", "prev_fig",
                    "venue", "course_type", "direction", "course", "race_class", "ground", "weight_carried", "jockey_id"]


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


def past_runs(df: pd.DataFrame, k_runs: int = PAST_RUNS) -> tuple[np.ndarray, np.ndarray]:
    """各行の直近 k_runs 走の、1走ごとの値 (行数, k_runs, 列) と、その走りがある(完走した)かの印 (行数, k_runs)。

    1走ごとの値は、その走りの結果(着順の位置・勝ち馬とのタイム差・上がりの順位と差・序盤の位置・タイム指数・頭数)と、
    今回との違い(クラス・距離の差、芝ダ・競馬場・回り・同じコースか、馬場、間隔、何走前か、斤量の差、同じ騎手か)。
    同じ馬の前の走りだけを使う。df の行の順に返す(並び順は問わない)。
    """
    d = df.sort_values(["date", "race_id", "number"], kind="stable")
    fin = pd.to_numeric(d["finish"], errors="coerce").astype(float)
    done = fin.notna() & (fin > 0)
    size = done.groupby(d["race_id"]).transform("sum").clip(lower=2).astype(float)
    t = pd.to_numeric(d["time_sec"], errors="coerce") if "time_sec" in d else pd.Series(np.nan, index=d.index)
    gap = (t - t.where(done).groupby(d["race_id"]).transform("min")).clip(0, 5)
    l3 = (pd.to_numeric(d["last3f"], errors="coerce") if "last3f" in d else pd.Series(np.nan, index=d.index)).where(done)
    dist = pd.to_numeric(d["distance"], errors="coerce").astype(float)
    fig = (d.groupby("horse_id", sort=False)["prev_fig"].shift(-1) if "prev_fig" in d
           else pd.Series(np.nan, index=d.index))  # その走りの指数 = 次の走りの prev_fig
    run = pd.DataFrame({
        "fin_rel": (fin - 1) / (size - 1), "win": (fin == 1).astype(float), "top3": (fin <= 3).astype(float),
        "gap": gap, "gap_k": gap / dist * 1000, "l3_rank": (l3.groupby(d["race_id"]).rank(method="min") - 1) / (size - 1),
        "l3_diff": (l3 - l3.groupby(d["race_id"]).transform("min")).clip(0, 4),
        "early": d["early_pos"] if "early_pos" in d else np.nan, "fig": fig, "size": size,
    }).where(done).to_numpy(np.float32)

    def col(name, default=""):
        return d[name].astype(str).to_numpy() if name in d else np.full(len(d), default, object)

    from .features import inner_outer
    venue, ct, direction = col("venue"), col("course_type"), col("direction")
    io = np.array([inner_outer(v, c, co) for v, c, co in zip(venue, ct, col("course"))], object)
    cls = pd.to_numeric(d.get("race_class"), errors="coerce").to_numpy(float) if "race_class" in d else np.full(len(d), np.nan)
    ground = d["ground"].astype(str).map(GROUND_CODE).to_numpy(float) if "ground" in d else np.full(len(d), np.nan)
    wc = pd.to_numeric(d["weight_carried"], errors="coerce").to_numpy(float) if "weight_carried" in d else np.full(len(d), np.nan)
    jockey, date, dist = col("jockey_id"), d["date"].to_numpy("datetime64[D]"), dist.to_numpy()
    done = done.to_numpy()

    nth = d.groupby("horse_id", sort=False).cumcount().to_numpy()
    hid = pd.factorize(d["horse_id"])[0]
    by_horse = np.argsort(hid, kind="stable")  # 馬ごと・日付順の並び
    where = np.empty(len(d), int)
    where[by_horse] = np.arange(len(d))
    n_col = run.shape[1] + 14
    X = np.zeros((len(d), k_runs, n_col), np.float32)
    mask = np.zeros((len(d), k_runs), bool)
    for k in range(1, k_runs + 1):
        i = np.flatnonzero(nth >= k)
        j = by_horse[where[i] - k]
        dd = (dist[j] - dist[i]) / 400
        X[i, k - 1] = np.concatenate([np.nan_to_num(run[j]), np.stack([
            np.nan_to_num(cls[j] - cls[i]), np.nan_to_num(dd), np.abs(np.nan_to_num(dd)), ct[j] == ct[i], venue[j] == venue[i],
            direction[j] == direction[i], (io[j] == io[i]) & (venue[j] == venue[i]) & (dist[j] == dist[i]),
            np.nan_to_num(ground[j]), np.nan_to_num(ground[j] - ground[i]),
            np.log1p((date[i] - date[j]).astype(float)), np.full(len(i), k, float), np.nan_to_num(wc[j] - wc[i]),
            jockey[j] == jockey[i], np.isnan(run[j, 8])], 1).astype(np.float32)], 1)
        mask[i, k - 1] = done[j]
    back = d.index.get_indexer(df.index)
    return X[back], mask[back]


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


def _pack(d: pd.DataFrame, num: np.ndarray, cat: np.ndarray, past: tuple[np.ndarray, np.ndarray] | None = None):
    """馬の行を (レース数, MAX_HORSES, 列) の箱に詰める。past は過去走の (行, 走, 列) と印(標準化ずみ)。"""
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
    out = dict(X=torch.tensor(X), C=torch.tensor(C), M=torch.tensor(M), F=torch.tensor(F), race=race, pos=pos, keep=keep)
    if past is not None:
        pv, pm = past
        P = np.zeros((n_race, MAX_HORSES) + pv.shape[1:], np.float32)
        PM = np.zeros((n_race, MAX_HORSES, pv.shape[1]), bool)
        P[r, p], PM[r, p] = pv[keep], pm[keep]
        out["P"], out["PM"] = torch.tensor(P), torch.tensor(PM)
    return out


def _net(groups: dict[str, list[int]], cat_sizes: list[int], n_past: int = 0):
    """要素ごとの塔の網。n_past > 0 なら過去走の塔も持つ。

    過去走の塔は、直近の走りを1走ずつ同じ小さな網に通し、どの走りを重く見るかの重み(今回との近さ・新しさ)も
    学んで平均し、点数にする。点数は PAST_TOWER の要素に足す。
    """
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
            self.past_at = list(groups).index(PAST_TOWER) if n_past and PAST_TOWER in groups else None
            if self.past_at is not None:
                self.enc = nn.Sequential(nn.Linear(n_past + n_cat, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU())
                self.att = nn.Linear(32, 1)
                self.head = nn.Sequential(nn.Linear(33, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))

        def forward(self, X, C, M, P=None, PM=None):
            """要素ごとの点数 (レース, 馬, 要素)。出走していない枠は大きな負の値。"""
            e = torch.cat([m(C[..., i]) for i, m in enumerate(self.emb)], -1)
            s = torch.cat([t(torch.cat([X[..., g], e], -1)) for t, g in zip(self.towers, self.idx)], -1)
            if self.past_at is not None:
                h = self.enc(torch.cat([P, e[:, :, None, :].expand(-1, -1, P.shape[2], -1)], -1))
                w = torch.softmax(self.att(h).squeeze(-1).masked_fill(~PM, -1e9), -1) * PM.any(-1, keepdim=True)
                n = PM.float().mean(-1, keepdim=True)
                add = self.head(torch.cat([(w[..., None] * h).sum(-2), n], -1))
                s = s + torch.nn.functional.pad(add, (self.past_at, s.shape[-1] - self.past_at - 1))
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


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, seed: int = 0,
                n_seeds: int | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """train(結果のある行)で学び、target の各馬の勝率(レース内で合計1)と、要素ごとの勝率の表を返す。

    乱数を変えて n_seeds 回(既定 N_SEEDS)学び、勝率は対数で平均してレース内で合計1にする(要素ごとの勝率も同じ)。
    1回だけだと、乱数しだいで印が入れ替わる(2025〜2026年の約6,400レースで、勝ち馬の対数尤度が乱数で ±0.004 ほど動く)。
    学習は TRAIN_FROM 以後のレース。最後の1割のレースで成績を見て、良くならなくなったら止める。
    馬体重は、学習の行の BODY_WEIGHT_MASK_RATE の割合で空にする(朝の予想では馬体重が出ていないため)。
    """
    assert train.index.is_unique and target.index.is_unique and not target.index.isin(train.index).any(), \
        "train と target の行の番号は重ならないこと(過去走を行の番号で引くため)"
    cols = [c for c in PAST_RUN_COLUMNS if c in train or c in target]
    both = pd.concat([train[[c for c in cols if c in train]], target[[c for c in cols if c in target]]])
    pv, pm = past_runs(both)
    past = {"X": pv, "M": pm, "pos": pd.Series(np.arange(len(both)), index=both.index)}
    runs = [_fit_predict_one(train, target, seed + k, past) for k in range(n_seeds or N_SEEDS)]
    if len(runs) == 1:
        return runs[0]
    race = target["race_id"]

    def geo(frames: list[pd.Series]) -> pd.Series:
        z = np.exp(np.mean([np.log(f.clip(lower=1e-12)) for f in frames], axis=0))
        z = pd.Series(z, index=frames[0].index).where(frames[0].notna())
        return z / z.groupby(race).transform("sum")

    p_win = geo([r[0] for r in runs])
    parts = pd.DataFrame({c: geo([r[1][c] for r in runs]) for c in runs[0][1].columns}, index=runs[0][1].index)
    return p_win, parts


def _fit_predict_one(train: pd.DataFrame, target: pd.DataFrame, seed: int,
                     past: dict | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """fit_predict の1回ぶん。past は past_runs の結果(行の位置つき)。なければ過去走の塔を使わない。"""
    import torch

    torch.manual_seed(seed)
    tr = train[train["date"] >= pd.Timestamp(TRAIN_FROM)].sort_values(["date", "race_id", "number"])
    order = target.index
    target = target.sort_values(["race_id", "number"])
    races = tr["race_id"].unique()
    n_val = len(races) // 10
    fit_r, val_r = (races[:-n_val], races[-n_val:]) if n_val else (races, races[:0])
    fit = tr[tr["race_id"].isin(fit_r)]
    prep = _Prep().fit(fit)
    groups = _groups()
    scale = None
    if past is not None:  # 過去走の値は、学習に使う行のある走りで標準化する
        valid = past["X"][past["pos"][fit.index].to_numpy()][past["M"][past["pos"][fit.index].to_numpy()]]
        scale = (valid.mean(0), np.maximum(valid.std(0), 1e-2)) if len(valid) else None  # ほぼ一定の列を大きくしすぎない

    def pack(d: pd.DataFrame):
        if scale is None:
            return _pack(d, *prep.transform(d))
        at = past["pos"][d.index].to_numpy()
        pm = past["M"][at]
        pv = np.where(pm[..., None], (past["X"][at] - scale[0]) / scale[1], 0.0).astype(np.float32)
        return _pack(d, *prep.transform(d), past=(pv, pm))

    net = _net(groups, [len(prep.levels[c]) for c in CATS], past["X"].shape[-1] if scale is not None else 0)
    D = pack(fit)
    V = pack(tr[tr["race_id"].isin(val_r)]) if n_val else None

    def score(B: dict, b=slice(None), X=None):
        return net(B["X"][b] if X is None else X, B["C"][b], B["M"][b], *((B["P"][b], B["PM"][b]) if "P" in B else ()))
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
            loss = _pl_loss(score(D, b, X).sum(-1), D["F"][b], D["M"][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if V is None:
            continue
        net.eval()
        with torch.no_grad():
            lp = torch.log_softmax(score(V).sum(-1), 1)
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
    T = pack(target)
    net.eval()
    with torch.no_grad():
        s = score(T)
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
