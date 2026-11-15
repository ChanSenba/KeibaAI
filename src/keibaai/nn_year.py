"""★・期待値のモデル(オッズ込み)に入れる、能力のみのニューラルネットの「学習に使っていない」予想。

各行に、その行の年の1月1日より前の結果だけで学んだネット(年ごとのネット)の、レース内の対数勝率(nn_lp)と、
市場の勝率との対数の差(nn_vs_mkt)を足す。学習の行も予想の行も同じ作り方なので、オッズ込みのモデルは
「ネットが市場とどれだけ違う見方をしたときに、どちらが当たるか」を学べる。

2024年までで学習し 2025-01〜2026-11 の 6,477 レースで、勝ち馬の対数尤度が今の形より +2.9/1000(95%区間 +0.9〜+4.9)、
確認用の 2023年(2022年までで学習、3,453 レース)で +2.8(+0.4〜+5.4)(docs/spec.md 8-24)。

年ごとのネットは、その年より前の結果だけで決まるので保存して使い回す(初回だけ1年あたり数分かかる)。
保存先は CACHE_DIR(use_db でつないだ DB の隣の model_cache。None なら保存しない)。
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .model import TRAIN_FROM

FEATURES = ["nn_lp", "nn_vs_mkt"]
SEEDS = 2  # 年ごとのネットの乱数の数(検証も2通りの平均)
CACHE_DIR: Path | None = None


def use_db(conn) -> None:
    """年ごとのネットを、conn の DB と同じフォルダの model_cache に保存する(メモリ上の DB なら保存しない)。"""
    global CACHE_DIR
    if not hasattr(conn, "execute"):
        return
    for _, name, file in conn.execute("PRAGMA database_list"):
        if name == "main":
            CACHE_DIR = Path(file).parent / "model_cache" if file else None


def use_db_path(db_path) -> None:
    """年ごとのネットを、db_path の DB と同じフォルダの model_cache に保存する。"""
    global CACHE_DIR
    CACHE_DIR = Path(db_path).parent / "model_cache"


def _source_hash() -> str:
    import torch

    h = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):  # 要素の定義(factors.py)なども網の形を左右する
        h.update(file.name.encode())
        h.update(file.read_bytes())
    h.update(torch.__version__.encode())
    return h.hexdigest()


def _train_key(train: pd.DataFrame, year: int) -> str:
    """年ごとのネットの指紋: 学習に使う行の中身(特徴量・結果)と計算方法。"""
    from .nn_model import NN_FEATURES, PAST_RUN_COLUMNS

    cols = sorted(set(c for c in NN_FEATURES + PAST_RUN_COLUMNS + ["finish"] if c in train))
    rows = train.sort_values(["race_id", "horse_id"])[cols]
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(rows, index=False).to_numpy().tobytes())
    h.update(f"{year}|{SEEDS}|{TRAIN_FROM}|{_source_hash()}".encode())
    return h.hexdigest()[:20]


def year_models(train: pd.DataFrame, year: int) -> list[dict]:
    """year の1月1日より前の結果(train)で学んだネット。保存があれば使う。"""
    from .nn_model import fit_models

    file = None
    if CACHE_DIR is not None:
        file = Path(CACHE_DIR) / f"nnyear-{year}-{_train_key(train, year)}.pkl"
        if file.exists():
            try:
                with file.open("rb") as f:
                    return pickle.load(f)
            except Exception:  # 壊れた保存は捨てて学び直す
                file.unlink(missing_ok=True)
    models = fit_models(train, n_seeds=SEEDS)
    if file is not None:
        file.parent.mkdir(parents=True, exist_ok=True)
        for old in file.parent.glob(f"nnyear-{year}-*.pkl"):
            old.unlink(missing_ok=True)
        tmp = file.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(models, f)
        tmp.replace(file)
    return models


def add_nn_year(df: pd.DataFrame, only: pd.Index | None = None) -> pd.DataFrame:
    """df の各行に nn_lp・nn_vs_mkt を足す(結果のない出馬表の行にも)。

    TRAIN_FROM の年の行は、それより前で学べないので空。結果のある行は finish が数字の行。
    only を渡すと、その行のある年だけ計算する(ほかの年の行は空。予想の行だけ要るときに速い)。
    """
    from .features import result_mask
    from .nn_model import predict_models

    if all(c in df for c in FEATURES):
        return df
    lp = pd.Series(np.nan, index=df.index)
    first = pd.Timestamp(TRAIN_FROM).year + 1
    done = result_mask(df)
    years = df["date"].dt.year
    wanted = set(years.unique()) if only is None else set(years[only].unique())
    for year in sorted(y for y in wanted if y >= first):
        cut = pd.Timestamp(f"{year}-01-01")
        train = df[done & (df["date"] < cut)]
        if not (train["date"] >= pd.Timestamp(TRAIN_FROM)).any():
            continue
        rows = df[years == year]
        p, _ = predict_models(year_models(train, int(year)), train, rows)
        lp[rows.index] = np.log(p.clip(lower=1e-12))
    odds = pd.to_numeric(df["win_odds"], errors="coerce")
    inv = (1.0 / odds).where(odds > 0)
    mkt = inv / inv.groupby(df["race_id"]).transform("sum")
    return pd.concat([df, pd.DataFrame({"nn_lp": lp, "nn_vs_mkt": lp - np.log(mkt)})], axis=1)
