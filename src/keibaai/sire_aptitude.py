"""種牡馬・母父ごとの「得意な条件」を、その日より前の産駒の出走だけから数える(血統の得意不得意の特徴量)。

各出走の超過成績 ex = 3着以内(0/1) − 期待(市場の3着以内の見込み、または 3/頭数)。
得意の度合いは、条件の組(例: 種牡馬×競馬場×芝ダ)での「超過 − その種牡馬の全体の超過」の合計を (n+k) で割ったもの
(回数が少ないほど 0 に引き寄せる)。強いだけの種牡馬(どこでも走る)は、全体を引くので得意にならない。
種牡馬の全体の超過(sire_ex_shr)は、系統 → 0 の順に引き寄せる。

2026-10-16 の診断(奇数年と偶数年で得意の度合いの相関を見た)で、偶然でないと分かった条件だけを使う:
芝ダ×距離帯・競馬場×芝ダ・コース・レース名・芝の道悪、と、ダートのスロー(前半が遅いレース)の得意。
スローかどうかはレースが終わるまで分からないので、そのレースの条件では引かず、種牡馬の『スローのときの得意』を値として持つ。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

DIST_BANDS = [0, 1300, 1600, 1900, 2200, 9999]


def prior_sums(df: pd.DataFrame, keys: list[str], value: str) -> tuple[pd.Series, pd.Series]:
    """各行について、同じ keys の行のうち、日付がその行より前のものの (件数, value の合計)。同じ日の行は入れない。"""
    k = df[keys + ["date"]].copy()
    k["_v"] = df[value]
    day = k.groupby(keys + ["date"], dropna=False, sort=True)["_v"].agg(["count", "sum"]).reset_index()
    day = day.sort_values(keys + ["date"])
    g = day.groupby(keys, dropna=False, sort=False)
    day["n"] = g["count"].cumsum() - day["count"]
    day["s"] = g["sum"].cumsum() - day["sum"]
    m = k.reset_index().merge(day[keys + ["date", "n", "s"]], on=keys + ["date"], how="left").set_index("index")
    m = m.reindex(df.index)
    bad = df[keys].isna().any(axis=1)
    return m["n"].where(~bad, 0.0).fillna(0.0), m["s"].where(~bad, 0.0).fillna(0.0)


def _named_race(name) -> str | None:
    if not isinstance(name, str):
        return None
    s = re.sub(r"\(.*\)$", "", name).strip()
    return None if not s or re.match(r"^\d歳", s) else s


def conditions(df: pd.DataFrame) -> pd.DataFrame:
    """条件の欄を足す(どれも発走前に分かるもの)。"""
    df = df.copy()
    ct = df["course_type"].astype(str)
    df["_ctdist"] = ct + pd.cut(df["distance"], DIST_BANDS, labels=False).astype("Int64").astype(str)
    df["_venuect"] = df["venue"].astype(str) + ct
    df["_course"] = df["_venuect"] + df["distance"].astype(str) + df["course"].fillna("").astype(str)
    # 名前のあるレースだけ(「3歳未勝利」「3歳以上1勝クラス」のような条件だけの名前は除く)。括弧の格付けは外す
    df["_racename"] = df["race_name"].map(_named_race)
    df["_turfwet"] = np.where(ct == "芝", np.where(df["ground"].astype(str) == "良", "芝良", "芝道悪"), None)
    return df


CONDITIONS = {"ctdist": "_ctdist", "venuect": "_venuect", "course": "_course", "racename": "_racename",
              "turfwet": "_turfwet"}


def add_sire_aptitude(df: pd.DataFrame, ex: str = "ex", k: float = 50, parents=("sire_id", "damsire_id"),
                      line_of: dict | None = None) -> pd.DataFrame:
    """出走の表(date, course_type, distance, venue, course, race_name, ground, pace_z, 種牡馬・母父, ex)に、得意の度合いを足す。

    ex は各出走の超過成績(3着以内 − 期待)。率はどれも、その日より前の出走だけ。
    列の名前は {sire|dam}_{条件}_apt と、全体の超過 {sire|dam}_ex_shr、ダートのスローの得意 {sire|dam}_dslow_apt。
    """
    df = conditions(df)
    line_of = line_of or {}
    out = {}
    for parent in parents:
        tag = "sire" if parent == "sire_id" else "dam"
        line = df[parent].map(line_of) if line_of else pd.Series(np.nan, index=df.index)
        df["_line"] = line
        ln, ls = prior_sums(df, ["_line"], ex)
        line_shr = ls / (ln + k)  # 系統の全体を 0 に引き寄せる
        pn, ps = prior_sums(df, [parent], ex)
        par = (ps + k * line_shr) / (pn + k)  # 種牡馬の全体を系統に引き寄せる
        out[f"{tag}_ex_shr"] = par.where(df[parent].notna())
        out[f"{tag}_n"] = pn
        raw = (ps / pn.where(pn > 0)).fillna(0.0)  # 種牡馬の全体の超過(引き寄せなし)
        for name, col in CONDITIONS.items():
            cn, cs = prior_sums(df, [parent, col], ex)
            # その条件での超過 − 種牡馬の全体の超過を、回数の少なさで 0 に引き寄せる(どこでも走るだけなら 0)
            out[f"{tag}_{name}_apt"] = ((cs - cn * raw) / (cn + k)).where(df[parent].notna() & df[col].notna())
        # ダートのスローの得意: スローだったダートの出走での超過(種牡馬の全体に引き寄せ)− 全体。レースの条件には使わない
        slow = (df["course_type"].astype(str) == "ダ") & (df["pace_z"] > 0.5)
        ss = prior_sums(df.assign(_slow_ex=df[ex].where(slow, 0.0)), [parent], "_slow_ex")[1]
        n_slow = prior_sums(df.assign(_is_slow=slow.astype(float)), [parent], "_is_slow")[1]
        out[f"{tag}_dslow_apt"] = ((ss - n_slow * raw) / (n_slow + k)).where(df[parent].notna())
    return pd.concat([df.drop(columns=[c for c in df.columns if c.startswith("_")]), pd.DataFrame(out, index=df.index)],
                     axis=1)


APT_COLUMNS = [f"{t}_{n}" for t in ("sire", "dam") for n in
               ["ex_shr", "ctdist_apt", "venuect_apt", "course_apt", "racename_apt", "turfwet_apt", "dslow_apt"]]
