"""これから走るレースの出馬表に、勝率と期待値を付ける。"""
from __future__ import annotations

import datetime as dt
import sqlite3

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import db
from .entries import apply_odds, parse_odds_json, parse_shutuba
from .features import ABILITY_MODEL_FEATURES, CATEGORICAL, MARKET_MODEL_FEATURES, build_features, load_results
from .factors import COLUMN, FACTORS, predict_factors
from .jvtraining import add_jv_features
from .model import PARAMS, normalize_within_race


def fetch_entries(conn: sqlite3.Connection, fetcher, date: str, now: dt.datetime | None = None,
                  on_race=None, race_ids: list[str] | None = None) -> int:
    """date(YYYY-MM-DD)の出馬表と単勝オッズを取得して保存し、保存したレース数を返す。

    race_ids を渡すとそのレースだけを取る。出馬表が出ていないレースは飛ばす。
    オッズが取れないときは出馬表だけ保存する。
    """
    stamp = (now or dt.datetime.now()).isoformat(timespec="seconds")
    ids = race_ids if race_ids is not None else fetcher.race_ids_on(date.replace("-", ""))
    n = 0
    for i, race_id in enumerate(ids):
        if on_race:
            on_race(i, len(ids), race_id)
        try:
            parsed = parse_shutuba(fetcher.shutuba_html(race_id), race_id)
        except ValueError as e:
            print(f"skip {race_id}: {e}")
            continue
        if not parsed.entries:  # 馬番の確定前(特別登録の段階)は馬が1頭もいない。保存せず、取得済みにもしない
            print(f"skip {race_id}: 出馬表がまだ確定していません(0頭)")
            continue
        if not parsed.race.date:
            parsed.race.date = date
        try:
            apply_odds(parsed, parse_odds_json(fetcher.odds_json(race_id)))
        except Exception as e:  # オッズが取れなくても出馬表は使える
            print(f"odds {race_id}: {e}")
        db.save_entries(conn, parsed, stamp)
        n += 1
        print(f"saved {race_id} ({len(parsed.entries)}頭)")
    return n


def load_entries(conn: sqlite3.Connection, date: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT e.*, c.name AS race_name, c.date, c.venue, c.round, c.course_type, c.direction, c.distance, c.ground,
               p.sire_id, p.damsire_id, t.rank AS train_rank, t.critic AS train_critic
        FROM entries e JOIN entry_races c USING (race_id) LEFT JOIN pedigree p USING (horse_id)
        LEFT JOIN training t ON t.race_id = e.race_id AND t.number = e.number AND t.seq = 0
        WHERE c.date = ?
        """,
        conn, params=(date,),
    )
    return add_jv_features(conn, df)


def _fit(train: pd.DataFrame, features: list[str], rounds: int):
    cats = [c for c in CATEGORICAL if c in features]
    return lgb.train(PARAMS, lgb.Dataset(train[features], train["win"], categorical_feature=cats), rounds)


def day_frame(conn: sqlite3.Connection, date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """その日より前の結果と、その日の出馬表をつないで特徴量を作る。(特徴量の表, 過去の結果) を返す。

    出馬表の行は finish が空(これから走るので結果がない)。
    """
    entries = load_entries(conn, date)
    if entries.empty:
        raise ValueError(f"{date} の出馬表がありません。先に出馬表を取得してください")
    hist = load_results(conn)
    hist = hist[hist["date"] < pd.Timestamp(date)]
    if hist.empty:
        raise ValueError("学習に使える過去の結果がありません")

    entries = entries.copy()
    for c in ("win_odds", "popularity", "weight_carried", "body_weight", "body_weight_diff", "distance"):
        entries[c] = pd.to_numeric(entries[c], errors="coerce")  # 空は None(object)で来るので数値にそろえる
    entries["date"] = pd.Timestamp(date)
    entries["finish"] = np.nan  # これから走るので結果はない
    return build_features(pd.concat([hist, entries], ignore_index=True)), hist


def blend_for_day(conn: sqlite3.Connection, date: str, valid_months: int = 12):
    """date の予想に使う、モデルとオッズの混ぜ方。date の直前 valid_months か月の結果で決める(それ以後は使わない)。"""
    from .blend import fit_blend
    from .model import evaluate

    df, _ = day_frame(conn, date)
    done = df[df["finish"].notna()]
    valid_from = pd.Timestamp(date) - pd.DateOffset(months=valid_months)
    _, valid = evaluate(done, str(valid_from.date()), use_market=False)
    return fit_blend(valid)


def predict_day(conn: sqlite3.Connection, date: str, rounds: int = 300) -> pd.DataFrame:
    """date(YYYY-MM-DD)の各馬の勝率(能力のみ / オッズ込み)と期待値(勝率×単勝オッズ)を返す。

    学習には、その日より前の結果だけを使う。期待値は 1 を超えると、予測どおりなら長い目で得になる目安。
    """
    df, hist = day_frame(conn, date)
    is_entry = df["finish"].isna()
    train, target = df[~is_entry], df[is_entry].copy()

    ability = _fit(train, ABILITY_MODEL_FEATURES, rounds)
    market = _fit(train, MARKET_MODEL_FEATURES, rounds)
    target["p_win_ability"] = normalize_within_race(target, ability.predict(target[ABILITY_MODEL_FEATURES]))
    pm = market.predict(target[MARKET_MODEL_FEATURES])
    target["p_win_market"] = np.where(target["win_odds"].notna(), pm, np.nan)
    ok = target["p_win_market"].notna()
    target.loc[ok, "p_win_market"] = normalize_within_race(target[ok], target.loc[ok, "p_win_market"].to_numpy())
    target["ev_ability"] = target["p_win_ability"] * target["win_odds"]
    target["ev_market"] = target["p_win_market"] * target["win_odds"]
    fp = predict_factors(train, target, rounds=rounds)  # 要素ごとの勝率(馬の質・騎手と厩舎・展開と馬場)
    for name, col in COLUMN.items():
        target[col] = fp[name]
    cols = ["race_id", "date", "venue", "round", "race_name", "number", "horse_name", "jockey", "win_odds",
            "popularity", "p_win_ability", "p_win_market", "ev_ability", "ev_market", *COLUMN.values()]
    out = target[cols].copy()
    out["venue"] = out["venue"].astype(str)  # 特徴量用の category のままだと画面側で文字連結ができない
    return out.sort_values(["venue", "round", "number"]).reset_index(drop=True)
