"""これから走るレースの出馬表に、勝率と期待値を付ける。"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import db
from .entries import apply_odds, parse_odds_json, parse_shutuba
from .features import (ABILITY_MODEL_FEATURES, CATEGORICAL, MARKET_MODEL_FEATURES, RESIDUAL_MODEL_FEATURES, build_features,
                       load_results, result_mask)
from .flow_review import forecast_flow
from .factors import COLUMN, FACTORS
from .jvtraining import add_jv_features
from .model import PARAMS, TRAIN_FROM, fit_residual, has_market, normalize_within_race, predict_residual, train_rows


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


def snapshot_odds(conn: sqlite3.Connection, fetcher, date: str, now: dt.datetime | None = None,
                  race_ids: list[str] | None = None) -> int:
    """date(YYYY-MM-DD)の各レースの単勝オッズを、取得時刻つきで odds_snapshots に追記し、記録したレース数を返す。

    レースは出馬表を取り込み済みのもの(entry_races)。オッズが取れない・空のレースは飛ばす。
    """
    stamp = (now or dt.datetime.now()).isoformat(timespec="seconds")
    ids = race_ids if race_ids is not None else [r[0] for r in conn.execute(
        "SELECT race_id FROM entry_races WHERE date = ? ORDER BY race_id", (date,))]
    n = 0
    for race_id in ids:
        try:
            text = fetcher.odds_json(race_id)
            odds = parse_odds_json(text)
            official = (json.loads(text).get("data") or {}).get("official_datetime")
        except Exception as e:
            print(f"odds {race_id}: {e}")
            continue
        if not odds:
            print(f"skip {race_id}: オッズがまだありません")
            continue
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO odds_snapshots (race_id, number, snapped_at, official_at, win_odds, popularity) VALUES (?,?,?,?,?,?)",
                [(race_id, num, stamp, official, o, pop) for num, (o, pop) in odds.items()])
        n += 1
    return n


def load_entries(conn: sqlite3.Connection, date: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT e.*, c.name AS race_name, c.date, c.venue, c.round, c.course_type, c.direction, c.course, c.rail, c.distance, c.ground,
               c.fetched_at,
               p.sire_id, p.damsire_id, t.rank AS train_rank, t.critic AS train_critic
        FROM entries e JOIN entry_races c USING (race_id) LEFT JOIN pedigree p USING (horse_id)
        LEFT JOIN training t ON t.race_id = e.race_id AND t.number = e.number AND t.seq = 0
        WHERE c.date = ?
        """,
        conn, params=(date,),
    )
    return add_jv_features(conn, df)


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
    entries["is_entry"] = True
    return build_features(pd.concat([hist, entries], ignore_index=True)), hist


CACHE_VERSION = 6  # 能力のみの勝率と要素ごとの勝率をニューラルネットにし、印用の勝率(p_win_mix)を足した。旧予測は使わない。
CACHE_DIR = "model_cache"  # DB と同じフォルダの中に作る


def _db_file(conn: sqlite3.Connection) -> Path | None:
    """接続先の DB ファイル。メモリ上の DB なら None(保存しない)。"""
    for _, name, file in conn.execute("PRAGMA database_list"):
        if name == "main":
            return Path(file) if file else None
    return None


def _data_key(conn: sqlite3.Connection, date: str, with_entries: bool, extra: tuple = ()) -> str:
    """計算結果を左右するデータの指紋。学習に使う結果・血統・調教・(予想なら)その日の出馬表とオッズ、特徴量と学習の設定。

    どれかが変われば別の値になるので、古い保存を誤って使うことはない。
    """
    # 計算方法だけの変更やファクター定義の変更でも、同じ名前の特徴量を再利用しない。
    source = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        source.update(file.name.encode())
        source.update(file.read_bytes())
    parts: list = [
        CACHE_VERSION, TRAIN_FROM, date, extra, ABILITY_MODEL_FEATURES, MARKET_MODEL_FEATURES, RESIDUAL_MODEL_FEATURES,
        sorted(PARAMS.items()),
        db.model_revision(conn), source.hexdigest(), CATEGORICAL, FACTORS,
        lgb.__version__, np.__version__, pd.__version__,
    ]
    if with_entries:
        rows = conn.execute(
            "SELECT e.* FROM entries e JOIN entry_races c USING (race_id) WHERE c.date = ? ORDER BY e.race_id, e.number",
            (date,)).fetchall()
        # 取得時刻だけ変わっても、中身が同じなら使い回す(列はあとから足されるので、名前で外す)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(entry_races)") if r[1] != "fetched_at"]
        races = conn.execute(f"SELECT {','.join(cols)} FROM entry_races WHERE date = ? ORDER BY race_id", (date,)).fetchall()
        parts += [rows, races]
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, default=str).encode()).hexdigest()[:20]


def _cached(conn: sqlite3.Connection, kind: str, date: str, with_entries: bool, extra: tuple, compute):
    """compute() の結果を DB の隣に保存し、データが同じなら次からはそれを返す。

    予想や混ぜ方の決定は、毎回1万レース以上で学習し直すので数十秒かかる。出馬表やオッズを取り直したり、
    成績を足したりすると指紋が変わり、計算し直す。同じ日・同じ種類の古い保存は消す。
    """
    path = _db_file(conn)
    if path is None or db.model_revision(conn) is None:
        return compute()
    folder = path.parent / CACHE_DIR
    file = folder / f"{kind}-{date}-{_data_key(conn, date, with_entries, extra)}.pkl"
    if file.exists():
        try:
            with file.open("rb") as f:
                return pickle.load(f)
        except Exception:  # 壊れた保存は捨てて計算し直す
            file.unlink(missing_ok=True)
    out = compute()
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{kind}-{date}-*.pkl"):
        old.unlink(missing_ok=True)
    tmp = file.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(out, f)
    tmp.replace(file)
    return out


def blend_for_day(conn: sqlite3.Connection, date: str, valid_months: int = 12):
    """date の買い方に使う、オッズ込みのモデルとオッズの混ぜ方。date の直前 valid_months か月の結果で決める(それ以後は使わない)。

    結果は保存して使い回す(その日の出馬表やオッズには左右されないので、成績などが変わるまで同じ)。
    """
    def compute():
        from .blend import fit_blend
        from .model import evaluate

        df, _ = day_frame(conn, date)
        done = df[result_mask(df)]
        valid_from = pd.Timestamp(date) - pd.DateOffset(months=valid_months)
        _, valid = evaluate(done, str(valid_from.date()), use_market=True)
        return fit_blend(valid)

    return _cached(conn, "blend", date, False, (valid_months,), compute)


def predict_day(conn: sqlite3.Connection, date: str, rounds: int = 300) -> pd.DataFrame:
    """date(YYYY-MM-DD)の各馬の勝率(能力のみ / オッズ込み)と期待値(勝率×単勝オッズ)を返す。

    学習には、その日より前の結果だけを使う。期待値は 1 を超えると、予測どおりなら長い目で得になる目安。
    結果は保存して使い回す。出馬表やオッズを取り直すと計算し直す。
    """
    return _cached(conn, "predict", date, True, (rounds,), lambda: _predict_day(conn, date, rounds)).copy()


def mix_with_odds(target: pd.DataFrame, p_ability: pd.Series) -> pd.Series:
    """印に使う勝率。能力のみの勝率(ニューラルネット)と、その時点の単勝オッズの確率を、レース単位の対数で混ぜる。

    重みは JRA-VAN の時系列オッズ(2025-09〜2026-10 の約3,900レース)で、予想を保存する時点のオッズを使って決めた。
    当日9時のオッズなら a=0.42・b=0.78、前日21時なら a=0.74・b=0.36(前日はオッズが当てにならないぶんモデルが重い。
    どちらも全期間で決めた値)。混ぜ方を2025年9〜12月だけで決めて2026年で測ると、当日9時で勝ち馬の対数尤度は オッズだけ −2.0254、
    いまのオッズ込みモデル(確定オッズで学習)−2.0285 に対して −1.9878、◎の的中は 31.5% → 32.4%。
    ただし確定オッズで払い戻すと回収率は上がらなかったので、★と期待値はいままでどおりオッズ込みモデルで出す。
    オッズが前日に取られたもの(fetched_at の日付がレースより前)なら前日の重みを使う。オッズのないレースは空。
    """
    from .blend import Blend, apply_blend

    t = target[["race_id", "number", "win_odds", "date"]].copy()
    t["p_win"] = p_ability
    fetched = pd.to_datetime(target["fetched_at"] if "fetched_at" in target else pd.NaT, errors="coerce")
    eve = pd.Series(fetched, index=target.index).dt.normalize() < pd.to_datetime(t["date"])
    out = pd.Series(np.nan, index=target.index)
    ok = t["win_odds"].notna() & (t["win_odds"] > 0)
    full = ok.groupby(t["race_id"]).transform("all")
    for is_eve, (a, b) in ((False, MIX_MORNING), (True, MIX_EVENING)):
        part = t[full & (eve == is_eve)]
        if part.empty:
            continue
        mixed = apply_blend(part.reset_index(), Blend(a, b, float("nan"), float("nan")))
        out.loc[mixed["index"].to_numpy()] = mixed["p_win"].to_numpy()
    return out


MIX_MORNING = (0.42, 0.78)  # 能力のみの勝率の重み a・オッズの確率の重み b(当日のオッズ)
MIX_EVENING = (0.74, 0.36)  # 前日のオッズ


def _predict_day(conn: sqlite3.Connection, date: str, rounds: int) -> pd.DataFrame:
    from .nn_model import fit_predict

    df, hist = day_frame(conn, date)
    train, target = df[result_mask(df)], df[df["is_entry"]].copy()

    p_ability, parts = fit_predict(train, target)  # 能力のみの勝率と、要素ごとの勝率(馬の質・騎手と厩舎・展開と馬場…)
    target["p_win_ability"] = p_ability
    target["p_win_market"] = np.where(has_market(target), predict_residual(fit_residual(train), target), np.nan)
    ok = target["p_win_market"].notna()
    target.loc[ok, "p_win_market"] = normalize_within_race(target[ok], target.loc[ok, "p_win_market"].to_numpy())
    target["p_win_mix"] = mix_with_odds(target, target["p_win_ability"])
    target["ev_ability"] = target["p_win_ability"] * target["win_odds"]
    target["ev_market"] = target["p_win_market"] * target["win_odds"]
    for name, col in COLUMN.items():
        target[col] = parts[name]
    cols = ["race_id", "date", "venue", "round", "race_name", "number", "horse_name", "jockey", "win_odds",
            "popularity", "p_win_ability", "p_win_market", "p_win_mix", "ev_ability", "ev_market", *COLUMN.values()]
    out = target[cols].copy()
    out["venue"] = out["venue"].astype(str)  # 特徴量用の category のままだと画面側で文字連結ができない
    out.attrs["flow"] = forecast_flow(target)  # 発走前の展開予想。weekly が予想と一緒に保存する
    return out.sort_values(["venue", "round", "number"]).reset_index(drop=True)
