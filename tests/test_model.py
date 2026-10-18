import datetime as dt

import numpy as np
import pandas as pd
import pytest

from keibaai import db
from keibaai.features import build_features, load_results
from keibaai import model
from keibaai.model import baseline_market, evaluate, format_report, train_rows
from keibaai.parse import ParsedRace, Race, Result


def synthetic_db(n_days=120, races_per_day=6, seed=0):
    """潜在的な馬の強さで着順が決まる架空のデータ(実データではない)。"""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1, 60)
    jockey_skill = rng.normal(0, 0.3, 10)
    conn = db.connect(":memory:")
    start = dt.date(2025, 1, 4)
    for d in range(n_days):
        date = start + dt.timedelta(days=7 * d)
        day_horses = rng.permutation(60)  # 同じ日に同じ馬が2回走ることはない
        for r in range(races_per_day):
            race_id = f"2025{r + 1:02d}{d // 100:02d}{d % 100:02d}{r + 1:02d}"
            horses = day_horses[r * 10:(r + 1) * 10]
            jockeys = rng.integers(0, 10, 10)
            score = strength[horses] + jockey_skill[jockeys] + rng.normal(0, 0.8, 10)
            order = np.argsort(-score)
            race = Race(race_id=race_id, date=date.isoformat(), venue="東京", round=r + 1,
                        course_type="芝", distance=1600 + 400 * (r % 3), ground="良")
            results = []
            for rank, i in enumerate(order, start=1):
                results.append(Result(
                    race_id=race_id, number=int(i) + 1, finish=rank, bracket=1 + int(i) // 2,
                    horse_id=f"h{horses[i]}", sex_age="牡4", weight_carried=56.0,
                    jockey_id=f"j{jockeys[i]}", trainer_id=f"t{horses[i] % 7}",
                    last3f=34.0 + rng.normal(), body_weight=480, body_weight_diff=0,
                    win_odds=float(np.clip(8 * np.exp(-0.9 * (strength[horses[i]] + jockey_skill[jockeys[i]])), 1.1, 99)),
                    popularity=1))
            # 人気はオッズ順
            for pop, res in enumerate(sorted(results, key=lambda x: x.win_odds), start=1):
                res.popularity = pop
            db.save_race(conn, ParsedRace(race=race, results=results))
    return conn


@pytest.fixture(scope="module")
def df():
    return build_features(load_results(synthetic_db()))


def test_first_start_has_no_history(df):
    first = df.groupby("horse_id").head(1)
    assert (first["n_prev"] == 0).all()
    assert first["prev_finish"].isna().all() and first["win_rate"].isna().all()


def test_history_uses_only_earlier_races(df):
    h = df[df["horse_id"] == df["horse_id"].iloc[0]].sort_values("date")
    assert h["prev_finish"].iloc[1:].tolist() == h["finish"].iloc[:-1].tolist()
    # n_prev 番目までの勝率 = これまでの勝ち数 / これまでの出走数
    wins_before = (h["finish"] == 1).cumsum().shift(1)
    expected = (wins_before / h["n_prev"].replace(0, np.nan)).iloc[1:]
    assert np.allclose(h["win_rate"].iloc[1:], expected)


def test_jockey_stats_exclude_same_day(df):
    day = df[df["date"] == df["date"].min()]
    assert day["jockey_n"].isna().sum() == 0 and (day["jockey_n"] == 0).all()
    assert day["jockey_win_rate"].isna().all()


def test_result_columns_do_not_leak_into_features(df):
    from keibaai.features import BASE_FEATURES

    assert not {"finish", "win", "top3", "time_sec", "win_odds", "popularity"} & set(BASE_FEATURES)


def test_model_beats_chance_and_reports(df):
    cut = df["date"].sort_values().iloc[int(len(df) * 0.7)].date().isoformat()
    rep, test = evaluate(df, cut, use_market=False, rounds=60)
    assert rep.win_auc > 0.6 and rep.top3_auc > 0.6
    assert rep.win_logloss < 0.35  # 10頭立てでの一様予測は約0.325。それ以下ならよい
    sums = test.groupby("race_id")["p_win"].sum()
    assert np.allclose(sums, 1.0)
    text = format_report([baseline_market(test), rep])
    assert "基準:オッズ/1番人気" in text and "オッズなし" in text


def test_evaluate_rejects_empty_split(df):
    with pytest.raises(ValueError):
        evaluate(df, "2100-01-01", use_market=False)


def test_class_and_early_position_parsers():
    from keibaai.features import parse_class, parse_early_position

    assert parse_class("3歳未勝利") == 1 and parse_class("3歳新馬") == 0
    assert parse_class("第44回ジャパンC(GI)") == 8 and parse_class("京都記念(GII)") == 7
    assert parse_class("1勝クラス") == 2 and np.isnan(parse_class(None))
    assert parse_early_position("13-12-12-7") == 13 and np.isnan(parse_early_position(""))
    assert np.isnan(parse_early_position(float("nan"))) and np.isnan(parse_early_position(None))


def test_new_features_are_past_only(df):
    # 初出走では前走との比較は作れない
    first = df.groupby("horse_id").head(1)
    assert first["distance_change"].isna().all() and first["jockey_changed"].isna().all()
    assert first["early_pos_rel_3"].isna().all()


def test_build_features_with_mixed_null_passing():
    """passing が NULL の行と値ありの行が混ざっても落ちない(pandas は NULL を NaN にする)。"""
    conn = synthetic_db(n_days=3)
    conn.execute("UPDATE results SET passing = '3-3-2-1' WHERE number % 2 = 0")  # 残りは NULL
    out = build_features(load_results(conn))
    assert out["early_pos_rel_3"].notna().any()


def test_train_rows_drop_races_before_train_from(monkeypatch):
    """学習の行は TRAIN_FROM 以後だけ。それより前の成績は過去走の特徴量にだけ使う。"""
    df = pd.DataFrame({"date": pd.to_datetime(["2018-12-28", "2019-01-05", "2020-06-01"])})
    assert list(train_rows(df)["date"].dt.year) == [2019, 2020]
    monkeypatch.setattr(model, "TRAIN_FROM", "2020-01-01")
    assert list(train_rows(df)["date"].dt.year) == [2020]
