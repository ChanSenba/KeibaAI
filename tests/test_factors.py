import numpy as np
import pandas as pd
import pytest

from keibaai import factors
from keibaai.features import BIAS_FEATURES, _track_bias, build_features, load_results

from test_model import synthetic_db


def day(date, front_wins=True, venue="東京"):
    # 9頭立て。前に付けた馬(early_pos 小)が上位なら先行有利、内の馬(馬番小)が上位なら内有利
    rows = []
    for n in range(1, 10):
        early = n / 9
        finish = n if front_wins else 10 - n
        rows.append({"venue": venue, "course_type": "芝", "date": pd.Timestamp(date), "finish": finish,
                     "number": n, "field_size": 9, "early_pos": early})
    return pd.DataFrame(rows)


def test_bias_uses_only_previous_days_within_window():
    df = pd.concat([day("2025-03-01"), day("2025-03-02", front_wins=False), day("2025-04-20")])
    b = _track_bias(df).set_index("date")
    assert np.isnan(b.loc["2025-03-01", "track_front_bias"])  # 初日は前の日がない
    assert b.loc["2025-03-02", "track_front_bias"] == 1.0 and b.loc["2025-03-02", "track_inner_bias"] == 1.0
    assert b.loc["2025-03-02", "track_front_bias"] == 1.0  # 前日が先行有利・内有利
    assert np.isnan(b.loc["2025-04-20", "track_front_bias"])  # 窓(10日)より前は使わない


def test_bias_ignores_same_day_and_later_results():
    a = pd.concat([day("2025-03-01"), day("2025-03-02")])
    b = pd.concat([day("2025-03-01"), day("2025-03-02", front_wins=False), day("2025-03-03", front_wins=False)])
    ba, bb = _track_bias(a).set_index("date"), _track_bias(b).set_index("date")
    assert ba.loc["2025-03-02"].equals(bb.loc["2025-03-02"])  # 2日目の偏りは2日目以後の結果に左右されない


def test_bias_is_per_venue():
    df = pd.concat([day("2025-03-01", venue="東京"), day("2025-03-02", front_wins=False, venue="京都")])
    b = _track_bias(df)
    assert b[b["venue"] == "京都"]["track_front_bias"].isna().all()


@pytest.fixture(scope="module")
def df():
    return build_features(load_results(synthetic_db()))


def test_factor_features_are_known_and_leak_free(df):
    for feats in {**factors.FACTORS, **factors.OVERALL}.values():
        assert set(feats) <= set(df.columns)
        assert not {"finish", "win", "top3", "win_odds", "popularity", "passing", "early_pos"} & set(feats)
    assert set(BIAS_FEATURES) <= set(df.columns)


def test_evaluate_factors_reports_every_factor(df):
    reports, uniform = factors.evaluate_factors(df, "2025-09-01", rounds=50)
    names = [r.name for r in reports]
    assert names == [*factors.FACTORS, *factors.OVERALL]
    by = {r.name: r for r in reports}
    assert by["馬の質"].win_logloss < uniform and by["馬の質"].win_auc > 0.6  # 合成データでは馬の強さが効く
    assert "全馬が同じ勝率" in factors.format_factor_report(reports, uniform)


def test_predict_factors_sum_to_one_per_race(df):
    train, target = df[df["date"] < "2025-09-01"], df[df["date"] >= "2025-09-01"]
    p = factors.predict_factors(train, target, rounds=30)
    assert list(p.columns) == list(factors.FACTORS)
    assert np.allclose(p["馬の質"].groupby(target["race_id"]).sum(), 1.0)


def test_cli_commands_return_without_falling_through(tmp_path, capsys):
    """表示だけのコマンドが、最後に collect の処理まで進んで落ちないこと(factors で起きた不具合の再発防止)。"""
    import sqlite3

    from keibaai import cli

    path = tmp_path / "k.db"
    dst = sqlite3.connect(path)
    synthetic_db(n_days=30, races_per_day=3).backup(dst)
    dst.close()
    for cmd in ("factors", "backtest", "train"):
        cli.main([cmd, "--db", str(path), "--test-from", "2025-06-01"])
    out = capsys.readouterr().out
    assert "全馬が同じ勝率" in out and "単勝を毎回" in out and "モデル" in out


def test_gate_bias_counts_inner_middle_outer_before_the_day(tmp_path):
    import pandas as pd

    from keibaai import db, factor_view

    conn = db.connect(tmp_path / "k.db")
    rows = []
    for i, (date, rail) in enumerate([("2026-09-01", "A"), ("2026-09-02", "A"), ("2026-09-03", "B"), ("2026-10-20", "A")]):
        rid = f"R{i}"
        conn.execute("INSERT INTO races (race_id, date, venue, course_type, distance, course, rail) "
                     "VALUES (?, ?, '京都', '芝', 1600, '外', ?)", (rid, date, rail))
        for n in range(1, 7):  # 内(1,2)が 1・2 着、外(5,6)が 3 着と着外
            finish = {1: 1, 2: 2, 5: 3}.get(n, n + 1)
            rows.append((rid, n, finish))
    conn.executemany("INSERT INTO results (race_id, number, finish) VALUES (?, ?, ?)", rows)
    conn.commit()
    race = pd.Series({"venue": "京都", "course_type": "芝", "distance": 1600, "course": "外", "rail": "A"})
    t = factor_view.gate_bias(conn, race, "2026-10-17").set_index("条件")
    assert t.loc["同コース", "レース数"] == 3 and t.loc["同じ Aコース", "レース数"] == 2  # 10/20 は数えない
    assert t.loc["同コース", "内枠"] == 1.0 and t.loc["同コース", "外枠"] == 0.5
    assert t.loc["同コース", "内−外"] == 50.0
    assert factor_view.gate_bias(conn, pd.Series({"venue": None}), "2026-10-17").empty