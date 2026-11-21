import numpy as np
import pandas as pd
import pytest

from keibaai.devig import market_probs, power, proportional, shin
from keibaai.devig_trials import (compare, evaluate, fit_power_gamma, load_market,
                                  markdown_report, resolve_fit_to)
from keibaai import db


def test_shin_sums_to_one_and_shifts_toward_favorite():
    odds = np.array([1.5, 3.0, 6.0, 12.0, 30.0, 80.0])  # 逆数の合計は約1.25(JRAの単勝の控除に近い)
    p, z = shin(odds)
    assert p.sum() == pytest.approx(1.0)
    assert 0.0 < z < 1.0
    prop = proportional(odds)
    assert p[0] > prop[0]  # 人気馬は比例より厚く
    assert p[-1] < prop[-1]  # 人気薄は比例より薄く(控除が厚く乗っている分を剥がす)


def test_shin_matches_proportional_without_margin():
    odds = np.array([2.0, 4.0, 4.0])  # 逆数の合計がちょうど1(控除なし)
    p, z = shin(odds)
    assert z == 0.0 and p == pytest.approx(proportional(odds))


def test_shin_equal_odds_gives_equal_probs():
    p, z = shin(np.array([2.0, 2.0, 2.0]))  # 逆数の合計 1.5 で、Shin の計算を通る
    assert z > 0 and p == pytest.approx([1 / 3] * 3)


def test_shin_satisfies_defining_equation():
    odds = np.array([1.5, 3.0, 6.0, 12.0, 30.0, 80.0])
    p, z = shin(odds)
    inv = 1.0 / odds
    # Shin の定義式: π_i^2 / S = z p_i + (1-z) p_i^2 (π はオッズの逆数、S はその合計)
    assert inv**2 / inv.sum() == pytest.approx(z * p + (1 - z) * p**2, rel=1e-6)


def test_market_probs_groups_by_race_and_marks_incomplete_nan():
    df = pd.DataFrame({"race_id": ["a"] * 3 + ["b"] * 2, "win_odds": [2.0, 4.0, 4.0, 2.0, None]})
    prop = market_probs(df, "proportional")
    assert prop.iloc[:3].tolist() == pytest.approx([0.5, 0.25, 0.25])
    assert prop.iloc[3:].isna().all()  # オッズの欠けたレースは丸ごと NaN
    sh = market_probs(df, "shin")
    assert sh.iloc[:3].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        market_probs(df, "unknown")


def test_market_probs_with_duplicated_index():
    df = pd.DataFrame({"race_id": ["a", "a", "b", "b"], "win_odds": [2.0, 2.0, 3.0, 3.0]}, index=[0, 0, 1, 1])
    prop = market_probs(df, "proportional")
    assert prop.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])


def _results_db(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    with conn:
        conn.execute("INSERT INTO races (race_id, date) VALUES ('r1', '2024-06-01'), ('r2', '2024-06-01')")
        conn.executemany("INSERT INTO results (race_id, number, finish, win_odds) VALUES (?,?,?,?)",
                         [("r1", 1, 1, 2.0), ("r1", 2, 2, 4.0), ("r1", 3, 3, 4.0),
                          ("r2", 1, 2, 2.0), ("r2", 2, 1, None)])  # r2 は勝ち馬のオッズが欠けるので使わない
    return conn


def test_load_market_and_compare(tmp_path):
    conn = _results_db(tmp_path)
    df = load_market(conn, "2024-01-01", "2025-01-01")
    assert set(df["race_id"]) == {"r1"}
    out = compare(df, reps=100)
    assert list(out["方法"]) == ["比例", "Shin", "差 (Shin − 比例)"]
    assert out.loc[0, "レース数"] == 1


ODDS6 = np.array([1.5, 3.0, 6.0, 12.0, 30.0, 80.0])  # 逆数の合計は約1.25


def test_power_matches_proportional_at_gamma_one():
    assert power(ODDS6, 1.0) == pytest.approx(proportional(ODDS6))


def test_power_sums_to_one_and_direction_of_gamma():
    prop = proportional(ODDS6)
    thick = power(ODDS6, 1.1)
    assert thick.sum() == pytest.approx(1.0)
    assert thick[0] > prop[0]  # γ>1: 人気馬は比例より厚く
    assert thick[-1] < prop[-1]  # γ>1: 人気薄は比例より薄く(FLB の補正方向)
    thin = power(ODDS6, 0.9)
    assert thin[0] < prop[0] and thin[-1] > prop[-1]  # γ<1 では逆に人気薄が持ち上がる
    assert power(np.array([3.0, 3.0, 3.0]), 1.15) == pytest.approx([1 / 3] * 3)  # 同オッズなら γ によらず均等


def test_power_rejects_nonpositive_gamma():
    with pytest.raises(ValueError):
        power(ODDS6, 0.0)
    with pytest.raises(ValueError):
        power(ODDS6, -1.0)


def test_market_probs_power_needs_gamma_and_applies_per_race():
    df = pd.DataFrame({"race_id": ["a"] * 3 + ["b"] * 2, "win_odds": [2.0, 4.0, 4.0, 2.0, 4.0]})
    with pytest.raises(ValueError):
        market_probs(df, "power")
    with pytest.raises(ValueError):
        market_probs(df, "power", gamma=0.0)
    p = market_probs(df, "power", gamma=1.1)
    assert p.iloc[:3].sum() == pytest.approx(1.0) and p.iloc[3:].sum() == pytest.approx(1.0)
    assert p.iloc[0] > market_probs(df, "proportional").iloc[0]  # γ>1 で人気馬が厚くなることが df 単位でも出る


def _sampled_market(gamma_true: float, seed: int, races: int = 2000, prefix: str = "r") -> pd.DataFrame:
    """勝ち馬を Power(γ=gamma_true) の確率で抽選した、load_market と同じ形の df。"""
    rng = np.random.default_rng(seed)
    winners = rng.choice(len(ODDS6), size=races, p=power(ODDS6, gamma_true))
    return pd.DataFrame(
        {"race_id": np.repeat([f"{prefix}{i}" for i in range(races)], len(ODDS6)),
         "win_odds": np.tile(ODDS6, races),
         "win": (np.tile(np.arange(len(ODDS6)), races) == np.repeat(winners, len(ODDS6))).astype(int)})


def test_fit_power_gamma_recovers_generating_gamma():
    assert fit_power_gamma(_sampled_market(1.10, seed=42, races=4000)) == pytest.approx(1.10, abs=0.05)
    assert fit_power_gamma(_sampled_market(0.95, seed=7, races=4000)) < 1.0


def test_fit_power_gamma_rejects_empty():
    with pytest.raises(ValueError):
        fit_power_gamma(pd.DataFrame({"race_id": [], "win": [], "win_odds": []}))


def test_resolve_fit_to_defaults_to_test_start_and_rejects_overlap():
    assert resolve_fit_to("2019-01-01", None, "2024-01-01") == "2024-01-01"
    assert resolve_fit_to("2019-01-01", "2023-01-01", "2024-01-01") == "2023-01-01"
    with pytest.raises(ValueError):
        resolve_fit_to("2019-01-01", "2024-06-01", "2024-01-01")  # 選択期間が評価期間に食い込む
    with pytest.raises(ValueError):
        resolve_fit_to("2024-01-01", None, "2024-01-01")  # 選択期間が空


def test_evaluate_gamma_does_not_depend_on_test_period():
    fit = _sampled_market(1.10, seed=1, races=500, prefix="f")
    test = _sampled_market(1.10, seed=2, races=100, prefix="t")
    rigged = test.copy()
    rigged["win"] = 0
    rigged.loc[rigged.groupby("race_id")["win_odds"].idxmax(), "win"] = 1  # 勝ち馬を全部人気薄に入れ替える
    g1, _ = evaluate(fit, test, reps=50)
    g2, _ = evaluate(fit, rigged, reps=50)
    assert g1 == g2  # 評価期間の結果をどう変えても、γ の選択は変わらない(漏れていない)


def test_compare_includes_power_rows_when_gamma_given():
    out = compare(_sampled_market(1.05, seed=3, races=60), gamma=1.05, reps=50)
    assert list(out["方法"]) == ["比例", "Shin", "Power (γ=1.05)", "差 (Shin − 比例)", "差 (Power − 比例)",
                                "差 (Power − Shin)"]
    row = out[out["方法"] == "差 (Power − 比例)"].iloc[0]
    assert row["区間下限"] <= row["勝ち馬の対数尤度"] <= row["区間上限"]


def test_markdown_report_has_table_gamma_and_periods():
    gamma, out = evaluate(_sampled_market(1.05, seed=4, races=200, prefix="f"),
                          _sampled_market(1.05, seed=3, races=60, prefix="t"), reps=50)
    md = markdown_report(out, test_from="2024-01-01", test_to="2025-01-01",
                         fit_from="2019-01-01", fit_to="2024-01-01", gamma=gamma,
                         n_races=60, z_mean=0.05, z_failed=2)
    assert "| 比例 |" in md and "Power" in md and "差 (Power − 比例)" in md
    assert f"{gamma:.2f}" in md and "2019-01-01" in md and "2024-01-01" in md
    assert "採用可否" in md


def test_fit_power_gamma_warns_at_grid_edge():
    with pytest.warns(UserWarning, match="張り付きました"):
        gamma = fit_power_gamma(_sampled_market(1.50, seed=5, races=1500))
    assert gamma == 1.20  # 真の最適が範囲の外にあるので、上限に張り付く


def test_evaluate_rejects_overlapping_races():
    df = _sampled_market(1.05, seed=6, races=50)
    with pytest.raises(ValueError):
        evaluate(df, df, reps=10)  # 選択期間と評価期間が同じレースでは比べない


def test_fit_power_gamma_maximizes_the_winner_loglik_compare_reports():
    df = _sampled_market(1.10, seed=11, races=800)
    gamma = fit_power_gamma(df)

    def winner_loglik(g: float) -> float:
        out = compare(df, gamma=g, reps=10)
        return float(out.loc[out["方法"] == f"Power (γ={g:.2f})", "勝ち馬の対数尤度"].iloc[0])

    for other in (0.95, 1.00, 1.15):
        if other != gamma:
            assert winner_loglik(gamma) >= winner_loglik(other) - 1e-9  # fit が最大化する物差しと compare の報告が一致する


def test_fit_power_gamma_rejects_bad_odds():
    df = _sampled_market(1.05, seed=8, races=10)
    df.loc[df.index[0], "win_odds"] = np.nan
    with pytest.raises(ValueError):
        fit_power_gamma(df)


def test_markdown_report_includes_gamma_note():
    out = compare(_sampled_market(1.05, seed=3, races=60), gamma=1.20, reps=50)
    md = markdown_report(out, test_from="2024-01-01", test_to="2025-01-01",
                         fit_from="2019-01-01", fit_to="2024-01-01", gamma=1.20,
                         n_races=60, z_mean=0.05, z_failed=0, gamma_note="注意: γ が端に張り付いた。")
    assert "注意: γ が端に張り付いた。" in md


def test_main_wires_periods_without_leaking_boundary_day(tmp_path, capsys):
    """main の配線: 評価開始日ちょうどのレースは選択側でなく評価側に入る(load_market の境界)。"""
    import sqlite3

    from keibaai.devig_trials import main

    path = tmp_path / "keiba.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE races (race_id TEXT, date TEXT)")
    conn.execute("CREATE TABLE results (race_id TEXT, number INTEGER, finish INTEGER, win_odds REAL, status TEXT)")
    races = [("f1", "2029-12-01"), ("f2", "2029-12-15"), ("t1", "2030-01-01"), ("t2", "2030-01-15")]
    for rid, date in races:
        conn.execute("INSERT INTO races VALUES (?, ?)", (rid, date))
        for number, odds in enumerate([2.0, 4.0, 8.0], start=1):
            conn.execute("INSERT INTO results VALUES (?, ?, ?, ?, '')",
                         (rid, number, 1 if number == 1 else 2, odds))
    conn.commit()
    conn.close()
    main(["--db", str(path), "--fit-from", "2029-01-01", "--test-from", "2030-01-01", "--test-to", "2030-02-01"])
    out = capsys.readouterr().out
    assert "2030-01-01 〜 2030-02-01 の 2 レース" in out  # 境界日の t1 は評価側の2レースに数える
    assert "Power" in out


def test_markdown_report_survives_nan_z():
    out = compare(_sampled_market(1.05, seed=3, races=60), reps=50)
    md = markdown_report(out, test_from="2024-01-01", test_to="2025-01-01",
                         fit_from="2019-01-01", fit_to="2024-01-01", gamma=1.05,
                         n_races=60, z_mean=float("nan"), z_failed=60)
    assert "採用可否" in md  # z が全レース推定できなくても落ちない
