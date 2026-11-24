import numpy as np
import pandas as pd
import pytest

from keibaai.calib import (BAND_LABELS, Calibrator, calibrate_probs, fit_calibrator,
                           odds_band)
from keibaai.calib_trials import (_topk_probs, evaluate_year, markdown_report,
                                  walk_forward)
from keibaai.devig import market_probs, power, proportional

ODDS6 = np.array([1.5, 3.0, 6.0, 12.0, 30.0, 80.0])  # 帯は [0, 1, 2, 3, 4, 4]


def test_odds_band_boundaries():
    odds = np.array([1.1, 1.9, 2.0, 4.9, 5.0, 9.9, 10.0, 29.9, 30.0, 200.0])
    assert odds_band(odds).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert len(BAND_LABELS) == 5


def test_apply_race_matches_power_without_offsets():
    """帯オフセットが 0 なら、校正は devig の Power 法と一致する(温度だけの校正 = Power)。"""
    cal = Calibrator(a=1.1, band_offsets=np.zeros(5))
    q = proportional(ODDS6)
    assert cal.apply_race(q, ODDS6) == pytest.approx(power(ODDS6, 1.1))
    assert cal.apply_race(q, ODDS6).sum() == pytest.approx(1.0)
    ident = Calibrator(a=1.0, band_offsets=np.zeros(5))
    assert ident.apply_race(q, ODDS6) == pytest.approx(q)  # a=1・オフセット0 は無補正


def test_calibrate_probs_matches_power_and_marks_incomplete_nan():
    df = pd.DataFrame({"race_id": ["a"] * 3 + ["b"] * 2, "win_odds": [2.0, 4.0, 4.0, 2.0, None]})
    cal = Calibrator(a=1.1, band_offsets=np.zeros(5))
    p = calibrate_probs(df, cal)
    assert p.iloc[:3].tolist() == pytest.approx(market_probs(df, "power", gamma=1.1).iloc[:3].tolist())
    assert p.iloc[:3].sum() == pytest.approx(1.0)
    assert p.iloc[3:].isna().all()  # オッズの欠けたレースは丸ごと NaN(market_probs と同じ)


def test_calibrate_probs_keeps_favorite_on_top_with_band_offsets():
    df = pd.DataFrame({"race_id": ["a"] * 6, "win_odds": ODDS6})
    cal = Calibrator(a=1.05, band_offsets=np.array([0.05, 0.02, 0.0, -0.02, -0.05]))
    p = calibrate_probs(df, cal)
    assert p.sum() == pytest.approx(1.0)
    assert int(np.argmax(p.to_numpy())) == 0  # 順序が大きくは崩れない(1番人気が1番のまま)


def _flb_market(a_true: float, offsets_true, seed: int, races: int, year: int,
                prefix: str) -> pd.DataFrame:
    """勝ち馬を「市場確率に FLB をかけた確率」で抽選した、load_market と同じ形の df。

    FLB の真の形は校正そのもの: p ∝ exp(a・log q + c_帯)。着順は勝ち馬が1着、残りは馬番順。
    """
    rng = np.random.default_rng(seed)
    n = len(ODDS6)
    offsets_true = np.asarray(offsets_true, dtype=float)
    all_odds, all_win, all_finish = [], [], []
    for _ in range(races):
        odds = np.maximum(np.round(ODDS6 * np.exp(rng.normal(0, 0.15, size=n)), 1), 1.1)
        q = proportional(odds)  # オッズはレースごとに揺らす(帯の中で q が動く分、a と帯オフセットが別々に定まる)
        w = np.exp(a_true * np.log(q) + offsets_true[odds_band(odds)])
        winner = int(rng.choice(n, p=w / w.sum()))
        win = (np.arange(n) == winner).astype(int)
        finish = np.empty(n, dtype=int)
        finish[winner] = 1
        finish[[i for i in range(n) if i != winner]] = np.arange(2, n + 1)
        all_odds.append(odds)
        all_win.append(win)
        all_finish.append(finish)
    return pd.DataFrame(
        {"race_id": np.repeat([f"{prefix}{i}" for i in range(races)], n),
         "date": f"{year}-06-01", "number": np.tile(np.arange(1, n + 1), races),
         "finish": np.concatenate(all_finish), "win_odds": np.concatenate(all_odds),
         "win": np.concatenate(all_win)})


TRUE_OFFSETS = [0.10, 0.05, 0.0, -0.05, -0.10]


def _true_probs(df: pd.DataFrame, a_true: float, offsets_true) -> np.ndarray:
    """生成に使った真の確率を、df のオッズから組み直す(校正の形そのもの)。"""
    q = market_probs(df, "proportional").to_numpy(dtype=float)
    s = a_true * np.log(q) + np.asarray(offsets_true, dtype=float)[odds_band(df["win_odds"].to_numpy(dtype=float))]
    codes, _ = pd.factorize(df["race_id"].to_numpy())
    w = np.exp(s - s.max())
    return w / np.bincount(codes, weights=w)[codes]


def test_fit_calibrator_recovers_flb_probabilities():
    """a と帯オフセットは互いに似た効き方をして単独では定まりにくいので、パラメータでなく確率の復元で見る。"""
    df = _flb_market(1.15, TRUE_OFFSETS, seed=42, races=4000, year=2028, prefix="r")
    cal = fit_calibrator(df)
    p_cal = calibrate_probs(df, cal).to_numpy()
    p_true = _true_probs(df, 1.15, TRUE_OFFSETS)
    q = market_probs(df, "proportional").to_numpy()
    assert np.abs(p_cal - p_true).mean() < np.abs(q - p_true).mean() / 3  # 市場のずれの大部分を消せる
    assert cal.n_races == 4000


def test_fit_calibrator_no_flb_stays_near_market():
    df = _flb_market(1.0, [0.0] * 5, seed=3, races=4000, year=2028, prefix="r")
    cal = fit_calibrator(df)
    p_cal = calibrate_probs(df, cal).to_numpy()
    q = market_probs(df, "proportional").to_numpy()
    assert np.abs(p_cal - q).mean() < 0.005  # FLB がなければ、校正は市場確率をほぼ動かさない


def test_fit_calibrator_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_calibrator(pd.DataFrame({"race_id": [], "win": [], "win_odds": []}))
    df = _flb_market(1.1, TRUE_OFFSETS, seed=1, races=10, year=2028, prefix="r")
    two = df.copy()
    two.loc[two.index[1], "win"] = 1  # 同じレースに勝ち馬を2頭にする
    with pytest.raises(ValueError):
        fit_calibrator(two)
    bad = df.copy()
    bad.loc[bad.index[0], "win_odds"] = np.nan
    with pytest.raises(ValueError):
        fit_calibrator(bad)


def test_evaluate_year_fit_does_not_depend_on_test_period():
    fit = _flb_market(1.15, TRUE_OFFSETS, seed=1, races=500, year=2029, prefix="f")
    test = _flb_market(1.15, TRUE_OFFSETS, seed=2, races=100, year=2030, prefix="t")
    rigged = test.copy()
    rigged["win"] = 0
    rigged.loc[rigged.groupby("race_id")["win_odds"].idxmax(), "win"] = 1  # 勝ち馬を全部人気薄に入れ替える
    r1 = evaluate_year(fit, test, lam=10.0)
    r2 = evaluate_year(fit, rigged, lam=10.0)
    assert r1["cal"].a == r2["cal"].a  # 評価期間の結果をどう変えても、校正の学びは変わらない(漏れていない)
    assert r1["cal"].band_offsets == pytest.approx(r2["cal"].band_offsets)


def test_evaluate_year_henery_ignores_horses_without_finish():
    fit = _flb_market(1.15, TRUE_OFFSETS, seed=1, races=200, year=2029, prefix="f")
    test = _flb_market(1.15, TRUE_OFFSETS, seed=2, races=50, year=2030, prefix="t")
    test.loc[test.index[3], "finish"] = np.nan  # 1頭だけ着順なし(中止など)にする
    r = evaluate_year(fit, test)
    assert np.isfinite(r["hen2_raw_sum"]) and np.isfinite(r["hen3_cal_sum"])
    assert r["n_horses"] == len(test) - 1  # Henery の対象は着順のある馬だけ


def test_evaluate_year_rejects_overlapping_races():
    df = _flb_market(1.1, TRUE_OFFSETS, seed=6, races=50, year=2030, prefix="r")
    with pytest.raises(ValueError):
        evaluate_year(df, df)


def test_walk_forward_improves_on_flb_data():
    parts = [_flb_market(1.2, [0.15, 0.08, 0.0, -0.08, -0.15], seed=10 + y, races=1200,
                         year=y, prefix=f"y{y}_") for y in (2027, 2028, 2029, 2030)]
    df = pd.concat(parts, ignore_index=True)
    res = walk_forward(df, "2029-01-01", "2031-01-01", reps=200)
    assert [y["年"] for y in res["years"]] == [2029, 2030]
    assert res["pooled_diff"] < 0  # FLB のある市場では、校正で勝ち馬 LogLoss が下がる
    assert res["n_improved"] == 2
    assert res["ci"][0] <= res["pooled_diff"] <= res["ci"][1]
    assert res["wabs_cal"] < res["wabs_raw"]  # reliability も良くなる


def test_walk_forward_year_boundary_uses_only_past():
    """境界日(評価年の1月1日)のレースは学び側でなく評価側に入る。"""
    fit = _flb_market(1.1, TRUE_OFFSETS, seed=5, races=300, year=2029, prefix="f")
    fit["date"] = "2029-12-31"
    test = _flb_market(1.1, TRUE_OFFSETS, seed=6, races=30, year=2030, prefix="t")
    test["date"] = "2030-01-01"
    df = pd.concat([fit, test], ignore_index=True)
    res = walk_forward(df, "2030-01-01", "2031-01-01", reps=50)
    assert res["years"][0]["レース数"] == 30  # 評価は境界日の30レースだけ


def test_topk_probs_sum_to_k():
    p = proportional(ODDS6)
    assert _topk_probs(p, 2).sum() == pytest.approx(2.0)  # 2着以内には必ず2頭入る
    assert _topk_probs(p, 3).sum() == pytest.approx(3.0)
    assert int(np.argmax(_topk_probs(p, 2))) == 0


def test_markdown_report_has_criteria_and_tables():
    parts = [_flb_market(1.15, TRUE_OFFSETS, seed=20 + y, races=400, year=y, prefix=f"y{y}_")
             for y in (2028, 2029)]
    df = pd.concat(parts, ignore_index=True)
    res = walk_forward(df, "2029-01-01", "2030-01-01", reps=100)
    md = markdown_report(res, test_from="2029-01-01", test_to="2030-01-01",
                         fit_from="2028-01-01", lam=10.0)
    assert "市場の勝率の校正" in md and "| 2029 |" in md
    assert "reliability" in md and "2着以内" in md and "帯オフセット" in md
    assert "対象年の7割以上" in md and "2024年〜" in md and "0.0005" in md  # 事前に固定した採否基準
    assert "採用可否: (結果を見て記入)" in md
    assert "再フィットは" in md  # Henery の γ は再フィットしない旨


def test_main_wires_years_without_leaking_boundary_day(tmp_path, capsys):
    """main の配線: 評価開始日ちょうどのレースは学び側でなく評価側に入る。"""
    import sqlite3

    from keibaai.calib_trials import main

    path = tmp_path / "keiba.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE races (race_id TEXT, date TEXT)")
    conn.execute("CREATE TABLE results (race_id TEXT, number INTEGER, finish INTEGER, win_odds REAL, status TEXT)")
    races = [("f1", "2029-12-01"), ("f2", "2029-12-15"), ("t1", "2030-01-01"), ("t2", "2030-01-15")]
    for rid, date in races:
        conn.execute("INSERT INTO races VALUES (?, ?)", (rid, date))
        for number, odds in enumerate([2.0, 4.0, 8.0], start=1):
            conn.execute("INSERT INTO results VALUES (?, ?, ?, ?, '')",
                         (rid, number, number, odds))
    conn.commit()
    conn.close()
    main(["--db", str(path), "--fit-from", "2029-01-01", "--test-from", "2030-01-01", "--test-to", "2030-02-01"])
    out = capsys.readouterr().out
    assert "2030-01-01 〜 2030-02-01 の 2 レース" in out  # 境界日の t1 は評価側の2レースに数える
    assert "| 2030 |" in out
