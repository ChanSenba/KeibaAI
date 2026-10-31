import numpy as np
import pandas as pd

from keibaai import bettypes, jvodds


def _db(tmp_path):
    conn = jvodds.connect(tmp_path / "odds.db")
    # 4頭立てのレース R1。単勝オッズ 1番 2.0倍・2番 4.0倍・3番 8.0倍・4番 16.0倍。勝ったのは2番(払戻 400円)
    conn.executemany("INSERT INTO odds_win VALUES ('R1', ?, ?, NULL)", [(1, 2.0), (2, 4.0), (3, 8.0), (4, 16.0)])
    conn.execute("INSERT INTO payouts VALUES ('R1', 'win', '02', 400, 2)")
    # 馬連 1-2 が 5.0倍で、1-2 で決まった(払戻 500円)
    conn.executemany("INSERT INTO odds_quinella VALUES ('R1', ?, ?, ?, NULL)", [(1, 2, 5.0), (1, 3, 9.0)])
    conn.execute("INSERT INTO payouts VALUES ('R1', 'quinella', '01-02', 500, 1)")
    conn.commit()
    return conn


def _test():
    return pd.DataFrame({"race_id": "R1", "number": [1, 2, 3, 4], "p_win": [0.5, 0.25, 0.125, 0.125]})


def test_evaluate_kind_win_ev_and_payout(tmp_path):
    b = bettypes.evaluate_kind(_test(), _db(tmp_path), "win")
    row = b.set_index(b["combo"].map(lambda c: c[0]))
    assert np.isclose(row.loc[1, "ev"], 0.5 * 2.0) and np.isclose(row.loc[2, "ev"], 0.25 * 4.0)
    assert row.loc[2, "payout"] == 400 and row.loc[1, "payout"] == 0  # 当たったのは2番だけ


def test_evaluate_kind_quinella_uses_harville_prob(tmp_path):
    b = bettypes.evaluate_kind(_test(), _db(tmp_path), "quinella", g2=1.0, g3=1.0)  # 補正なし(ハーヴィルそのまま)
    p = [0.5, 0.25, 0.125, 0.125]
    expect = p[0] * p[1] / (1 - p[0]) + p[1] * p[0] / (1 - p[1])  # 1-2 または 2-1 の順
    r = b[b["combo"] == (1, 2)].iloc[0]
    assert np.isclose(r["prob"], expect) and r["payout"] == 500 and np.isclose(r["ev"], expect * 5.0)


def test_roi_table_counts_only_selected_bets_and_all_bets_row(tmp_path):
    b = bettypes.evaluate_kind(_test(), _db(tmp_path), "win")
    t = bettypes.roi_table(b, ev_mins=(0.99,), min_hit=0.0, n_boot=50).set_index("rule")
    assert t.loc["全買い目", "n_bets"] == 4 and np.isclose(t.loc["全買い目", "roi"], 400 / 400 * 100)
    # 期待値は 1番・2番・3番が 1.0、4番が 2.0(浮動小数の誤差を見て 0.99 以上にする)。当たりの2番だけ払戻 400
    assert t.loc["期待値≧0.99", "n_bets"] == 4 and np.isclose(t.loc["期待値≧0.99", "roi"], 100.0)


def test_races_without_odds_or_payouts_are_skipped(tmp_path):
    conn = _db(tmp_path)
    extra = pd.DataFrame({"race_id": "R2", "number": [1, 2, 3], "p_win": [0.5, 0.3, 0.2]})
    b = bettypes.evaluate_kind(pd.concat([_test(), extra]), conn, "win")
    assert set(b["race_id"]) == {"R1"}
