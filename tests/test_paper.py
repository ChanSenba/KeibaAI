"""仕様 8-25 のペーパートレード(paper.py): 確率の作り方・閾値・記録のガード・精算。"""
import itertools

import numpy as np
import pandas as pd
import pytest

from keibaai import db, jvodds, paper

DAY = "2030-01-05"
RID = "203001050101"
WIN = {(i,): 4.0 for i in range(1, 5)}  # 4頭とも単勝4.0倍 → 勝率は均等(0.25)


def _odds(trio=5.0, trifecta=30.0, exacta=15.0):
    return {"単勝": dict(WIN),
            "馬単": {c: exacta for c in itertools.permutations(range(1, 5), 2)},
            "三連複": {c: trio for c in itertools.combinations(range(1, 5), 3)},
            "三連単": {c: trifecta for c in itertools.permutations(range(1, 5), 3)}}


def test_bets_for_race_matches_hand_calc():
    """均等な勝率なら並びの確率は解析的に決まる。ヘネリーのべき乗は均等な勝率では効かない。

    並び(a,b,c) = 1/4 × 1/3 × 1/2 = 1/24。三連単はそのまま、馬単は3着の分を足して 1/12、三連複は6通りの和で 1/4。
    """
    bets = paper.bets_for_race([1, 2, 3, 4], [4.0] * 4, _odds())
    by_kind = {}
    for b in bets:
        by_kind.setdefault(b.kind, []).append(b)
    assert len(by_kind["三連単"]) == 24 and len(by_kind["三連複"]) == 4
    assert all(abs(b.prob - 1 / 24) < 1e-12 and abs(b.ev - 30 / 24) < 1e-9 for b in by_kind["三連単"])
    assert all(abs(b.prob - 0.25) < 1e-12 and abs(b.ev - 1.25) < 1e-9 for b in by_kind["三連複"])
    # 馬単は期待値 15/12 = 1.25 でも、的中確率 1/12 ≈ 8.3% が下限 10% に届かず落ちる
    assert "馬単" not in by_kind


def test_bets_for_race_thresholds():
    # 期待値ちょうど 1.2(三連複のオッズ 4.8 = 1.2 / 0.25)は残り、わずかに下は落ちる
    assert paper.bets_for_race([1, 2, 3, 4], [4.0] * 4, _odds(trio=4.8))
    assert not [b for b in paper.bets_for_race([1, 2, 3, 4], [4.0] * 4, _odds(trio=4.79)) if b.kind == "三連複"]
    # 三連単は的中確率 3% が下限: 8頭均等だと並びの確率は 1/336 ≈ 0.3% で、期待値が大きくても落ちる
    odds8 = {"単勝": {(i,): 8.0 for i in range(1, 9)},
             "三連単": {c: 500.0 for c in itertools.permutations(range(1, 9), 3)}}
    assert not [b for b in paper.bets_for_race(list(range(1, 9)), [8.0] * 8, odds8) if b.kind == "三連単"]


def _pred(conn, race_id, saved_at, after_result=0):
    cols = ("race_id, number, saved_at, date, venue, round, race_name, horse_name, jockey, win_odds, popularity, "
            "mark, star, p_win_ability, p_win_market, ev_ability, ev_market, after_result")
    rows = [(race_id, i, saved_at, DAY, "東京", 1, "テスト", f"馬{i}", "騎手", 4.0, i, "", None, 0.25, None, None, None,
             after_result) for i in range(1, 5)]
    conn.executemany(f"INSERT INTO predictions ({cols}) VALUES ({','.join('?' * 18)})", rows)
    conn.commit()


def _snap(conn, race_id, stamp, **kw):
    db.save_bet_odds_snapshot(conn, race_id, _odds(**kw), stamp)


def test_record_day_and_idempotent():
    conn = db.connect(":memory:")
    _pred(conn, RID, "2030-01-04T21:05:00")
    _snap(conn, RID, "2030-01-04T21:00:00")
    assert paper.record_day(conn, DAY) == 28  # 三連単24 + 三連複4(馬単は確率の下限で落ちる)
    assert paper.record_day(conn, DAY) == 0  # 同じ買い目は二重に記録しない
    rows = conn.execute("SELECT kind, combo, odds, prob, stake, snapped_at, payout FROM paper_bets ORDER BY kind, combo").fetchall()
    assert len(rows) == 28
    tri = [r for r in rows if r[0] == "三連単"]
    assert tri[0][1] == "01-02-03" and tri[0][2] == 30.0 and abs(tri[0][3] - 1 / 24) < 1e-12
    assert all(r[4] == 100 and r[5] == "2030-01-04T21:00:00" and r[6] is None for r in rows)


def test_record_day_guards():
    conn = db.connect(":memory:")
    _pred(conn, "r-after", "2030-01-04T21:05:00", after_result=1)  # 結果のあとに保存した予想
    _snap(conn, "r-after", "2030-01-04T21:00:00")
    _pred(conn, "r-late", "2030-01-06T10:00:00", after_result=None)  # 開催日より後に保存(列を足す前の保存の形)
    _snap(conn, "r-late", "2030-01-04T21:00:00")
    _pred(conn, "r-future-snap", "2030-01-04T21:05:00")
    _snap(conn, "r-future-snap", "2030-01-05T08:00:00")  # 保存より後のスナップショットしかない
    assert paper.record_day(conn, DAY) == 0
    assert conn.execute("SELECT COUNT(*) FROM paper_bets").fetchone()[0] == 0


def test_record_day_does_not_add_bets_from_a_later_snapshot():
    """記録したあとに別の時点のスナップショットで再実行しても、買い目を足さない(2時点の和集合にしない)。"""
    conn = db.connect(":memory:")
    _pred(conn, RID, "2030-01-04T21:05:00")
    _snap(conn, RID, "2030-01-04T21:00:00")
    assert paper.record_day(conn, DAY) == 28
    # 後から保存し直した予想と新しいスナップショット(三連複のオッズが下がって買い目が増える形)があっても、記録は変えない
    conn.execute("UPDATE predictions SET saved_at = '2030-01-05T07:05:00' WHERE race_id = ?", (RID,))
    conn.commit()
    _snap(conn, RID, "2030-01-05T07:00:00", trio=4.8)
    assert paper.record_day(conn, DAY) == 0
    assert conn.execute("SELECT COUNT(*) FROM paper_bets").fetchone()[0] == 28
    assert conn.execute("SELECT DISTINCT snapped_at FROM paper_bets").fetchall() == [("2030-01-04T21:00:00",)]


def test_record_day_post_time_guards():
    """発走時刻を過ぎた保存と、発走後のスナップショットは使わない(races.start_time で判定、live と同じ流儀)。"""
    conn = db.connect(":memory:")
    conn.execute("INSERT INTO races (race_id, date, start_time) VALUES (?, ?, ?)", (RID, DAY, "10:00"))
    conn.execute("INSERT INTO races (race_id, date, start_time) VALUES (?, ?, ?)", ("r-post-snap", DAY, "10:00"))
    conn.commit()
    _pred(conn, RID, "2030-01-05T10:30:00")  # 発走(10:00)より後に保存した予想
    _snap(conn, RID, "2030-01-05T09:00:00")
    _pred(conn, "r-post-snap", "2030-01-05T09:30:00")  # 保存は発走前
    _snap(conn, "r-post-snap", "2030-01-05T10:30:00")  # スナップショットは発走後だけ
    assert paper.record_day(conn, DAY) == 0
    assert conn.execute("SELECT COUNT(*) FROM paper_bets").fetchone()[0] == 0


def test_record_day_uses_snapshot_at_or_before_save():
    conn = db.connect(":memory:")
    _pred(conn, RID, "2030-01-04T21:05:00")
    _snap(conn, RID, "2030-01-04T20:00:00", trifecta=30.0)  # 保存以前: こちらを使う
    _snap(conn, RID, "2030-01-05T08:00:00", trifecta=99.0)  # 保存より後: 使わない
    assert paper.record_day(conn, DAY) == 28
    got = conn.execute("SELECT DISTINCT odds, snapped_at FROM paper_bets WHERE kind = '三連単'").fetchall()
    assert got == [(30.0, "2030-01-04T20:00:00")]


def _settled_conn(tmp_path):
    conn = db.connect(tmp_path / "keiba.db")
    _pred(conn, RID, "2030-01-04T21:05:00")
    _pred(conn, "r-nopay", "2030-01-04T21:05:00")
    _snap(conn, RID, "2030-01-04T21:00:00")
    _snap(conn, "r-nopay", "2030-01-04T21:00:00")
    assert paper.record_day(conn, DAY) == 56
    return conn


def test_settle_and_summary(tmp_path):
    conn = _settled_conn(tmp_path)
    oconn = jvodds.connect(tmp_path / "jravan_odds.db")
    # 払戻は当たった組だけ入る。1-2-3 の三連単が 3,000 円(100円あたり)
    oconn.execute("INSERT INTO payouts VALUES (?,?,?,?,?)", (RID, "trifecta", "01-02-03", 3000, 1))
    oconn.commit()
    # 三連単の払戻だけ入った段階では三連単だけ精算し、三連複は外れで確定させない。払戻のない r-nopay は未精算のまま
    assert paper.settle(conn) == 24
    assert conn.execute("SELECT payout FROM paper_bets WHERE race_id = ? AND combo = '01-02-03'", (RID,)).fetchone()[0] == 3000
    assert conn.execute("SELECT COUNT(*) FROM paper_bets WHERE payout = 0").fetchone()[0] == 23
    assert conn.execute("SELECT COUNT(*) FROM paper_bets WHERE payout IS NULL").fetchone()[0] == 32
    assert paper.settle(conn) == 0  # 精算済みはやり直さない
    # 三連複の払戻が入ったら、三連複も精算する(1-2-3 が 500 円)
    oconn.execute("INSERT INTO payouts VALUES (?,?,?,?,?)", (RID, "trio", "01-02-03", 500, 1))
    oconn.commit()
    assert paper.settle(conn) == 4
    s = paper.summary(conn).set_index("券種")
    assert s.loc["三連単", "点数"] == 24 and s.loc["三連単", "的中"] == 1
    assert s.loc["三連単", "払戻(円)"] == 3000 and s.loc["三連単", "回収率(%)"] == pytest.approx(125.0)
    assert s.loc["三連複", "的中"] == 1 and s.loc["三連複", "回収率(%)"] == pytest.approx(125.0)
    assert np.isnan(s.loc["三連単", "区間下限(%)"])  # 開催日が1日だけでは区間は出ない
    t = paper.trend(conn)
    assert list(t["累計点数"]) == [28] and t["累計回収率(%)"].iloc[0] == pytest.approx(3500 / 2800 * 100)


def test_day_bootstrap_roi_uses_days():
    bets = pd.DataFrame({"date": ["d1"] * 2 + ["d2"] * 2, "stake": [100] * 4, "payout": [0, 500, 0, 0]})
    lo, hi = paper.day_bootstrap_roi(bets)
    assert lo <= 125.0 <= hi  # 全体の回収率 125% は区間の中
    one = pd.DataFrame({"date": ["d1"], "stake": [100], "payout": [300]})
    assert all(np.isnan(x) for x in paper.day_bootstrap_roi(one))
