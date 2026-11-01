import datetime as dt

import pandas as pd

from keibaai import db, race_review, weekly
from keibaai.parse import ParsedRace, Race, Result


def _race(conn, rid, date, rows, **kw):
    """rows = [(number, finish, horse_id, jockey_id, popularity, passing, last3f), ...]"""
    race = Race(race_id=rid, name=kw.pop("name", "テスト特別"), date=date, venue="東京", round=1, course_type="芝",
                distance=kw.pop("distance", 2000), ground="良", **kw)
    n = len(rows)
    db.save_race(conn, ParsedRace(race=race, results=[
        Result(race_id=rid, number=num, finish=fin, horse_id=h, horse_name=f"馬{h}", jockey_id=j, jockey=f"騎手{j}",
               popularity=pop, win_odds=2.0 * pop, passing=pas, last3f=l3) for num, fin, h, j, pop, pas, l3 in rows]))
    return n


def _db():
    """6頭立て。直近1年の騎手の成績: J1 は毎回1着、J3 は毎回2着、J2 は毎回6着(3着内率 100%・100%・0%)。"""
    conn = db.connect(":memory:")
    for i in range(30):  # 直近1年の騎手の成績(30騎乗ずつ)
        rid = f"2026060101{i:02d}"
        _race(conn, rid, f"2026-0{1 + i % 9}-{10 + i % 15}", [
            (1, 1, f"x{i}a", "J1", 1, "1-1", 35.0), (2, 6, f"x{i}b", "J2", 6, "6-6", 36.0),
            (3, 2, f"x{i}c", "J3", 2, "2-2", 35.1), (4, 3, f"x{i}d", "J4", 3, "3-3", 35.2),
            (5, 4, f"x{i}e", "J5", 4, "4-4", 35.3), (6, 5, f"x{i}f", "J6", 5, "5-5", 35.4)])
    # 今回の出走馬の前走(中央): h1 は J1 で勝った。h2 は 2026-02 に走ったきり(このあと海外へ)
    _race(conn, "202605010101", "2026-02-01", [(1, 1, "h1", "J1", 1, "2-2", 34.0), (2, 2, "h2", "J3", 2, "1-1", 34.5),
                                               (3, 3, "h3", "J3", 3, "3-3", 34.6), (4, 4, "h4", "J4", 4, "4-4", 34.7),
                                               (5, 5, "h5", "J5", 5, "5-5", 34.8), (6, 6, "h6", "J6", 6, "6-6", 34.9)])
    conn.execute("INSERT INTO other_runs (race_id, number, horse_id, date, region, venue_code, course_type, distance, finish, name) "
                 "VALUES ('2026A6000005', 4, 'h2', '2026-07-25', '海外', 'A6', '芝', 2390, 7, 'キングジョージ')")
    # 今回(2026-11-01): 後ろの3頭が1〜3着の差し決着。1番人気 h1 は J1 → J2 の乗り替わりで5着、2番人気 h2 は海外帰りで6着
    _race(conn, "202605041111", "2026-11-01", [
        (1, 5, "h1", "J2", 1, "1-1", 35.5), (2, 6, "h2", "J3", 2, "2-2", 35.6), (3, 4, "h3", "J3", 3, "3-3", 35.0),
        (4, 1, "h4", "J4", 4, "6-6", 33.0), (5, 2, "h5", "J5", 5, "5-5", 33.5), (6, 3, "h6", "J6", 6, "4-4", 34.0)],
          pace_first3f=34.7, pace_last3f=34.0)
    return conn


def test_flow_says_closers_won_and_names_the_winner():
    conn = _db()
    v = race_review.race_view(conn, "202605041111")
    flow = "".join(v["flow"])
    assert "前半3F 34.7秒・後半3F 34.0秒の平均ペース" in flow
    assert "勝った馬h4(4番人気)は序盤6番手から、上がり33.0秒(1位)" in flow
    assert "差し・追い込みが決まった" in flow and "序盤3番手以内の3頭の最高は4着" in flow


def test_favorites_get_overseas_return_and_jockey_downgrade_as_candidates():
    conn = _db()
    v = race_review.race_view(conn, "202605041111")
    fav = {line.split(":")[0]: line for line in v["favorites"]}
    assert v["favorites"][0] == "1〜3番人気のうち3着以内は0頭。"
    one = fav["1番人気 馬h1"]
    assert "5着" in one and "敗因の候補" in one and "乗り替わり(騎手J1 → 騎手J2。直近1年の3着内率 100% → 0%、格下げ)" in one
    assert "39週ぶり" in one  # 中央の前走から
    two = fav["2番人気 馬h2"]
    assert "前走はイギリス(2026-07-25 キングジョージ 7着)から中14週の休み明け" in two
    assert "前に付けて、差し・追い込みの決着に巻き込まれた" in two
    trends = "".join(v["trends"])
    assert "海外帰りの1〜3番人気" in trends and "乗り替わり(騎手の格下げ)の1〜3番人気" in trends


def test_jockey_upgrade_is_not_a_reason_to_lose():
    conn = _db()
    race = {"date": "2026-11-01", "distance": 2000}
    ch = race_review.horse_changes(conn, {"horse_id": "h1", "jockey_id": "J3", "jockey": "騎手J3"}, race)
    same = [c for c in ch if "乗り替わり" in c[0]][0]
    assert "100% → 100%" in same[1] and "格" not in same[1] and same[2] is True  # 同格の乗り替わりは候補に残す
    ch = race_review.horse_changes(conn, {"horse_id": "h1", "jockey_id": "J1", "jockey": "騎手J1"}, race)
    assert not [c for c in ch if "乗り替わり" in c[0]]  # 同じ騎手
    ch = race_review.horse_changes(conn, {"horse_id": "h6", "jockey_id": "J1", "jockey": "騎手J1"}, race)
    up = [c for c in ch if "乗り替わり" in c[0]][0]
    assert "格上げ" in up[1] and up[2] is False


def test_rate_text_verdicts():
    assert "市場の見込みどおり" in race_review.rate_text("乗り替わり", (1000, 210, 200.0))
    assert "割り引く根拠になりうる" in race_review.rate_text("休み明け", (5000, 800, 1000.0))
    assert "例が少なく" in race_review.rate_text("海外帰り", (50, 10, 11.0))
    assert "過去の例がありません" in race_review.rate_text("地方帰り", (0, 0, 0.0))


def test_review_markdown_has_race_view_and_prediction_line():
    conn = _db()
    rows = [{"race_id": "202605041111", "date": pd.Timestamp("2026-11-01"), "venue": "東京", "round": 11,
             "race_name": "天皇賞", "number": n, "horse_name": f"馬h{n}", "jockey": "", "win_odds": None, "popularity": None,
             "p_win_ability": p, "p_win_market": None, "ev_ability": None, "ev_market": None}
            for n, p in zip(range(1, 7), [0.3, 0.25, 0.1, 0.2, 0.1, 0.05])]
    conn.execute("DELETE FROM results WHERE race_id = '202605041111'")  # 結果の前に予想を保存
    conn.execute("DELETE FROM races WHERE race_id = '202605041111'")
    weekly.save_predictions(conn, pd.DataFrame(rows), dt.datetime(2026, 10, 31, 20))
    _race(conn, "202605041111", "2026-11-01", [
        (1, 5, "h1", "J2", 1, "1-1", 35.5), (2, 6, "h2", "J3", 2, "2-2", 35.6), (3, 4, "h3", "J3", 3, "3-3", 35.0),
        (4, 1, "h4", "J4", 4, "6-6", 33.0), (5, 2, "h5", "J5", 5, "5-5", 33.5), (6, 3, "h6", "J6", 6, "4-4", 34.0)],
          pace_first3f=34.7, pace_last3f=34.0)
    r = weekly.review_day(conn, "2026-11-01")
    md = weekly.render_markdown(r)
    assert "  - 流れ: 前半3F" in md and "  - 人気馬:" in md and "  - 過去の傾向:" in md
    assert "  - 予想: ◎馬h1は5着(1番人気)。勝った馬h4は予想3位(▲)で、人気(4番人気)より1つ上に見ていた。" in md
    assert weekly.review_day(conn, "2026-11-01", views=False).views is None


def test_each_horse_gets_win_and_loss_tags():
    conn = _db()
    v = race_review.race_view(conn, "202605041111")
    lines = {line.split(" ")[1].split("(")[0].rstrip(":"): line for line in v["horses"]}
    # 差し決着(前の3頭が4〜6着)。勝った h4 は後ろから → 展開に恵まれた。前で負けた h1 は展開が不利
    assert "勝因の候補 展開に恵まれた" in lines["馬h4"]
    assert "敗因の候補 展開が不利" in lines["馬h1"]
    assert v["horses"][0].startswith("1着 馬h4")


def test_next_run_rates_count_the_tag_of_the_previous_race():
    conn = _db()
    rates = race_review.next_run_rates(conn, "2027-01-01")
    assert set(race_review.TAGS) <= set(rates) and rates["全体"][0] > 0
    table = race_review.next_run_table(rates)
    assert {"前走の印", "実績÷見込み"} <= set(table.columns)
