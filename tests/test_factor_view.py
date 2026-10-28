import numpy as np
import pandas as pd
import pytest

from keibaai import factor_view as fv
from keibaai.features import load_results

from test_entries import _entries_for_future
from test_model import synthetic_db

DAY = "2030-01-05"


@pytest.fixture(scope="module")
def conn():
    c = synthetic_db(n_days=40)
    _entries_for_future(c, DAY)
    return c


def test_style_label():
    nan = float("nan")
    assert fv.style_label(nan, nan) == "不明"
    assert fv.style_label(0.06, 0.0) == "逃げ"
    assert fv.style_label(0.3, 0.0) == "先行"
    assert fv.style_label(0.6, 0.1) == "差し"
    assert fv.style_label(0.9, 0.1) == "追込"
    assert fv.style_label(0.9, 0.3) == "マクリ"
    assert fv.style_label(0.2, 0.5) == "先行"  # 前に付ける馬はマクリにしない


def test_rotation_label():
    nan = float("nan")
    assert fv.rotation_label(nan, 1) == "初出走"
    assert fv.rotation_label(7, 3) == "連闘"
    assert fv.rotation_label(120, 1) == "休み明け(17週ぶり)"
    assert fv.rotation_label(28, 2) == "叩き2走目" and fv.rotation_label(28, 3) == "叩き3走目"
    assert fv.rotation_label(28, 6) == "中4週"


def test_pace_forecast():
    s = lambda *x: pd.Series(list(x))
    assert fv.pace_forecast(s("逃げ", "逃げ", "差し", "差し")) == "ハイペース寄り"
    assert fv.pace_forecast(s("差し", "差し", "追込", "差し", "追込")) == "スロー寄り"
    assert fv.pace_forecast(s("逃げ", "差し", "差し", "追込", "差し")) == "平均的"
    assert fv.pace_forecast(s("不明", "不明")) == "不明"


def test_values_have_every_listed_column_and_one_row_per_horse(conn):
    v = fv.factor_values(conn, DAY)
    assert len(v) == 10 and v["number"].tolist() == list(range(1, 11))
    for s in fv.SECTIONS:
        for col, _ in s["cols"]:
            assert col in v, col


def test_records_match_hand_count(conn):
    v = fv.factor_values(conn, DAY)
    raw = load_results(conn)
    for _, row in v.iterrows():
        h = raw[raw["horse_id"] == f"h{row['number'] - 1}"]
        f = h["finish"]
        assert row["rec_ct"] == f"{(f == 1).sum()}-{(f == 2).sum()}-{(f == 3).sum()}-{(f > 3).sum()}"  # 全部「芝」
        near = h[(h["distance"] - 1600).abs() <= 200]["finish"]  # 今回は1600m
        assert row["rec_dist"] == f"{(near == 1).sum()}-{(near == 2).sum()}-{(near == 3).sum()}-{(near > 3).sum()}"
        recent = h.sort_values("date")["finish"].tail(5)[::-1]
        assert row["last5"] == "-".join(str(int(x)) for x in recent)


def test_jockey_rank_counts_this_years_wins_before_the_day():
    c = synthetic_db(n_days=40)
    _entries_for_future(c, "2025-12-28")  # 合成データの2025年の途中までの成績と同じ年
    v = fv.factor_values(c, "2025-12-28")
    raw = load_results(c)
    wins = raw[raw["date"].dt.year == 2025].assign(w=lambda d: d["finish"] == 1).groupby("jockey_id")["w"].sum()
    ranks = wins.rank(method="min", ascending=False)
    row = v.iloc[0]
    assert row["jockey_wins_year"] == wins.get("j0", 0) and row["jockey_rank"] == ranks.get("j0")


def test_pedigree_names_and_status_follow_what_is_collected(conn):
    st = {s["name"]: s for s in fv.section_status(conn)}
    assert st["血統"]["state"] == "missing" and st["調教"]["state"] == "missing"
    assert st["ラップ・ペース"]["state"] == "missing" and st["適性(着度数)"]["state"] == "ready"
    conn.execute("INSERT INTO pedigree (horse_id, sire_id, sire, damsire_id, damsire, sire_line) "
                 "VALUES ('h0', 's1', 'ハーツクライ', 'd1', 'Vindication', 'Halo系')")
    conn.commit()
    try:
        v = fv.factor_values(conn, DAY)
        first = v[v["number"] == 1].iloc[0]
        assert (first["sire"], first["damsire"], first["sire_line"]) == ("ハーツクライ", "Vindication", "Halo系")
        assert v[v["number"] == 2].iloc[0]["sire"] != v[v["number"] == 2].iloc[0]["sire"]  # 未取得の馬は空(NaN)
        assert {s["name"]: s for s in fv.section_status(conn)}["血統"]["state"] == "partial"
    finally:
        conn.execute("DELETE FROM pedigree")
        conn.commit()


def test_pedigree_status_counts_only_horses_in_results(conn):
    """血統の表に成績にいない馬が大量に入っていても、「取得済み」は成績の馬の数を超えない(分子 > 分母にならない)。"""
    n_horses = conn.execute("SELECT COUNT(DISTINCT horse_id) FROM results").fetchone()[0]
    conn.execute("INSERT INTO pedigree (horse_id) VALUES ('h0')")
    conn.executemany("INSERT INTO pedigree (horse_id) VALUES (?)", [(f"outsider{i}",) for i in range(n_horses + 5)])
    conn.commit()
    try:
        text = {s["name"]: s for s in fv.section_status(conn)}["血統"]["text"]
        assert text == f"1 / {n_horses:,} 頭ぶん取得済み"
    finally:
        conn.execute("DELETE FROM pedigree")
        conn.commit()


def test_training_values_and_status_follow_what_is_collected(conn):
    race_id = conn.execute("SELECT race_id FROM entry_races WHERE date = ?", (DAY,)).fetchone()[0]
    past = conn.execute("SELECT race_id FROM races ORDER BY date LIMIT 1").fetchone()[0]
    rows = [(race_id, 1, 0, "仕上良好", "A"), (race_id, 2, 0, "前走並み", "C"), (race_id, 3, 0, None, None)]
    conn.executemany("INSERT INTO training (race_id, number, seq, critic, rank) VALUES (?, ?, ?, ?, ?)", rows)
    conn.execute("INSERT INTO training_fetched VALUES (?, 3, '2030-01-04')", (past,))
    conn.commit()
    try:
        v = fv.factor_values(conn, DAY)
        r = v[v["race_id"] == race_id].set_index("number")
        assert (r.loc[1, "train_critic"], r.loc[1, "train_rank"]) == ("仕上良好", "A")
        # A=4, C=2 の平均 3 との差(評価のない馬は平均に入れない)
        assert (r.loc[1, "train_rank_vs_field"], r.loc[2, "train_rank_vs_field"]) == (1.0, -1.0)
        assert r.loc[3, "train_rank_vs_field"] != r.loc[3, "train_rank_vs_field"]
        assert {s["name"]: s for s in fv.section_status(conn)}["調教"]["state"] == "partial"
    finally:
        conn.execute("DELETE FROM training")
        conn.execute("DELETE FROM training_fetched")
        conn.commit()


def test_training_place_names():
    assert fv.training_place("slope", "1", None, None) == "坂路"
    assert fv.training_place("wood", "1", "2", "0") == "CW(右)"
    assert fv.training_place("wood", "0", "3", "1") == "ウッドD(左)"


def test_training_places_in_one_section(conn):
    race_id, number, horse = conn.execute(
        "SELECT race_id, number, horse_id FROM entries WHERE race_id IN "
        "(SELECT race_id FROM entry_races WHERE date = ?) ORDER BY race_id, number LIMIT 1", (DAY,)).fetchone()
    rows = [(horse, "2029-12-28", "0700", "wood", "1", "2", "0", 52.0),  # 一番時計
            (horse, "2030-01-02", "0700", "slope", "1", None, None, 53.5),  # 最後の追い切り
            (horse, "2030-01-04", "0700", "slope", "1", None, None, 64.0),  # 遅い(流し)ので数えない
            (horse, "2029-12-01", "0700", "wood", "1", "2", "0", 50.0)]  # 14日より前
    conn.executemany("INSERT INTO jv_training (horse_id, date, time, kind, center, course, turn, f4) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.execute("INSERT INTO training (race_id, number, seq, course, date, rider, load, partner) "
                 "VALUES (?, ?, 0, '栗坂', '2030-01-02', '助手', '一杯', '内ロングトールサリー馬也と併せ０秒２遅れ')", (race_id, number))
    conn.commit()
    try:
        v = fv.factor_values(conn, DAY)
        r = v[(v["race_id"] == race_id) & (v["number"] == number)].iloc[0]
        assert (r["train_center"], r["train_last_place"], r["train_best_place"]) == ("栗東", "坂路", "CW(右)")
        assert (r["train_day"], r["train_rider"], r["train_load"]) == ("2030-01-02", "助手", "一杯")
        assert "ロングトールサリー" in r["train_partner"]
        others = v[(v["race_id"] == race_id) & (v["number"] != number)]
        assert others["train_partner"].isna().all()  # 詳細が取れていない馬は空
        assert {s["name"]: s for s in fv.section_status(conn)}["調教"]["state"] == "ready"
        names = [s["name"] for s in fv.SECTIONS]
        assert names.count("調教") == 1 and not any("JRA-VAN" in n for n in names)
    finally:
        conn.execute("DELETE FROM jv_training")
        conn.execute("DELETE FROM training")
        conn.commit()

def test_no_entries_raises(conn):
    with pytest.raises(ValueError):
        fv.factor_values(conn, "2031-01-01")


def test_every_column_has_an_explanation():
    for s in fv.SECTIONS:
        for col, _ in s["cols"]:
            info, good = fv.COLUMN_INFO[col]
            assert info and good in (None, *fv.GOOD_LABEL), col


def test_top_rank_colors_the_best_five_in_the_good_direction():
    s = pd.Series([0.1, 0.5, 0.3, np.nan, 0.9, 0.2, 0.4, 0.0, 0.6, 0.7, 0.8, 0.05])
    hi = fv.top_rank(s, "high")
    assert hi.notna().sum() == 5 and hi[4] == 1 and set(hi.dropna().index) == {4, 10, 9, 8, 1}
    lo = fv.top_rank(s, "low")
    assert lo[7] == 1 and set(lo.dropna().index) == {7, 11, 0, 5, 2}
    assert fv.top_rank(pd.Series([-6.0, 2.0, 0.0, 10.0]), "near0").dropna().to_dict() == {2: 1, 1: 2}  # 4頭なら2頭まで
    assert fv.top_rank(s, None).isna().all()
    assert fv.top_rank(pd.Series([0.2] * 8), "high").isna().all()  # 差がなければ色なし


def test_top_rank_for_records_and_recent_finishes():
    rec = pd.Series(["0-0-0-5", "2-1-0-3", "1-2-0-1", "0-0-1-0", "—", "3-0-0-9"])
    r = fv.top_rank(rec, "record")
    assert r[5] == 1 and r[1] == 2 and r.notna().sum() == 2  # 比べられる4頭の半分まで
    last5 = pd.Series(["1-2-3", "8-9", "—", "4-4-4-4-4", "2-1"])
    assert fv.top_rank(last5, "last5").dropna().to_dict() == {4: 1, 0: 2}


def test_summary_columns_exist_and_are_explained(conn):
    v = fv.factor_values(conn, DAY)
    own = {c for s in fv.SECTIONS for c, _ in s["cols"]}
    for col, _ in fv.SUMMARY["cols"]:
        assert col in v and col in own and col in fv.COLUMN_INFO, col


def test_column_points_ranks_within_the_race():
    s = pd.Series([50.0, 52.0, np.nan, 51.0])
    p = fv.column_points(s, "low")  # 小さいほど良い
    assert p.tolist() == [10.0, 0.0, 5.0, 5.0]  # 欠けは5点
    assert (fv.column_points(pd.Series([1.0, 1.0, 1.0]), "high") == 5.0).all()  # 差がなければ全員5点


def test_score_race_sums_points_and_marks_the_top(conn):
    one = fv.factor_values(conn, DAY)
    sc = fv.score_race(one)
    assert sc["score"].between(0, 100).all() and sc["score_rank"].min() == 1
    assert sorted(sc.loc[sc["score_mark"] != "", "score_mark"]) == sorted(["◎", "○", "▲", "△", "△"])  # 10頭立て
    best = sc["score"].idxmax()
    assert sc.loc[best, "score_mark"] == "◎"
    assert not any(c in fv.SCORE_EXCLUDE for c in (k[3:] for k in sc if k.startswith("pt_")))
    assert len(fv.score_day(one)) == len(one)


def test_score_race_follows_a_hand_calculation():
    one = pd.DataFrame({"number": [1, 2, 3], "avg_finish_5": [2.0, 5.0, 3.0], "jockey_rank": [3, 1, 2]})
    sc = fv.score_race(one)
    # 着順は小さいほど良い: 1番10点・3番5点・2番0点 / リーディング順位は小さいほど良い: 2番10点・3番5点・1番0点
    assert sc["score"].round(2).tolist() == [50.0, 50.0, 50.0]


def test_front_edge_and_pace_fit_direction():
    nan = float("nan")
    assert fv.front_edge("スロー寄り", 2.0, 0.1) == 1.0 and fv.front_edge("ハイペース寄り", -3.0, -0.2) == -1.0
    assert fv.front_edge("平均的", nan, nan) == 0.0
    assert fv.pace_label(1.4) == "スロー" and fv.pace_label(-1.4) == "ハイペース" and fv.pace_label(0.2) == "平均的"
    assert fv.STYLE_FRONT["逃げ"] > fv.STYLE_FRONT["先行"] > fv.STYLE_FRONT["差し"] > fv.STYLE_FRONT["追込"]


def test_forecast_race_flow_text_and_favored_style(conn):
    one = fv.factor_values(conn, DAY)
    hist = pd.DataFrame({"ペース差": [-2.0, -1.5], "先行有利度": [-0.2, -0.1]})
    flow = fv.forecast_race_flow(one, hist)
    assert flow["favored"] == "差し・追込が有利" and "同コース直近2レース" in flow["text"]
    assert sum(len(v) for v in flow["lineup"].values()) == len(one)
    assert fv.forecast_race_flow(one, pd.DataFrame())["favored"]  # 履歴がなくても文章は出る


def test_course_history_lists_recent_days_before_the_date(conn):
    h = fv.course_history(conn, {"venue": "東京", "course_type": "芝", "distance": 1600}, DAY)
    assert not h.empty and (pd.to_datetime(h["日付"]) < pd.Timestamp(DAY)).all()
    assert h["日付"].is_monotonic_decreasing
    assert fv.course_history(conn, {"venue": None, "course_type": "芝", "distance": 1600}, DAY).empty


def test_model_factor_points_and_commentary():
    import pandas as pd
    from keibaai import factor_view as fv

    one = pd.DataFrame({
        "number": [1, 2, 3], "horse_name": ["A", "B", "C"], "popularity": [1, 2, 3], "win_odds": [2.0, 4.0, 9.0],
        "p_win_ability": [0.10, 0.40, 0.20], "p_win_market": [0.40, 0.35, 0.10],
        "ev_ability": [0.2, 1.6, 1.8], "ev_market": [0.8, 1.4, 0.9],
        "p_win_quality": [0.2, 0.5, 0.3], "p_win_training": [None, None, None],
        "mark": ["◎", "○", ""], "star": ["", "", ""]})
    pts = fv.model_factor_points(one)
    assert pts["pt_p_win_quality"].tolist() == [0.0, 10.0, 5.0]
    assert "pt_p_win_training" not in pts and pts["factor_total"].tolist() == [0.0, 100.0, 50.0]
    vals = pd.DataFrame({
        "number": [1, 2, 3], "last5": ["1-2-3", "4-5-6", "—"], "avg_finish_5": [2.0, 5.0, np.nan],
        "time_diff_5": [0.3, 0.9, np.nan], "l3f_rank_3": [0.2, 0.5, np.nan], "class_change": [0, -1, np.nan],
        "rec_ct": ["1-0-0-1", "0-0-0-3", "0-0-0-0"], "rec_dist": ["1-0-0-1", "0-0-0-3", "0-0-0-0"],
        "rec_dir": ["—", "—", "—"], "rec_venue": ["0-0-0-0"] * 3, "rec_ground": ["0-0-0-0"] * 3,
        "rotation": ["叩き2走目", "休み明け", "—"], "days_since_last": [28.0, 120.0, np.nan],
        "jockey": ["甲", "乙", "丙"], "jockey_rank": [1, 10, np.nan], "jockey_wins_year": [50, 5, np.nan],
        "jockey_ct_top3_rate": [0.4, 0.2, np.nan], "pair_n": [3, 0, np.nan], "pair_top3_rate": [0.5, np.nan, np.nan],
        "jockey_change": ["継続", "乗り替わり", "—"], "style": ["逃げ", "差し", "不明"], "pace_forecast": ["ハイペース"] * 3,
        "pace_fit": [-0.5, 0.5, 0.0], "inner_pos": [0.1, 0.5, 0.9], "sire": [None, None, None]})
    reviews = fv.race_reviews(one, pts, vals)
    assert len(reviews) == 3 and reviews[0]["name"] == "A"  # 印のない馬も載る
    assert "人気が勝率を大きく押し上げて" in reviews[0]["summary"] and reviews[0]["sub"].startswith("総合スコア0.0/100")
    assert "強みは馬の質10/10" in reviews[1]["summary"] and reviews[2]["sub"].startswith("総合スコア50.0/100")
    quality = reviews[1]["rows"][0]
    assert quality["name"] == "馬の質" and quality["pt"] == 10.0 and quality["grade"] == "高評価"
    assert "近5走は4-5-6" in quality["text"] and "格下げ" in quality["text"]
    assert [r["name"] for r in reviews[0]["rows"]] == ["馬の質"]  # 値のない要素は出さない
    assert fv.EVIDENCE["p_win_pedigree"](vals.set_index("number", drop=False), 1) == "血統のデータを取得していないため、評価に入れていません。"
    assert "初コンビ" not in fv.EVIDENCE["p_win_human"](vals.set_index("number", drop=False), 1)  # 3回騎乗済み
    assert fv.race_reviews(one, pts)[0]["rows"][0]["text"] == ""  # 要素の値がなければ点数だけ


def test_same_day_history_uses_only_earlier_rounds_of_the_same_day_venue_and_surface():
    from keibaai import db

    c = db.connect(":memory:")
    def race(rid, rnd, venue, ct, first, last):
        c.execute("INSERT INTO races (race_id, date, venue, round, course_type, distance, name, ground, pace_first3f, pace_last3f) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?)", (rid, "2030-02-02", venue, rnd, ct, 1600, f"{rnd}R", "良", first, last))
        for n in range(1, 7):  # 1〜3番(内)が上位、前に付けた馬(通過順1〜2位)が上位
            c.execute("INSERT INTO results (race_id, number, horse_id, finish, passing) VALUES (?,?,?,?,?)",
                      (rid, n, f"{rid}{n}", n, f"{n}-{n}"))
    race("a1", 1, "東京", "芝", 36.0, 34.0)   # 同じ日・場・芝で前のR → 入る
    race("a2", 2, "東京", "ダ", 36.0, 34.0)   # ダートは入らない
    race("a3", 3, "京都", "芝", 36.0, 34.0)   # 別の競馬場は入らない
    race("a4", 4, "東京", "芝", 34.0, 36.0)   # 同じ・または後のRは入らない
    c.commit()
    got = fv.same_day_history(c, {"venue": "東京", "course_type": "芝", "round": 4}, "2030-02-02")
    assert got["R"].tolist() == [1] and got["ペース"].tolist() == ["スロー"] and got["勝ち馬"].tolist() == [1]
    assert got["先行有利度"].iloc[0] > 0 and got["内有利度"].iloc[0] > 0
    assert "ここまで1レース" in fv.same_day_text("東京", "芝", got) and "目安" in fv.same_day_text("東京", "芝", got)
    assert fv.same_day_history(c, {"venue": "東京", "course_type": "芝", "round": 1}, "2030-02-02").empty


def test_training_text_says_partner_and_rider_or_that_they_are_missing():
    vals = pd.DataFrame({"number": [1, 2, 3], "train_last_place": ["坂路"] * 3, "jv_last_f4": [52.0, 53.0, 54.0],
                         "train_day": ["2030-01-02", "2030-01-02", None], "train_rider": ["助手", None, None],
                         "train_load": ["一杯", None, None], "train_partner": ["内Aと併せ０秒２遅れ", None, None]}
                        ).set_index("number", drop=False)
    one = fv.EVIDENCE["p_win_training"](vals, 1)
    assert "騎乗は助手・脚色は一杯" in one and "併せ馬は内Aと併せ０秒２遅れ" in one
    assert "騎乗は不明" in fv.EVIDENCE["p_win_training"](vals, 2) and "併せ馬の記載はなし(単走)" in fv.EVIDENCE["p_win_training"](vals, 2)
    assert "騎乗者と併せ馬は取得できていません" in fv.EVIDENCE["p_win_training"](vals, 3)
