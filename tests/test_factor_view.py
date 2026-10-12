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
