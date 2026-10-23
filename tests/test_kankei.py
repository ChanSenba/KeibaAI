import math

from keibaai import db, kankei
from keibaai.parse import ParsedRace, Race, Result


def _db(rows):
    """rows: (日付, 騎手, 調教師, 馬主, 着順, 状態) の出走。1行を1レースにする(架空のデータ)。"""
    conn = db.connect(":memory:")
    for i, (date, j, t, o, f, status) in enumerate(rows):
        rid = f"2026{i:08d}"
        db.save_race(conn, ParsedRace(Race(race_id=rid, date=date), [Result(
            race_id=rid, number=1, finish=f, status=status, jockey_id=j, jockey=f"騎手{j}",
            trainer_id=t, trainer=f"厩舎{t}", owner=o)]))
    return conn


def sample():
    rows = []
    rows += [("2026-05-01", "A", "X", "サンデーレーシング", 1, None)] * 6
    rows += [("2026-05-02", "A", "Y", "ビッグレッドファーム", 4, None)] * 4
    rows += [("2026-05-03", "B", "X", "山田太郎", 2, None)] * 14
    rows += [("2026-05-04", "B", "Y", "サンデーレーシング", 5, None)] * 2
    rows += [("2026-05-05", "A", "X", "サンデーレーシング", None, "取")] * 5  # 取消は数えない
    rows += [("2022-01-01", "A", "Y", "山田太郎", 1, None)] * 9  # 期間より前
    return _db(rows)


def test_two_way_shares():
    runs = kankei.load_runs(sample(), "2026-10-18", 3)
    assert len(runs) == 26
    rel = kankei.relations(runs, "jockey", "A")
    assert rel["n"] == 10 and rel["wins"] == 6
    x = rel["inner"].loc["X"]
    # A の 10 騎乗のうち X が 6(60%・1位)。X の 20 出走のうち A が 6(30%・B に次いで2位)
    assert x["n"] == 6 and math.isclose(x["share_center"], 0.6) and x["rank_center"] == 1
    assert math.isclose(x["share_partner"], 0.3) and x["rank_partner"] == 2 and not x["top"]
    y = rel["inner"].loc["Y"]
    assert math.isclose(y["share_partner"], 4 / 6) and y["rank_partner"] == 1 and y["top"]
    assert math.isclose(x["top3_rate"], 1.0) and y["wins"] == 0
    o = rel["outer"]
    assert o.loc["サンデーレーシング", "group"] == "ノーザンF系"
    assert o.loc["ビッグレッドファーム", "group"] == "マイネル系"
    assert set(map(tuple, rel["links"][["partner", "owner"]].values)) == {
        ("X", "サンデーレーシング"), ("Y", "ビッグレッドファーム")}


def test_trainer_center_and_people():
    runs = kankei.load_runs(sample(), "2026-10-18", 3)
    rel = kankei.relations(runs, "trainer", "X")
    assert rel["inner"].index.tolist() == ["B", "A"]
    assert rel["inner"].loc["B", "top"]  # B から見て X は 14/16 で1位
    # 30 騎乗に届かない騎手は一覧に出ない。調教師は出走があれば出る
    assert kankei.people(runs, "jockey", "2026-10-18").empty
    assert set(kankei.people(runs, "trainer", "2026-10-18")["id"]) == {"X", "Y"}
    assert kankei.breakdown(runs, rel, ("inner", "B"))["name"].tolist() == ["山田太郎"]


def test_svg_highlight():
    runs = kankei.load_runs(sample(), "2026-10-18", 3)
    rel = kankei.relations(runs, "jockey", "A")
    svg = kankei.diagram_svg(rel)
    assert svg.startswith("<svg") and "騎手A" in svg and 'kb-kk-dim"' not in svg and 'class="kb-kk-gold"' in svg
    assert 'kb-kk-dim"' in kankei.diagram_svg(rel, ("inner", "X"))
    assert kankei.diagram_img(rel, theme="dark").startswith('<img class="kb-kk" src="data:image/svg+xml;base64,')
