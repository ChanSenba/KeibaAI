import pandas as pd
import pytest

from keibaai import past_races as pr

from test_entries import _entries_for_future
from test_model import synthetic_db

DAY = "2030-01-05"


@pytest.fixture(scope="module")
def conn():
    c = synthetic_db(n_days=40)
    _entries_for_future(c, DAY)
    return c


def test_every_entry_has_verdict_and_comments(conn):
    horses = pr.horse_past_races(conn, "202609990101", limit=4)
    assert len(horses) == 10
    h = horses[0]
    assert h["verdict"] and len(h["runs"]) <= 4
    assert {"見解", "kind", "日付"} <= set(h["runs"].columns)
    assert h["runs"]["見解"].str.len().gt(0).all()
    assert (h["runs"]["日付"] < DAY).all()


def test_unknown_race_raises(conn):
    with pytest.raises(ValueError):
        pr.horse_past_races(conn, "nope")


def test_outlook_ranks_field_from_past_ratings(conn):
    horses = pr.horse_past_races(conn, "202609990101", limit=5)
    rated = [h for h in horses if h["rating"] is not None]
    assert sorted(h["rank"] for h in rated) == list(range(1, len(rated) + 1))
    best = min(rated, key=lambda h: h["rank"])
    assert best["outlook"] == "最上位評価" and best["rating"] == max(h["rating"] for h in rated)
    assert "今回の見込みは" in best["verdict"]
    runs = best["runs"]
    assert abs(runs["重み"].sum() - 1) < 1e-9
    assert runs["評価"].between(0, 100).all()


def test_head_to_head_is_symmetric(conn):
    horses = pr.horse_past_races(conn, "202609990101", limit=5)
    wins = sum(int(h["h2h"].split("勝")[0]) for h in horses if h["h2h"])
    losses = sum(int(h["h2h"].split("勝")[1].split("敗")[0]) for h in horses if h["h2h"])
    assert wins == losses > 0


def test_close_condition_weighs_more():
    target = {"course_type": "芝", "distance": 1600, "venue": "東京", "direction": "左", "ground": "良"}
    base = {"finish": 1, "field_size": 10, "time_gap": 0.0, "l3f_rank": 1, "status": None, "popularity": 1,
            "date": "2029-12-01", "race_name": "x", "venue": "東京", "ground": "良", "passing": "", "last3f": 34.0,
            "jockey": "j", "direction": "左"}
    near = pd.Series({**base, "course_type": "芝", "distance": 1600})
    far = pd.Series({**base, "course_type": "ダ", "distance": 2400})
    frame = pr._horse_runs(pd.DataFrame([far, near]), target)
    assert frame.loc[1, "重み"] > frame.loc[0, "重み"] * 1.5
    assert pr.run_rating(near) == 100.0


def test_same_distance_elsewhere_is_not_same_course():
    target = {"course_type": "芝", "distance": 1600, "venue": "東京", "direction": "左", "ground": "良", "course": None,
              "date": "2030-01-05"}
    base = {"finish": 1, "field_size": 10, "time_gap": 0.0, "l3f_rank": 1, "status": None, "popularity": 1,
            "date": "2029-12-01", "race_name": "x", "ground": "良", "passing": "", "last3f": 34.0,
            "jockey": "j", "course_type": "芝", "distance": 1600, "course": None}
    course = pd.Series({**base, "venue": "東京", "direction": "左"})
    dist = pd.Series({**base, "venue": "中京", "direction": "左"})
    frame = pr._horse_runs(pd.DataFrame([dist, course]), target)
    assert frame["kind"].tolist() == ["同じ距離", "同コース"]
    assert frame.loc[1, "重み"] > frame.loc[0, "重み"]
    assert "同じ距離での経験" in frame.loc[0, "見解"] and "同コースではない" in frame.loc[0, "見解"]
    assert "同コース" in frame.loc[1, "見解"]
    plus, _ = pr._cautions(frame.iloc[[0]], target)
    assert plus == ["同じ距離(芝1600m)で好走歴あり(同コースではない)"]
