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
