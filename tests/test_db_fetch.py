from pathlib import Path

from keibaai import db
from keibaai.fetch import Fetcher, extract_jra_race_ids
from keibaai.parse import parse_race

HTML = (Path(__file__).parent / "fixtures" / "race_sample.html").read_text(encoding="utf-8")


def test_save_is_idempotent():
    conn = db.connect(":memory:")
    parsed = parse_race(HTML, "202405050812")
    db.save_race(conn, parsed)
    db.save_race(conn, parsed)
    assert conn.execute("SELECT COUNT(*) FROM races").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 2
    assert db.has_race(conn, "202405050812") and not db.has_race(conn, "x")


def test_extract_only_jra_ids():
    html = 'race_id=202405050812 race_id=202444100101 race_id=202405050812 race_id=202406010101'
    assert extract_jra_race_ids(html) == ["202405050812", "202406010101"]


class FakeResp:
    def __init__(self, status=200, text="<html/>"):
        self.status_code, self.text, self.encoding = status, text, None

    def raise_for_status(self):
        assert self.status_code < 400


class FakeSession:
    def __init__(self, resps):
        self.headers, self.resps, self.calls = {}, list(resps), 0

    def get(self, url, timeout):
        self.calls += 1
        return self.resps.pop(0)


def test_cache_prevents_refetch_and_rate_limit(tmp_path):
    t = {"now": 0.0}
    sleeps = []
    sess = FakeSession([FakeResp(), FakeResp()])
    f = Fetcher(tmp_path, session=sess, sleep=lambda s: (sleeps.append(s), t.update(now=t["now"] + s)),
                clock=lambda: t["now"])
    f.race_html("202405050812")
    f.race_html("202405050812")  # キャッシュ
    assert sess.calls == 1
    f.race_html("202405050813")
    assert sess.calls == 2 and sleeps and sleeps[0] >= 3.0


def test_backoff_on_429(tmp_path):
    sess = FakeSession([FakeResp(429), FakeResp(200, "ok")])
    sleeps = []
    f = Fetcher(tmp_path, session=sess, sleep=sleeps.append, clock=lambda: 0.0)
    assert f.race_html("202405050812") == "ok"
    assert 30 in sleeps
