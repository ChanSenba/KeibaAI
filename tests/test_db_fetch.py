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
    def __init__(self, status=200, text="<table class=\"race_table_01\"></table>"):
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
    sess = FakeSession([FakeResp(429), FakeResp(200, "ok race_table_01")])
    sleeps = []
    f = Fetcher(tmp_path, session=sess, sleep=sleeps.append, clock=lambda: 0.0)
    assert f.race_html("202405050812") == "ok race_table_01"
    assert 30 in sleeps


def test_default_requests_user_agent_is_replaced(tmp_path):
    import requests

    f = Fetcher(tmp_path, session=requests.Session())
    assert "python-requests" not in f.session.headers["User-Agent"]


def test_page_without_result_table_is_not_cached(tmp_path):
    import pytest

    from keibaai.fetch import PageUnavailable

    f = Fetcher(tmp_path, session=FakeSession([FakeResp(200, "<html>blocked</html>")]),
                sleep=lambda s: None, clock=lambda: 0.0)
    with pytest.raises(PageUnavailable):
        f.race_html("202605040301")
    assert not (tmp_path / "race" / "202605040301.html").exists()


def test_collect_day_skips_bad_race_and_aborts_on_streak(tmp_path):
    import pytest

    from keibaai import cli
    from keibaai.fetch import PageUnavailable

    class FakeFetcher:
        def __init__(self, ids, bad):
            self.ids, self.bad, self.discarded = ids, set(bad), []

        def race_ids_on(self, date):
            return self.ids

        def race_html(self, race_id):
            if race_id in self.bad:
                raise PageUnavailable(race_id)
            return HTML

        def discard(self, race_id):
            self.discarded.append(race_id)

    conn = db.connect(":memory:")
    ff = FakeFetcher(["202405050811", "202405050801", "202405050812"], bad=["202405050801"])
    assert cli.collect_day(conn, ff, "20241124", []) == 2  # 1件飛ばして続行
    assert ff.discarded == ["202405050801"]

    ids = [f"20240505080{i}" for i in range(1, 7)]
    with pytest.raises(RuntimeError):
        cli.collect_day(conn, FakeFetcher(ids, bad=ids), "20241124", [])


def test_reparse_cache_updates_db_and_skips_bad(tmp_path):
    from keibaai import cli

    (tmp_path / "race").mkdir()
    (tmp_path / "race" / "202405050812.html").write_text(HTML, encoding="utf-8")
    (tmp_path / "race" / "202605040301.html").write_text("<html></html>", encoding="utf-8")
    conn = db.connect(":memory:")
    assert cli.reparse_cache(conn, tmp_path) == (1, 1)
    assert db.has_race(conn, "202405050812")


def test_unpublished_day_is_skipped_without_counting_failures(tmp_path):
    from keibaai import cli
    from keibaai.fetch import ResultNotPublished

    class FakeFetcher:
        def race_ids_on(self, date):
            return [f"20260505040{i}" for i in range(1, 13)]

        def race_html(self, race_id):
            raise ResultNotPublished(race_id)

        def discard(self, race_id):
            pass

    failures = []
    assert cli.collect_day(db.connect(":memory:"), FakeFetcher(), "20261010", failures) == 0
    assert failures == []  # 連続失敗に数えない


def test_1970_template_page_is_detected(tmp_path):
    import pytest

    from keibaai.fetch import ResultNotPublished

    f = Fetcher(tmp_path, session=FakeSession([FakeResp(200, "<html>1970年1月01日</html>")]),
                sleep=lambda s: None, clock=lambda: 0.0)
    with pytest.raises(ResultNotPublished):
        f.race_html("202605040301")
