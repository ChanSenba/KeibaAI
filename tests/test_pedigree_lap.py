from pathlib import Path

import numpy as np
import pytest

from keibaai import db
from keibaai.features import ARTICLE_FEATURES, LAP_FEATURES, PEDIGREE_FEATURES, build_features, load_results
from keibaai.parse import ParsedRace, Race, Result, parse_race
from keibaai.pedigree import collect_pedigrees, parse_pedigree, pending_horse_ids, save_pedigree

from test_model import synthetic_db

SAMPLE = (Path(__file__).parent / "data" / "blood_table_sample.html").read_text(encoding="utf-8")

LAP_HTML = """<html><body><dl class="racedata"><dd><h1>ジャパンC</h1><p>芝右2400m / 天候:晴 / 芝:良 / 発走:15:40</p></dd></dl>
<p class="smalltxt">2024年11月24日 5回東京8日目</p>
<table class="race_table_01"><tr><th>着順</th><th>馬番</th></tr><tr><td>1</td><td>1</td></tr></table>
<table width="100%" class="result_table_02" summary="ラップタイム"><caption>ラップタイム</caption>
<tr><th>ラップ</th><td class="race_lap_cell">12.7 - 11.4 - 13.0 - 12.9 - 12.2 - 12.3 - 12.5 - 12.6 - 12.5 - 11.5 - 10.8 - 11.1</td></tr>
<tr><th>ペース</th><td class="race_lap_cell">12.7 - 24.1 - 37.1 - 50.0 - 62.2 - 74.5 - 87.0 - 99.6 - 112.1 - 123.6 - 134.4 - 145.5&nbsp;(37.1-33.4)</td></tr>
</table></body></html>"""


def test_parse_lap_table():
    r = parse_race(LAP_HTML, "202405050812").race
    assert r.lap_times.startswith("12.7-11.4-13.0") and r.lap_times.endswith("10.8-11.1")
    assert (r.pace_first3f, r.pace_last3f) == (37.1, 33.4)


def test_lap_falls_back_to_summing_when_no_parentheses():
    html = LAP_HTML.replace("&nbsp;(37.1-33.4)", "")
    r = parse_race(html, "202405050812").race
    assert (r.pace_first3f, r.pace_last3f) == (37.1, 33.4)


def test_race_without_lap_table_still_parses():
    html = LAP_HTML.split('<table width="100%"')[0] + "</body></html>"
    r = parse_race(html, "202405050812").race
    assert r.lap_times is None and r.pace_first3f is None


def test_parse_pedigree():
    p = parse_pedigree(SAMPLE, "2019105283")
    assert (p.sire_id, p.sire, p.sire_line) == ("2001103038", "ハーツクライ", "Halo系")
    assert (p.sire_sire_id, p.sire_sire) == ("000a00033a", "サンデーサイレンス")
    assert (p.dam_id, p.dam) == ("000a0137a4", "ダストアンドダイヤモンズ")
    assert (p.damsire_id, p.damsire) == ("000a010544", "Vindication (米)")


def test_parse_pedigree_without_table_fails():
    with pytest.raises(ValueError):
        parse_pedigree("<html></html>", "x")


def test_old_db_gets_new_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE races (race_id TEXT PRIMARY KEY, name TEXT, date TEXT)")
    c.commit(); c.close()
    cols = {r[1] for r in db.connect(path).execute("PRAGMA table_info(races)")}
    assert {"lap_times", "pace_first3f", "pace_last3f"} <= cols


class FakeFetcher:
    def __init__(self, bad=()):
        self.asked, self.bad = [], set(bad)

    def pedigree_html(self, horse_id):
        self.asked.append(horse_id)
        if horse_id in self.bad:
            raise ValueError("no table")
        return SAMPLE

    def discard_pedigree(self, horse_id):
        pass


def test_pending_orders_entries_first_and_collect_resumes():
    conn = synthetic_db(n_days=3, races_per_day=2)
    conn.execute("INSERT INTO entry_races (race_id, date) VALUES ('e1', '2030-01-01')")
    conn.execute("INSERT INTO entries (race_id, number, horse_id) VALUES ('e1', 1, 'h59')")
    ids = pending_horse_ids(conn)
    assert ids[0] == "h59" and len(ids) == len(set(ids))
    f = FakeFetcher()
    assert collect_pedigrees(conn, f, limit=3, log=lambda *_: None) == 3
    assert len(pending_horse_ids(conn)) == len(ids) - 3
    assert collect_pedigrees(conn, FakeFetcher(), log=lambda *_: None) == len(ids) - 3
    assert pending_horse_ids(conn) == []


def test_collect_stops_after_consecutive_failures():
    conn = synthetic_db(n_days=3, races_per_day=2)
    f = FakeFetcher(bad=set(pending_horse_ids(conn)))
    with pytest.raises(RuntimeError):
        collect_pedigrees(conn, f, log=lambda *_: None)
    assert len(f.asked) == 5


@pytest.fixture(scope="module")
def conn_with_extras():
    conn = synthetic_db(n_days=40, races_per_day=4)
    conn.execute("UPDATE results SET last3f = 33 + (finish % 7) / 2.0")
    conn.execute("UPDATE races SET pace_first3f = 35 + (substr(race_id, 11, 2) % 3), pace_last3f = 34 + (substr(race_id, 9, 2) % 4)")
    for i in range(60):
        conn.execute("INSERT INTO pedigree (horse_id, sire_id, damsire_id) VALUES (?, ?, ?)", (f"h{i}", f"s{i % 5}", f"d{i % 4}"))
    conn.commit()
    return conn


def test_lap_and_pedigree_features_are_filled_and_past_only(conn_with_extras):
    raw = load_results(conn_with_extras)
    df = build_features(raw.copy())
    late = df[df["n_prev"] >= 5]
    for c in ["prev_pace_diff", "pace_diff_3", "l3f_vs_lap_3", "sire_top3_rate", "sire_ctdist_top3_rate", "damsire_top3_rate"]:
        assert late[c].notna().mean() > 0.9, c
    assert set(LAP_FEATURES + PEDIGREE_FEATURES) <= set(ARTICLE_FEATURES)

    # 評価日以後の結果を入れ替えても、その日の特徴量は変わらない
    day = sorted(raw["date"].unique())[25]
    other = raw.copy()
    m = other["date"] >= day
    rng = np.random.default_rng(1)
    other.loc[m, "finish"] = rng.integers(1, 11, int(m.sum()))
    other.loc[m, "last3f"] = rng.uniform(32, 40, int(m.sum()))
    other.loc[m, "pace_first3f"] = rng.uniform(33, 40, int(m.sum()))
    other.loc[m, "pace_last3f"] = rng.uniform(33, 40, int(m.sum()))
    a = build_features(raw.copy()); b = build_features(other)
    key = ["race_id", "number"]
    a = a[a["date"] == day].sort_values(key).reset_index(drop=True)
    b = b[b["date"] == day].sort_values(key).reset_index(drop=True)
    for c in LAP_FEATURES + PEDIGREE_FEATURES:
        assert np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True), c


def test_features_work_without_pedigree_or_lap():
    df = build_features(load_results(synthetic_db(n_days=10, races_per_day=2)))
    assert df["sire_top3_rate"].isna().all() and df["prev_pace_diff"].isna().all()


def test_cli_pedigree_command(tmp_path, monkeypatch, capsys):
    from keibaai import cli
    path = tmp_path / "k.db"
    conn = synthetic_db(n_days=3, races_per_day=2)
    disk = db.connect(path)
    conn.backup(disk._conn if hasattr(disk, "_conn") else disk)
    disk.close()
    monkeypatch.setattr(cli, "Fetcher", lambda cache_dir: FakeFetcher())
    cli.main(["pedigree", "--db", str(path), "--limit", "4"])
    out = capsys.readouterr().out
    assert "done: 4 horses" in out and "未取得" in out
