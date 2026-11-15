from test_model import synthetic_db

from keibaai import db, diskcache, features


def disk_db(tmp_path):
    disk = db.connect(tmp_path / "k.db")
    synthetic_db(n_days=10).backup(disk)
    return disk


def test_results_are_reused_until_the_db_changes(tmp_path, monkeypatch):
    conn = disk_db(tmp_path)
    calls = []
    real = features._load_results
    monkeypatch.setattr(features, "_load_results", lambda c: calls.append(1) or real(c))
    first = features.load_results(conn)
    again = features.load_results(conn)
    assert len(calls) == 1
    assert first.equals(again)
    conn.execute("UPDATE results SET finish = finish + 1 WHERE rowid = (SELECT MAX(rowid) FROM results)")
    conn.commit()
    features.load_results(conn)
    assert len(calls) == 2
    assert len(list((tmp_path / diskcache.CACHE_DIR).glob("results-*.pkl"))) == 1  # 古い保存は消える


def test_memory_db_is_not_cached(tmp_path):
    conn = synthetic_db(n_days=5)
    assert diskcache.cached(conn, "x", lambda: 1) == 1
    assert not (tmp_path / diskcache.CACHE_DIR).exists()


def test_resaving_the_same_oikiri_does_not_change_the_db(tmp_path):
    from keibaai.training import TrainingRun, save_training

    conn = disk_db(tmp_path)
    runs = [TrainingRun(race_id="r1", number=1, seq=0, horse_id="h1", date="2030-01-01", critic="良", rank="A")]
    save_training(conn, "r1", runs)
    rev = db.model_revision(conn)
    save_training(conn, "r1", runs)  # 同じ中身の取り直し
    assert db.model_revision(conn) == rev
    save_training(conn, "r1", [TrainingRun(race_id="r1", number=1, seq=0, horse_id="h1", date="2030-01-01",
                                           critic="良", rank="B")])
    assert db.model_revision(conn) != rev
    assert conn.execute("SELECT rank FROM training WHERE race_id = 'r1'").fetchone()[0] == "B"
