"""既存DBを開くだけの画面表示が、他接続の読み書きを妨げない。"""
import sqlite3

import pytest

from keibaai import db


@pytest.mark.parametrize('transaction', ['BEGIN', 'BEGIN IMMEDIATE'])
def test_initialized_db_opens_during_another_transaction(tmp_path, monkeypatch, transaction):
    path = tmp_path / 'k.db'
    setup = db.connect(path)
    setup.execute("INSERT INTO races(race_id) VALUES ('r')")
    setup.commit()
    setup.close()
    other = sqlite3.connect(path)
    other.execute(transaction)
    other.execute('SELECT * FROM races').fetchall()  # 読み取りトランザクションも実際にロックする
    real_connect = sqlite3.connect
    monkeypatch.setattr(db.sqlite3, 'connect', lambda path, **kw: real_connect(path, timeout=0.05))
    try:
        conn = db.connect(path)
        try:
            assert conn.execute('SELECT COUNT(*) FROM races').fetchone()[0] == 1
            assert conn.total_changes == 0
            assert not conn.in_transaction
        finally:
            conn.close()
    finally:
        other.rollback()
        other.close()


def test_partial_revision_install_is_repaired(tmp_path):
    path = tmp_path / 'k.db'
    conn = db.connect(path)
    conn.execute('DROP TRIGGER model_revision_results_update')
    conn.commit()
    conn.close()
    conn = db.connect(path)
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='model_revision_results_update'").fetchone()
    conn.close()


def test_failed_initialization_closes_connection(tmp_path, monkeypatch):
    opened = []
    real_connect = sqlite3.connect
    def connect(path, **kw):
        conn = real_connect(path, **kw)
        opened.append(conn)
        return conn
    def fail(conn):
        conn.execute("INSERT INTO races(race_id) VALUES ('uncommitted')")
        raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(db.sqlite3, 'connect', connect)
    monkeypatch.setattr(db, '_install_model_revision', fail)
    path = tmp_path / 'k.db'
    with pytest.raises(sqlite3.OperationalError):
        db.connect(path)
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        opened[0].execute('SELECT 1')
    other = real_connect(path, timeout=0.05)
    other.execute("INSERT INTO races(race_id) VALUES ('recovered')")
    other.commit()
    assert other.execute('SELECT race_id FROM races').fetchall() == [('recovered',)]
    other.close()


def test_connect_waits_for_other_writers(tmp_path, monkeypatch):
    seen = {}
    real_connect = sqlite3.connect
    monkeypatch.setattr(db.sqlite3, 'connect', lambda path, **kw: seen.update(kw) or real_connect(path, **kw))
    db.connect(tmp_path / 'k.db').close()
    assert seen['timeout'] == db.LOCK_TIMEOUT >= 30
