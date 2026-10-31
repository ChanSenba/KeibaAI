import pandas as pd
from test_entries import _entries_for_future
from test_model import synthetic_db

from keibaai import db, predict

DAY = "2030-01-05"


def disk_db(tmp_path):
    conn = synthetic_db(n_days=40)
    _entries_for_future(conn, DAY)
    disk = db.connect(tmp_path / "k.db")
    conn.backup(disk)
    return disk


def count_fits(monkeypatch):
    calls = []
    real = predict.fit_residual
    monkeypatch.setattr(predict, "fit_residual", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def test_predict_day_is_reused_until_the_card_changes(tmp_path, monkeypatch):
    conn = disk_db(tmp_path)
    fits = count_fits(monkeypatch)
    first = predict.predict_day(conn, DAY)
    n = len(fits)
    assert n >= 1  # オッズ込みは必ず学習する(能力のみはニューラルネット)
    again = predict.predict_day(conn, DAY)
    assert len(fits) == n  # 2回目は保存した結果を返すので、学習し直さない
    pd.testing.assert_frame_equal(first, again)
    again.loc[0, "p_win_ability"] = 9.0  # 返した表を書き換えても、保存した結果は変わらない
    assert predict.predict_day(conn, DAY).loc[0, "p_win_ability"] != 9.0
    assert len(list((tmp_path / predict.CACHE_DIR).glob("predict-*.pkl"))) == 1

    conn.execute("UPDATE entries SET win_odds = win_odds + 1 WHERE number = 1")  # オッズを取り直した
    conn.commit()
    changed = predict.predict_day(conn, DAY)
    assert len(fits) > n  # 中身が変わったので計算し直す
    assert not changed["win_odds"].equals(first["win_odds"])
    assert len(list((tmp_path / predict.CACHE_DIR).glob("predict-*.pkl"))) == 1  # 古い保存は消える


def test_fetch_time_alone_does_not_recompute(tmp_path, monkeypatch):
    conn = disk_db(tmp_path)
    fits = count_fits(monkeypatch)
    predict.predict_day(conn, DAY)
    n = len(fits)
    conn.execute("UPDATE entry_races SET fetched_at = '2030-01-05T09:00:00'")  # 取り直したが中身は同じ
    conn.commit()
    predict.predict_day(conn, DAY)
    assert len(fits) == n


def test_blend_is_reused_until_results_change(tmp_path, monkeypatch):
    conn = disk_db(tmp_path)
    calls = []
    import keibaai.blend as blend
    import keibaai.model as model
    # 学習と混ぜ方の決定そのものは別のテストで見ている。ここでは「何回計算したか」だけを数える
    monkeypatch.setattr(model, "evaluate", lambda df, cut, use_market: calls.append(use_market) or (None, df))
    monkeypatch.setattr(blend, "fit_blend", lambda valid: blend.Blend(0.6, 0.4, -1.9, -1.95))
    b1 = predict.blend_for_day(conn, DAY, valid_months=1)
    assert calls == [True]  # オッズ込みのモデルで決める
    assert len(calls) == 1
    conn.execute("UPDATE entries SET win_odds = win_odds + 1")  # 出馬表のオッズは混ぜ方に関係しない
    conn.commit()
    assert predict.blend_for_day(conn, DAY, valid_months=1) == b1 and len(calls) == 1
    conn.execute("UPDATE results SET finish = finish + 1 WHERE rowid = (SELECT MAX(rowid) FROM results)")  # 成績が変わった
    conn.commit()
    predict.blend_for_day(conn, DAY, valid_months=1)
    assert len(calls) == 2


def test_memory_db_is_never_cached(tmp_path, monkeypatch):
    conn = synthetic_db(n_days=40)
    _entries_for_future(conn, DAY)
    fits = count_fits(monkeypatch)
    predict.predict_day(conn, DAY)
    n = len(fits)
    predict.predict_day(conn, DAY)
    assert len(fits) == 2 * n


def test_broken_cache_file_is_recomputed(tmp_path):
    conn = disk_db(tmp_path)
    first = predict.predict_day(conn, DAY)
    (file,) = (tmp_path / predict.CACHE_DIR).glob("predict-*.pkl")
    file.write_bytes(b"not a pickle")
    pd.testing.assert_frame_equal(predict.predict_day(conn, DAY), first)


def test_factor_values_are_reused_until_the_data_changes(tmp_path, monkeypatch):
    from keibaai import factor_view

    conn = disk_db(tmp_path)
    calls = []
    real = factor_view.factor_values
    monkeypatch.setattr(factor_view, "factor_values", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = factor_view.factor_values_cached(conn, DAY)
    assert calls == [1]
    again = factor_view.factor_values_cached(conn, DAY)
    assert calls == [1]  # 2回目は保存した値を返す
    pd.testing.assert_frame_equal(first, again)
    assert len(list((tmp_path / predict.CACHE_DIR).glob("factors-*.pkl"))) == 1

    conn.execute("INSERT INTO pedigree (horse_id) VALUES ('hx')")  # 血統を取得して件数が変わった
    conn.commit()
    factor_view.factor_values_cached(conn, DAY)
    assert calls == [1, 1]
