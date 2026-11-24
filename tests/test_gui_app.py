import datetime as dt

import pandas as pd

from keibaai import db, gui_app, weekly

from test_weekly import add_result, pred_table


def cards_table():
    return pd.DataFrame([
        {"number": 1, "bracket": 1, "horse_name": "テスト<馬>", "jockey": "騎手&", "mark": "◎", "star": "★",
         "conf_mark": "◆",
         "factor_total": 7.5, "p_win_ability": 0.4, "p_win_market": 0.35, "ev_ability": 0.8, "ev_market": 1.4,
         "win_odds": 3.5, "popularity": 1, "pt_p_win_quality": 8.0},
        {"number": 2, "bracket": 2, "horse_name": "二番", "jockey": "騎手", "mark": "", "star": "",
         "conf_mark": "",
         "factor_total": None, "p_win_ability": None, "p_win_market": None, "ev_ability": None, "ev_market": None,
         "win_odds": None, "popularity": None, "pt_p_win_quality": None},
    ])


def test_race_cards_html_escapes_and_shows_key_values():
    rv = {"mark": "◎", "number": 1, "name": "テスト<馬>", "sub": "", "summary": "勝率40%",
          "rows": [{"name": "馬の質", "pt": 8.0, "grade": "強み", "text": "根拠<テスト>"}]}
    out = gui_app.race_cards_html(cards_table(), [rv])
    assert "テスト&lt;馬&gt;" in out and "テスト<馬>" not in out  # 馬名はエスケープする
    assert "騎手&amp;" in out  # 騎手もエスケープする
    assert "根拠&lt;テスト&gt;" in out  # 見解の本文もエスケープする
    assert "◎" in out and "★" in out
    assert "総合スコア" in out and "7.5" in out
    assert "勝率(能力)" in out and "40%" in out
    assert "期待値" in out and "1.40" in out  # オッズ込みの期待値を優先する
    assert "単勝" in out and "3.5倍" in out
    assert "<details" in out and "馬の質" in out  # 要素ごとの点数と見解は畳んだ先にある
    assert out.count("総合スコア") == 1  # 値のない馬には出さない


def test_race_cards_html_wraps_always_visible_part_for_sticky_header():
    rv = {"mark": "◎", "number": 1, "name": "テスト<馬>", "sub": "", "summary": "勝率40%",
          "rows": [{"name": "馬の質", "pt": 8.0, "grade": "強み", "text": "根拠"}]}
    table = cards_table()
    out = gui_app.race_cards_html(table, [rv])
    # 常時表示の部分(見出しと数字)は .kb-card-top で包み、スマホ幅ではカード内に固定する(style.css)
    assert out.count('class="kb-card-top"') == len(table)  # カード1枚(表の1行)に1つ
    assert "テスト&lt;馬&gt;" in out and "テスト<馬>" not in out  # 包んでも馬名のエスケープは崩れない
    top = out.index('class="kb-card-top"')
    head = out.index('class="kb-card-head"')
    nums = out.index('class="kb-card-nums"')
    details = out.index("<details")
    assert top < head < nums < details  # 包むのは詳細より前の部分だけ


def test_marks_display_strips_dataframe_attrs():
    # live_marks の出力は attrs に◆の成績(DataFrame)を持つ。成績ページの表示では
    # _marks_display で取り分けて attrs を空にする(2026-11-24 の実機エラー対策)
    conn = db.connect(":memory:")
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    add_result(conn, "202505010101", "2025-03-01", [1, 2, 3, 4, 5])
    marks = weekly.live_marks(conn)
    out, conf, races = gui_app._marks_display(marks)
    assert out.attrs == {} and marks.attrs.get("conf") is not None  # 元の表には手を付けない
    assert races == 1 and conf is not None and not conf.empty
    html = out.drop(columns=["top3"]).style.map(gui_app.mark_style, subset=["mark"]).to_html()
    assert "◎" in html


def test_race_cards_html_confidence_mark():
    out = gui_app.race_cards_html(cards_table(), [])
    assert "◎◆" in out  # 本命感は◎の隣に出す
    one = cards_table().drop(columns=["conf_mark"])  # ◆を付け始める前の予想(列がない)でも、そのまま出る
    assert "◆" not in gui_app.race_cards_html(one, [])


def test_race_cards_html_without_prediction():
    one = cards_table().drop(columns=["mark", "star", "conf_mark", "factor_total", "p_win_ability", "p_win_market",
                                      "ev_ability", "ev_market", "pt_p_win_quality"])
    out = gui_app.race_cards_html(one, [])
    assert "二番" in out and "単勝" in out and "3.5倍" in out  # 出馬表だけでも、オッズまでは出す
    assert "<details" not in out  # 点数も見解もなければ、畳むものはない


def test_results_key_ignores_odds_writes(tmp_path):
    path = str(tmp_path / "keiba.db")
    conn = db.connect(path)
    weekly.save_predictions(conn, pred_table(), dt.datetime(2025, 2, 28, 20))
    add_result(conn, "202505010101", "2025-03-01", [1, 2, 3, 4, 5])
    conn.close()
    key = gui_app._results_key(path)
    conn = db.connect(path)
    with conn:  # 開催日のオッズ取得は、この3つの表に書く。成績のキーには触れないはず
        conn.execute("INSERT INTO odds_snapshots (race_id, number, snapped_at, win_odds)"
                     " VALUES ('202505010101', 1, '2025-03-01T09:00:00', 2.5)")
        conn.execute("INSERT INTO bet_odds_snapshots (race_id, snapped_at, kind, combo, odds)"
                     " VALUES ('202505010101', '2025-03-01T09:00:00', '単勝', '1', 2.5)")
    conn.close()
    assert gui_app._results_key(path) == key
    conn = db.connect(path)  # 結果が増えたらキーは変わる
    weekly.save_predictions(conn, pred_table("202505010201", "2025-03-01"), dt.datetime(2025, 2, 28, 21))
    add_result(conn, "202505010201", "2025-03-01", [1, 2, 3, 4, 5])
    conn.close()
    assert gui_app._results_key(path) != key
