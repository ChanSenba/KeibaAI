from keibaai.jvodds import (ODDS, PAYOUTS, check_payouts, connect, import_odds, parse_odds, parse_payouts)

RACE = "2026100305040111"  # 2026-10-03 東京(05) 4回 1日 11R


def rec(kind, length, fields, kubun="5", made="20261005", race=RACE):
    s = list(" " * length)
    for pos, v in {0: kind, 2: kubun, 3: made, 11: race, 27: "10031545", **fields}.items():
        s[pos:pos + len(v)] = v
    return "".join(s)


def odds_rec(kind, entries):
    """券種 kind のオッズのレコード。entries は 組(文字列)→ 値(オッズ・人気をつないだ文字列)。"""
    code, _, n, start, width, count, *_ = ODDS[kind]
    length = {"O1": 960, "O2": 2040, "O3": 2652, "O4": 4029, "O5": 12291, "O6": 83283}[code]
    return rec(code, length, {start + i * width: k + v for i, (k, v) in enumerate(entries.items())})


def hr_rec(fields):
    f = {}
    for kind, items in fields.items():
        start, _, n, pop_w = PAYOUTS[kind]
        width = (n if kind == "bracket" else 2 * n) + 9 + pop_w
        for i, (combo, pay, pop) in enumerate(items):
            f[start + i * width] = combo + f"{pay:09d}" + f"{pop:0{pop_w}d}"
    return rec("HR", 717, f, kubun="2")


def test_parse_odds_each_kind():
    assert parse_odds(odds_rec("win", {"01": "003502", "02": "----00", "03": "000000"}), "win") == [(1, 3.5, 2)]
    assert parse_odds(odds_rec("place", {"01": "0012001501"}), "place") == [(1, 1.2, 1.5, 1)]
    assert parse_odds(odds_rec("bracket", {"12": "0012305"}), "bracket") == [(1, 2, 12.3, 5)]
    assert parse_odds(odds_rec("wide", {"0102": "0012300150004"}), "wide") == [(1, 2, 12.3, 15.0, 4)]
    assert parse_odds(odds_rec("exacta", {"0201": "001234010"}), "exacta") == [(2, 1, 123.4, 10)]
    assert parse_odds(odds_rec("trio", {"010203": "000567011"}), "trio") == [(1, 2, 3, 56.7, 11)]
    assert parse_odds(odds_rec("trifecta", {"030102": "00238830437"}), "trifecta") == [(3, 1, 2, 2388.3, 437)]


def test_parse_payouts():
    r = hr_rec({"win": [("09", 390, 2)], "bracket": [("57", 930, 5)], "trifecta": [("091410", 238830, 437)]})
    assert parse_payouts(r) == [("win", "09", 390, 2), ("bracket", "5-7", 930, 5), ("trifecta", "09-14-10", 238830, 437)]


def write(tmp_path, recs):
    (tmp_path / "RACE.txt").write_text("".join(f"x.jvd\t{r}\n" for r in recs), encoding="utf-8")


def test_import_uses_latest_record_and_checks_payouts(tmp_path):
    old = odds_rec("trifecta", {"091410": "99999990001"})
    new = odds_rec("trifecta", {"091410": "00238830437", "140910": "00300000500"})
    new = new[:2] + "5" + "20261006" + new[11:]  # 作成日が新しい
    nar = rec("O6", 83283, {}, race="2026100345010111")  # 地方の場コードは入れない
    hr = hr_rec({"trifecta": [("091410", 238830, 437)], "win": [("09", 390, 2)]})
    write(tmp_path, [new, old, nar, hr])
    conn = connect(tmp_path / "o.db")
    counts = import_odds(conn, tmp_path, log=lambda *a: None)
    assert counts["trifecta"] == [1, 2] and counts["payouts"] == [1, 2]
    assert conn.execute("SELECT odds, popularity FROM odds_trifecta WHERE a=9 AND b=14 AND c=10").fetchone() == (2388.3, 437)
    assert check_payouts(conn)["trifecta"] == (1, 1, 0)
    assert check_payouts(conn)["win"] == (0, 0, 0)
    # 入れ直しても増えない
    import_odds(conn, tmp_path, log=lambda *a: None)
    assert conn.execute("SELECT COUNT(*) FROM odds_trifecta").fetchone()[0] == 2


def test_import_since_and_missing_file(tmp_path):
    conn = connect(tmp_path / "o.db")
    assert import_odds(conn, tmp_path) == {}
    write(tmp_path, [odds_rec("quinella", {"0102": "000123004"})])
    assert import_odds(conn, tmp_path, since="20270101", log=lambda *a: None)["quinella"] == [0, 0]
    assert import_odds(conn, tmp_path, since="20190101", log=lambda *a: None)["quinella"] == [1, 1]
