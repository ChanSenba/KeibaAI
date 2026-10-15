"""JRA-VAN のレース成績(RA・SE)と馬のマスタ(UM)を読んで、races / results / pedigree に入れる。

書き出し(jvexport.py、32bit Python)で作った生データを読む。1行は「ファイル名<タブ>レコード」。
レコードは Shift_JIS の固定長。JV-Link は1バイトを1文字として返すので、いったんバイト列に戻してから切り出す(位置は 0 始まり)。

  共通: 0-1 種別, 2 区分, 3-10 作成日
  RA(レース, 1272バイト): 11-14 年, 15-18 月日, 19-20 場, 21-22 回, 23-24 日, 25-26 R, 32-91 レース名,
      614 グレード, 616-617 競走種別, 634-636 条件, 697-700 距離, 705-706 トラック, 873-876 発走時刻,
      887 天候, 888 芝の馬場, 889 ダートの馬場, 890- ラップ(3桁×25), 969-971 前半3F, 975-977 後半3F
  SE(出走馬, 555バイト): 11-26 レース(RA と同じ), 27 枠, 28-29 馬番, 30-39 馬の番号, 40-75 馬名, 78 性別, 82-83 馬齢,
      85-89 調教師コード, 90-97 調教師名, 104-167 馬主名, 288-290 斤量, 296-300 騎手コード, 306-313 騎手名,
      324-326 馬体重, 327 増減の符号, 328-330 増減, 331 異常区分, 334-335 確定着順, 338-341 タイム,
      351-358 コーナー通過順(2桁×4), 359-362 単勝オッズ, 363-364 人気, 365-372 本賞金, 373-380 付加賞金, 390-392 後3F
  UM(馬, 1609バイト): 11-20 馬の番号, 204- 3代血統(繁殖登録番号10桁 + 馬名36バイト)が14頭ぶん。
      並びは 父, 母, 父の父, 父の母, 母の父, 母の母, …

レース ID(年+場+回+日+R)と、馬・騎手・調教師の番号は netkeiba と同じ。確定成績(区分 7)の中央競馬(場 01〜10)だけを使う。
netkeiba から集めた 10,371 レース・142,774 頭と突き合わせ、日付・距離・馬場・ラップ・着順・タイム・オッズ・人気・馬体重・賞金が一致することを確かめてある。
違いが残るのは、着差(JRA-VAN はコード)、直線コースの通過順、降着の表記、レース名の表記。

血統の番号は JRA-VAN の繁殖登録番号で、netkeiba の馬 ID とは別の番号。両方を混ぜると同じ種牡馬が別扱いになるので、
取り込むときに netkeiba の番号の行は消す(以後 `keibaai pedigree` は使わない)。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

VENUES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
          "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
WEATHER = {"1": "晴", "2": "曇", "3": "雨", "4": "小雨", "5": "雪", "6": "小雪"}
GROUND = {"1": "良", "2": "稍重", "3": "重", "4": "不良"}
AGE = {"11": "2歳", "12": "3歳", "13": "3歳以上", "14": "4歳以上", "18": "障害3歳以上", "19": "障害4歳以上"}
COND = {"701": "新馬", "702": "未出走", "703": "未勝利", "005": "1勝クラス", "010": "2勝クラス",
        "016": "3勝クラス", "999": "オープン"}
COND_SHORT = {"701": "新馬", "702": "未出走", "703": "未勝利", "005": "1勝", "010": "2勝", "016": "3勝", "999": "OP"}
GRADE = {"A": "GI", "B": "GII", "C": "GIII", "L": "L", "F": "J.GI", "G": "J.GII", "H": "J.GIII"}
SEX = {"1": "牡", "2": "牝", "3": "セ"}
STATUS = {"1": "取", "2": "除", "3": "除", "4": "中", "5": "失"}  # 異常区分 → netkeiba と同じ1文字
RACE_COLS = ["race_id", "name", "date", "venue", "round", "course_type", "direction", "distance", "weather", "ground",
             "start_time", "lap_times", "pace_first3f", "pace_last3f", "course", "rail"]
RESULT_COLS = ["race_id", "number", "finish", "status", "bracket", "horse_id", "horse_name", "sex_age",
               "weight_carried", "jockey", "jockey_id", "time_sec", "margin", "passing", "last3f", "win_odds",
               "popularity", "body_weight", "body_weight_diff", "trainer", "trainer_id", "owner", "prize"]
PEDIGREE_COLS = ["horse_id", "sire_id", "sire", "dam_id", "dam", "damsire_id", "damsire", "sire_sire_id", "sire_sire"]
PED_START, PED_WIDTH = 204, 46

# JV-Link は Shift_JIS の1バイトを Windows-1252 の1文字として返す。その文字を元のバイトに戻す表
_TO_BYTE = {}
for _b in range(0x80, 0xA0):
    try:
        _TO_BYTE[ord(bytes([_b]).decode("cp1252"))] = _b
    except UnicodeDecodeError:
        pass


def to_bytes(rec: str) -> bytes:
    """JVRead が返した文字列を、元の Shift_JIS のバイト列に戻す。"""
    return rec.translate(_TO_BYTE).encode("latin-1", errors="replace")


def from_bytes(b: bytes) -> str:
    """to_bytes の逆(テストでレコードを作るのに使う)。"""
    return b.decode("latin-1").translate({v: k for k, v in _TO_BYTE.items()})


def _txt(b: bytes) -> str:
    return b.decode("cp932", errors="replace").replace("　", " ").strip()


def _num(b: bytes, div: int = 1):
    """数字の欄を数にする。空・0・数字でないものは None。div で割る(0.1 単位の欄は 10)。"""
    s = b.decode("ascii", errors="replace").strip()
    if not s.isdigit() or int(s) == 0:
        return None
    return int(s) / div if div != 1 else int(s)


def _track(code: str) -> tuple[str | None, str | None]:
    """トラックコード → (芝/ダ/障, 左/右/直線)。"""
    c = int(code) if code.isdigit() else 0
    if 10 <= c <= 22:
        return "芝", "直線" if c == 10 else ("左" if c <= 16 else "右")
    if 23 <= c <= 29:
        return "ダ", {23: "左", 24: "右", 25: "左", 26: "右", 27: "左", 28: "右", 29: "直線"}[c]
    if 51 <= c <= 59:
        return "障", None
    return None, None


# トラックコード → 内回り・外回りの別(netkeiba の「芝右 外」「芝右 外-内」と同じ書き方)
_COURSE = {12: "外", 13: "内-外", 14: "外-内", 15: "内2周", 16: "外2周",
           18: "外", 19: "内-外", 20: "外-内", 21: "内2周", 22: "外2周", 25: "内", 26: "内"}
INNER_OUTER_VENUES = {"新潟", "中山", "京都", "阪神"}  # 芝に内回りと外回りがある競馬場


def course_of(code: str, venue: str | None) -> str | None:
    """トラックコードと競馬場 → 外 / 内 / 外-内 など。内外の別がなければ None。"""
    c = int(code) if code.isdigit() else 0
    if c in (11, 17) and venue in INNER_OUTER_VENUES:
        return "内"
    return _COURSE.get(c)


def rail_of(code: bytes) -> str | None:
    """コース区分(A〜E。仮柵の位置)。空ならNone(ダートや、出走馬名表の段階)。"""
    c = code.decode("ascii", errors="replace").strip()
    return c if c in ("A", "B", "C", "D", "E") else None


def race_id_of(b: bytes) -> str:
    return (b[11:15] + b[19:27]).decode("ascii", errors="replace")


RESULT_KUBUNS = (b"6", b"7")  # 7 は確定成績(月曜)、6 は速報成績(全馬の着順確定)


def is_jra_result(b: bytes, kubuns: tuple = RESULT_KUBUNS) -> bool:
    """中央競馬のレコードで、区分が kubuns のどれかか。既定は成績の入った区分。"""
    return b[2:3] in kubuns and b[19:21].decode("ascii", errors="replace") in VENUES and b[11:19].isdigit()


def race_name(title: str, grade: str, kind: str, cond: str) -> str | None:
    """netkeiba の書き方に寄せたレース名。features.parse_class が格付けを読めるように、格や条件を括弧で付ける。"""
    if title:
        tag = GRADE.get(grade) or COND_SHORT.get(cond)
        return f"{title}({tag})" if tag else title
    return (AGE.get(kind, "") + COND.get(cond, "")) or None


def parse_ra(b: bytes, kubuns: tuple = RESULT_KUBUNS) -> tuple | None:
    """RA の1件を RACE_COLS の順の組にする。対象外は None。"""
    if b[:2] != b"RA" or len(b) < 978 or not is_jra_result(b, kubuns):
        return None
    ctype, direction = _track(b[705:707].decode("ascii", errors="replace"))
    turf, dirt = chr(b[888]), chr(b[889])
    ground = GROUND.get(dirt if ctype == "ダ" else turf) or GROUND.get(turf) or GROUND.get(dirt)
    laps = [b[890 + 3 * i:893 + 3 * i] for i in range(25)]
    laps = [f"{int(x) / 10:.1f}" for x in laps if x.isdigit() and int(x) > 0]
    st = b[873:877].decode("ascii", errors="replace")
    ymd = b[11:19].decode()
    name = race_name(_txt(b[32:92]), chr(b[614]), b[616:618].decode("ascii", errors="replace"),
                     b[634:637].decode("ascii", errors="replace"))
    venue = VENUES[b[19:21].decode()]
    return (race_id_of(b), name, f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}", venue, int(b[25:27]),
            ctype, direction, _num(b[697:701]), WEATHER.get(chr(b[887])), ground,
            f"{st[:2]}:{st[2:]}" if st.isdigit() and st != "0000" else None,
            "-".join(laps) or None, _num(b[969:972], 10), _num(b[975:978], 10),
            course_of(b[705:707].decode("ascii", errors="replace"), venue), rail_of(b[709:711]))


def parse_se(b: bytes, kubuns: tuple = RESULT_KUBUNS) -> tuple | None:
    """SE の1件を RESULT_COLS の順の組にする。対象外は None。"""
    if b[:2] != b"SE" or len(b) < 393 or not is_jra_result(b, kubuns) or not b[28:30].isdigit():
        return None
    status = STATUS.get(chr(b[331]))
    finish = None if status else _num(b[334:336])
    t = b[338:342].decode("ascii", errors="replace")
    time_sec = int(t[0]) * 60 + int(t[1:]) / 10 if t.isdigit() and int(t) > 0 else None
    corners = [b[351 + 2 * i:353 + 2 * i] for i in range(4)]
    corners = [str(int(c)) for c in corners if c.isdigit() and int(c) > 0]
    bw = _num(b[324:327])
    if bw is not None and bw >= 999:  # 計量不能
        bw = None
    d, sign = b[328:331].decode("ascii", errors="replace"), chr(b[327])
    diff = None
    if bw is not None and d.isdigit() and int(d) < 999 and sign in "+- ":
        diff = -int(d) if sign == "-" else int(d)
    elif bw is not None and not d.strip():  # 初出走などで前走比がないとき。netkeiba は 0 と表示する
        diff = 0
    last3f = _num(b[390:393], 10)
    if last3f is not None and last3f >= 99.9:
        last3f = None
    prize = ((_num(b[365:373]) or 0) + (_num(b[373:381]) or 0)) / 100 or None  # 本賞金 + 付加賞金(万円)
    return (race_id_of(b), int(b[28:30]), finish, status, _num(b[27:28]), b[30:40].decode("ascii", errors="replace"),
            _txt(b[40:76]), f"{SEX.get(chr(b[78]), '')}{_num(b[82:84]) or ''}", _num(b[288:291], 10),
            _txt(b[306:314]), b[296:301].decode("ascii", errors="replace"), time_sec, None,
            "-".join(corners) or None, last3f, _num(b[359:363], 10), _num(b[363:365]), bw, diff,
            _txt(b[90:98]), b[85:90].decode("ascii", errors="replace"), _txt(b[104:168]).replace(" ", ""), prize)


def parse_um(b: bytes) -> tuple | None:
    """UM の1件を (PEDIGREE_COLS の順の組, 作成日) にする。対象外は None。"""
    if b[:2] != b"UM" or len(b) < PED_START + PED_WIDTH * 14 or b[2:3] == b"0":
        return None
    anc = []
    for i in range(14):
        o = PED_START + PED_WIDTH * i
        hn = b[o:o + 10].decode("ascii", errors="replace")
        anc.append((hn if hn.isdigit() and int(hn) > 0 else None, _txt(b[o + 10:o + PED_WIDTH]) or None))
    horse = b[11:21].decode("ascii", errors="replace")
    if not horse.isdigit():
        return None
    (sire, dam, sire_sire, _sire_dam, damsire) = anc[:5]
    return (horse, *sire, *dam, *damsire, *sire_sire), b[3:11].decode("ascii", errors="replace")


def _records(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            _, _, rec = line.rstrip("\n").partition("\t")
            if rec:
                yield to_bytes(rec)


def import_races(conn: sqlite3.Connection, raw_dir: str | Path, since: str = "0000-00-00",
                 replace: bool = False, refresh_since: str | None = None) -> tuple[int, int]:
    """raw_dir の成績(jvexport の出力)から races と results に入れる。(レース数, 出走数) を返す。

    読むのは RT_0B15.txt(速報成績。開催の当日から)と RACE.txt(確定成績。翌週に出る)。同じレースが両方にあれば確定成績を使う。
    since(YYYY-MM-DD)より前のレースは入れない。すでに DB にあるレースは、replace=True のときだけ入れ替える。
    ただし refresh_since 以後のレースは、DB にあっても入れ替える(速報で入れたものを、あとから確定成績に置き換えるため)。
    レースと出走馬は同じレースをひとまとめに扱う。
    """
    paths = [p for p in (Path(raw_dir) / "RT_0B15.txt", Path(raw_dir) / "RACE.txt") if p.exists()]
    if not paths:
        return 0, 0
    if replace:
        have = set()
    else:
        have = {r[0] for r in conn.execute("SELECT race_id FROM races WHERE date < ? OR date IS NULL",
                                           (refresh_since or "9999-99-99",))}
    races, results, level = {}, {}, {}
    for path in paths:
        for b in _records(path):
            kind = b[:2]
            if kind not in (b"RA", b"SE"):
                continue
            row = parse_ra(b) if kind == b"RA" else parse_se(b)
            if not row or row[0] in have:
                continue
            rid, kubun = row[0], b[2:3]
            if kubun < level.get((rid, kind), b"0"):  # 確定成績(7)を速報(6)で上書きしない
                continue
            if kubun > level.get((rid, kind), b"0"):  # 区分が上がったら、そのレースの古い行は捨てる
                level[(rid, kind)] = kubun
                if kind == b"SE":
                    results.pop(rid, None)
            if kind == b"RA":
                if row[2] >= since:
                    races[rid] = row
            else:
                results.setdefault(rid, {})[row[1]] = row
    ids = [i for i in races if i in results]  # 出走馬のないレースや、レースのない出走馬は入れない
    n_res = 0
    with conn:
        for i in ids:
            conn.execute(f"INSERT OR REPLACE INTO races ({','.join(RACE_COLS)}) VALUES ({','.join('?' * len(RACE_COLS))})",
                         races[i])
            conn.execute("DELETE FROM results WHERE race_id = ?", (i,))
            conn.executemany(
                f"INSERT INTO results ({','.join(RESULT_COLS)}) VALUES ({','.join('?' * len(RESULT_COLS))})",
                list(results[i].values()))
            n_res += len(results[i])
    return len(ids), n_res


def import_pedigree(conn: sqlite3.Connection, raw_dir: str | Path) -> int:
    """raw_dir の DIFN.txt(馬のマスタ)から pedigree に入れる。入れた頭数を返す。

    同じ馬が何度も出てくるので、作成日が新しいものを使う。書き出しに入っていた馬だけを上書きする
    (差分の書き出しには一部の馬しか入っていないので、表を空にしてから入れてはいけない)。
    netkeiba から取った血統は番号の体系が違うので、残っていれば消す。
    """
    path = Path(raw_dir) / "DIFN.txt"
    if not path.exists():
        return 0
    best: dict[str, tuple[str, tuple]] = {}
    for b in _records(path):
        parsed = parse_um(b)
        if parsed and (parsed[0][0] not in best or parsed[1] >= best[parsed[0][0]][0]):
            best[parsed[0][0]] = (parsed[1], parsed[0])
    if not best:
        return 0
    with conn:
        # INSERT OR REPLACE だと行ごと入れ替わり、血統データ(jvblood)で入れた sire_line が消えるので、列だけ更新する
        update = ", ".join(f"{c} = excluded.{c}" for c in PEDIGREE_COLS[1:])
        conn.executemany(
            f"INSERT INTO pedigree ({','.join(PEDIGREE_COLS)}) VALUES ({','.join('?' * len(PEDIGREE_COLS))}) "
            f"ON CONFLICT(horse_id) DO UPDATE SET {update}",
            [row for _, row in best.values()])
        conn.execute(f"DELETE FROM pedigree WHERE {NETKEIBA_ID}")
    return len(best)


# netkeiba の馬 ID の形の父・母の父(生まれ年で始まる 19…/20…、輸入馬の 000a…)。JRA-VAN の繁殖登録番号は 11…/12…/14… で始まる
NETKEIBA_ID = " OR ".join(f"{c} GLOB '{p}*'" for c in ("sire_id", "damsire_id") for p in ("19", "20", "00"))
