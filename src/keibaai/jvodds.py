"""JRA-VAN の確定オッズ(全券種)と払戻を読んで、jravan_odds.db の券種ごとの表に入れる。

書き出し(jvexport.py --odds の RACE.txt)を読む。レコードは数字と空白だけの固定長で、位置は 0 始まり。

  共通: 0-1 種別, 2 区分, 3-10 作成日, 11-26 レース(年+月日+場+回+日+R), 27-34 発表月日時分, 35-36 登録頭数, 37-38 出走頭数
  O1(単勝・複勝・枠連, 962): 43- 単勝 28×(馬番2 オッズ4 人気2), 267- 複勝 28×(馬番2 最低4 最高4 人気2),
      603- 枠連 36×(組2 オッズ5 人気2)
  O2(馬連, 2042): 40- 153×(組4 オッズ6 人気3)      O3(ワイド, 2654): 40- 153×(組4 最低5 最高5 人気3)
  O4(馬単, 4031): 40- 306×(組4 オッズ6 人気3)      O5(3連複, 12293): 40- 816×(組6 オッズ6 人気3)
  O6(3連単, 83285): 40- 4896×(組6 オッズ7 人気4)
  HR(払戻, 719): 102- 単勝 3×(馬番2 払戻9 人気2), 141- 複勝 5×13, 206- 枠連 3×13, 245- 馬連 3×(組4 払戻9 人気3),
      293- ワイド 7×16, 453- 馬単 6×16, 549- 3連複 3×(組6 払戻9 人気3), 603- 3連単 6×(組6 払戻9 人気4)
  オッズは 0.1 倍単位、払戻は 100 円あたりの円。

位置は書き出したデータの長さと中身(払戻 ÷ 100 と確定オッズの一致)から確かめた推定(JV-Data 仕様書は手元になかった)。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .db import LOCK_TIMEOUT

JRA_VENUES = {f"{i:02d}" for i in range(1, 11)}

# 券種 → (レコード種別, 表, 組の馬番の数, 始まり, 1件の幅, 件数, オッズの幅(複勝・ワイドは2つ), 人気の幅)
ODDS = {
    "win": ("O1", "odds_win", 1, 43, 8, 28, (4,), 2),
    "place": ("O1", "odds_place", 1, 267, 12, 28, (4, 4), 2),
    "bracket": ("O1", "odds_bracket", 2, 603, 9, 36, (5,), 2),
    "quinella": ("O2", "odds_quinella", 2, 40, 13, 153, (6,), 3),
    "wide": ("O3", "odds_wide", 2, 40, 17, 153, (5, 5), 3),
    "exacta": ("O4", "odds_exacta", 2, 40, 13, 306, (6,), 3),
    "trio": ("O5", "odds_trio", 3, 40, 15, 816, (6,), 3),
    "trifecta": ("O6", "odds_trifecta", 3, 40, 17, 4896, (7,), 4),
}
# 払戻: 券種 → (始まり, 件数, 組の馬番の数, 人気の幅)
PAYOUTS = {
    "win": (102, 3, 1, 2), "place": (141, 5, 1, 2), "bracket": (206, 3, 2, 2), "quinella": (245, 3, 2, 3),
    "wide": (293, 7, 2, 3), "exacta": (453, 6, 2, 3), "trio": (549, 3, 3, 3), "trifecta": (603, 6, 3, 4),
}
KINDS = ("O1", "O2", "O3", "O4", "O5", "O6", "HR")
NUM_COLS = ("a", "b", "c")

SCHEMA = "\n".join(
    f"CREATE TABLE IF NOT EXISTS {table} (race_id TEXT NOT NULL, {', '.join(f'{c} INTEGER NOT NULL' for c in NUM_COLS[:n])}, "
    + ("odds_min REAL, odds_max REAL" if len(widths) == 2 else "odds REAL")
    + f", popularity INTEGER, PRIMARY KEY (race_id, {', '.join(NUM_COLS[:n])})) WITHOUT ROWID;"
    for _, table, n, _s, _w, _c, widths, _p in ODDS.values()) + """
CREATE TABLE IF NOT EXISTS payouts (race_id TEXT NOT NULL, kind TEXT NOT NULL, combo TEXT NOT NULL, payout INTEGER,
    popularity INTEGER, PRIMARY KEY (race_id, kind, combo)) WITHOUT ROWID;
-- どの区分・発表時刻のオッズを入れたか(レースとレコード種別ごと)
CREATE TABLE IF NOT EXISTS odds_sources (race_id TEXT NOT NULL, kind TEXT NOT NULL, kubun TEXT, made TEXT,
    announced TEXT, PRIMARY KEY (race_id, kind)) WITHOUT ROWID;
"""


def race_id_of(rec: str) -> str:
    return rec[11:15] + rec[19:27]


def is_jra(rec: str) -> bool:
    return rec[19:21] in JRA_VENUES and rec[11:19].isdigit()


def _int(s: str) -> int | None:
    return int(s) if s.isdigit() and int(s) > 0 else None


def _combo(s: str, n: int, bracket: bool = False) -> tuple[int, ...] | None:
    w = 1 if bracket else 2
    nums = [s[i * w:(i + 1) * w] for i in range(n)]
    if not all(x.isdigit() and int(x) > 0 for x in nums):
        return None
    return tuple(int(x) for x in nums)


def parse_odds(rec: str, kind: str) -> list[tuple]:
    """1件のオッズのレコードから、その券種の (組..., オッズ[, オッズ], 人気) の並び。発売のない組・取消は除く。"""
    _, _, n, start, width, count, widths, pop_w = ODDS[kind]
    bracket = kind == "bracket"
    key_w = n if bracket else 2 * n
    out = []
    for i in range(count):
        e = rec[start + i * width:start + (i + 1) * width]
        if len(e) < width:
            break
        combo = _combo(e[:key_w], n, bracket)
        if not combo:
            continue
        p = key_w
        vals = []
        for w in widths:
            v = _int(e[p:p + w])
            vals.append(v / 10 if v is not None else None)
            p += w
        if vals[0] is None:
            continue
        out.append((*combo, *vals, _int(e[p:p + pop_w])))
    return out


def parse_payouts(rec: str) -> list[tuple]:
    """HR の1件から (券種, 組 '01-02' の形, 払戻(円/100円), 人気) の並び。"""
    out = []
    for kind, (start, count, n, pop_w) in PAYOUTS.items():
        bracket = kind == "bracket"
        key_w = n if bracket else 2 * n  # 枠連の組は1桁ずつ
        width = key_w + 9 + pop_w
        for i in range(count):
            e = rec[start + i * width:start + (i + 1) * width]
            combo = _combo(e[:key_w], n, bracket)
            pay = _int(e[key_w:key_w + 9])
            if combo and pay:
                out.append((kind, "-".join(f"{x:02d}" if not bracket else str(x) for x in combo), pay,
                            _int(e[key_w + 9:key_w + 9 + pop_w])))
    return out


def _lines(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            _, _, rec = line.rstrip("\n").partition("\t")
            if rec[:2] in KINDS and is_jra(rec):
                yield rec


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=LOCK_TIMEOUT)
    conn.executescript(SCHEMA)
    return conn


def import_odds(conn: sqlite3.Connection, raw_dir: str | Path, since: str = "0000", log=print) -> dict:
    """raw_dir/RACE.txt の確定オッズと払戻を入れる。同じレース・種別が何度も出てくるので、区分・作成日が最後のものを使う。

    since(YYYYMMDD)より前のレースは入れない。入れたレースの行は、券種ごとに入れ替える。券種ごとのレース数と組み合わせ数を返す。
    """
    conn.executescript(SCHEMA)
    path = Path(raw_dir) / "RACE.txt"
    if not path.exists():
        return {}
    best: dict[tuple[str, str], tuple] = {}  # 1回目: どの行を使うか決める(3連単は1件が大きいので、中身は2回目に読む)
    for i, rec in enumerate(_lines(path)):
        if rec[11:19] < since:
            continue
        key = (race_id_of(rec), rec[:2])
        order = (rec[2], rec[3:11], rec[27:35], i)
        if key not in best or order >= best[key]:
            best[key] = order
    use = {v[3]: k for k, v in best.items()}
    counts = {k: [0, 0] for k in [*ODDS, "payouts"]}
    with conn:
        for i, rec in enumerate(_lines(path)):
            if i not in use:
                continue
            rid, kind = use[i]
            o = best[(rid, kind)]
            conn.execute("INSERT OR REPLACE INTO odds_sources VALUES (?,?,?,?,?)", (rid, kind, o[0], o[1], o[2]))
            if kind == "HR":
                rows = parse_payouts(rec)
                conn.execute("DELETE FROM payouts WHERE race_id = ?", (rid,))
                conn.executemany("INSERT OR REPLACE INTO payouts VALUES (?,?,?,?,?)", [(rid, *r) for r in rows])
                counts["payouts"][0] += 1
                counts["payouts"][1] += len(rows)
                continue
            for name, spec in ODDS.items():
                if spec[0] != kind:
                    continue
                rows = parse_odds(rec, name)
                table = spec[1]
                conn.execute(f"DELETE FROM {table} WHERE race_id = ?", (rid,))
                if rows:
                    conn.executemany(f"INSERT OR REPLACE INTO {table} VALUES ({','.join('?' * (len(rows[0]) + 1))})",
                                     [(rid, *r) for r in rows])
                    counts[name][0] += 1
                    counts[name][1] += len(rows)
            if kind == "O6" and counts["trifecta"][0] % 2000 == 0:
                log(f"  3連単 {counts['trifecta'][0]} レース")
    return counts


NORMAL_PAYOUTS = {"win": 1, "place": 3, "bracket": 1, "quinella": 1, "wide": 3, "exacta": 1, "trio": 1, "trifecta": 1}


def check_payouts(conn: sqlite3.Connection) -> dict:
    """払戻 ÷ 100 と確定オッズが合うかを券種ごとに数える。{券種: (照合した数, 合った数, 同着以外で合わなかった数)}。

    複勝・ワイドはオッズに幅があるので、払戻 ÷ 100 が最低〜最高に入れば合ったとする。
    同着のレース(払戻の組がいつもより多い)は、払戻が組ごとに分かれてオッズより安くなるので、合わなくてよい。
    """
    conn.execute("DROP TABLE IF EXISTS temp.dead_heat")
    conn.execute("CREATE TEMP TABLE dead_heat AS " + " UNION ".join(
        f"SELECT race_id FROM payouts WHERE kind = '{k}' GROUP BY race_id HAVING COUNT(*) > {n}"
        for k, n in NORMAL_PAYOUTS.items()))
    out = {}
    for name, (_, table, n, *_rest) in ODDS.items():
        cols = NUM_COLS[:n]
        if name == "bracket":
            combo = " || '-' || ".join(f"o.{c}" for c in cols)
        else:
            combo = " || '-' || ".join(f"substr('0' || o.{c}, -2)" for c in cols)
        two = name in ("place", "wide")
        ok = ("p.payout / 100.0 BETWEEN o.odds_min - 0.05 AND o.odds_max + 0.05" if two
              else "abs(p.payout / 100.0 - o.odds) < 0.05")
        out[name] = conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM({ok}), 0), "
            f"COALESCE(SUM(NOT ({ok}) AND p.race_id NOT IN (SELECT race_id FROM temp.dead_heat)), 0) "
            f"FROM payouts p JOIN {table} o ON o.race_id = p.race_id AND {combo} = p.combo WHERE p.kind = ?",
            (name,)).fetchone()
    return out
