"""JRA-VAN(JV-Link)からの取得を、64bit 側(CLI・GUI・毎週の処理)からまとめて呼ぶ。

JV-Link は 32bit の部品なので、書き出しは 32bit Python で jvexport を動かし、ここではその結果を DB に入れる。
netkeiba と違ってアクセス間隔の待ちはない(JV-Link の内部サービスのため、開き直しの間は数秒あける)。
"""
from __future__ import annotations

import contextlib
import datetime as dt
import functools
import inspect
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import jvexport

try:
    import msvcrt  # Windows のファイルロック
except ImportError:
    msvcrt = None
try:
    import fcntl  # POSIX のファイルロック
except ImportError:
    fcntl = None

def _find_python32() -> str:
    """pywin32 を入れた 32bit Python。環境変数 KEIBAAI_PYTHON32 があればそれ、なければ決まった場所で見つかったもの。"""
    candidates = [os.environ.get("KEIBAAI_PYTHON32"), r"C:\Python311-32\jvenv\Scripts\python.exe",
                  r"C:\Python311-32\python.exe"]
    return next((c for c in candidates if c and Path(c).exists()), candidates[1])


PYTHON32 = _find_python32()
RAW_DIR = "data/jv_raw"
RECENT_DAYS = 14  # この日数より新しいレースは、取り込み直すたびに入れ替える(速報成績 → 確定成績)


class JVError(RuntimeError):
    pass


_last_run = 0.0  # 前の書き出しが終わった時刻(続けて開き直すと JV-Link の内部サービスが落ちることがあったため)

# 取得の排他と、一時的なファイルロックへの再試行の設定(テストで差し替えられるよう定数で持つ)
FETCH_LOCK_FILE = ".fetch.lock"  # raw_root 直下に置く、取得の順番待ちのためのファイル
LOCK_POLL = 0.5  # ほかの取得が終わるのを待つときの確認間隔(秒)
LOCK_RETRY = 5  # ファイルが一時的にロックされていたときの再試行回数(最初の試行を含む)
LOCK_RETRY_INTERVAL = 0.2  # 再試行の間隔(秒)

_lock_slots: dict[str, dict] = {}  # ロックファイルごとの、プロセス内での持ち主(スレッドと入れ子の深さ)
_lock_slots_guard = threading.Lock()


def _lock_slot(path: Path) -> dict:
    with _lock_slots_guard:
        return _lock_slots.setdefault(str(path), {"mutex": threading.Lock(), "owner": None, "depth": 0})


def _try_lock(f) -> bool:
    """開いたロックファイルを、待たずにロックできたら True。"""
    if msvcrt is not None:
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    if fcntl is not None:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True
    return True  # ロックの仕組みがない環境では、プロセス内の mutex だけで守る


def _unlock(f) -> None:
    try:
        if msvcrt is not None:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        elif fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass  # ハンドルを閉じれば OS が解放するので、失敗してもそのまま閉じる


@contextlib.contextmanager
def fetch_lock(raw_root: str | Path):
    """JV-Link の取得を1つずつにする(プロセスをまたいで効く)。

    書き出し先のファイル名が取得どうしで共通のため、同時に走ると片方の削除・読み込みが
    「ファイルが使用中」(WinError 32)で落ちる。raw_root 直下のロックファイルを OS のファイルロックで押さえ、
    取れるまで待つ(待たされる側はエラーにせず、順番に実行する)。ロックは OS が管理するので、
    持ったままプロセスが終わっても残らない。同じスレッドからの入れ子はそのまま通す
    (取得の中で取得を呼んでも止まらないように)。
    """
    root = Path(raw_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / FETCH_LOCK_FILE
    slot = _lock_slot(path.resolve())
    ident = threading.get_ident()
    with _lock_slots_guard:
        nested = slot["owner"] == ident
        if nested:
            slot["depth"] += 1
    if nested:
        try:
            yield
        finally:
            with _lock_slots_guard:
                slot["depth"] -= 1
        return
    slot["mutex"].acquire()  # 同じプロセスのほかのスレッドとは、まずここで順番待ちする
    try:
        with path.open("a+b") as f:
            if f.seek(0, os.SEEK_END) == 0:
                f.write(b"\0")
                f.flush()  # msvcrt のバイト範囲ロックは、実在するバイトにしか掛けられない
            waited = False
            while not _try_lock(f):
                if not waited:
                    print("ほかの取得が終わるのを待っています…", flush=True)
                    waited = True
                time.sleep(LOCK_POLL)
            with _lock_slots_guard:
                slot["owner"], slot["depth"] = ident, 1
            try:
                yield
            finally:
                with _lock_slots_guard:
                    slot["owner"], slot["depth"] = None, 0
                _unlock(f)
    finally:
        slot["mutex"].release()


def _locked_fetch(fn):
    """取得の関数全体を fetch_lock で囲むデコレータ。排他の単位は raw_root 引数の場所。"""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        # 省略されたときは、関数自身の既定値(定義時の RAW_DIR)を使う。本体の書き出し先とロックの場所をそろえるため
        raw_root = sig.bind_partial(*args, **kw).arguments.get("raw_root", sig.parameters["raw_root"].default)
        with fetch_lock(raw_root):
            return fn(*args, **kw)

    return wrapper


def _retry_locked(fn):
    """ファイルが一時的にロックされている(PermissionError)間だけ、間隔をあけて数回まで試す。

    書き出し側の終了直後や、ウイルス対策ソフトの確認で一瞬だけ掴まれることがあるため。
    それ以外のエラーはそのまま投げる。
    """
    tries = max(1, LOCK_RETRY)
    for i in range(tries):
        try:
            return fn()
        except PermissionError:
            if i + 1 >= tries:
                raise
            time.sleep(LOCK_RETRY_INTERVAL)


def _unlink(path: Path) -> None:
    """前の書き出しを消す(なければ何もしない)。一時的にロックされていても少し待って消す。"""
    _retry_locked(lambda: path.unlink(missing_ok=True))


def _open_read(path: Path):
    """読み込み用に開く。一時的にロックされていても少し待って開く。"""
    return _retry_locked(lambda: path.open(encoding="utf-8"))


def available(python32: str = PYTHON32) -> bool:
    """JV-Link を呼ぶ 32bit Python があるか(Windows でセットアップ済みの PC だけ)。"""
    return sys.platform == "win32" and Path(python32).exists()


def export(out: str | Path, from_ymd: str, specs: list[str], option: int = 1, rt: list[str] = (),
           python32: str = PYTHON32, timeout: float | None = None) -> None:
    """jvexport を 32bit Python で動かす。出力(件数など)はそのまま print する。失敗・timeout 秒を超えたら JVError。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in [f"{s}.txt" for s in specs] + [f"RT_{r.split(':')[0]}.txt" for r in rt]:
        _unlink(out / name)  # 「該当データなし」のとき、前の日の書き出しが残らないように
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent), "PYTHONIOENCODING": "utf-8"}
    cmd = [python32, "-m", "keibaai.jvexport", "--from", from_ymd, "--out", str(out), "--option", str(option),
           "--specs", *specs, "--rt", *rt]
    global _last_run
    time.sleep(max(0.0, _last_run + jvexport.RT_INTERVAL - time.monotonic()))
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired as e:
        _last_run = time.monotonic()
        raise JVError(f"JV-Link からの書き出しが {timeout:.0f} 秒で終わらなかったので止めました") from e
    _last_run = time.monotonic()
    for line in (p.stdout + p.stderr).splitlines():
        print(line)
    if p.returncode:
        raise JVError("JV-Link からの書き出しに失敗しました(上のメッセージを確認)。"
                      "JV-Link の内部サービスが止まっている場合は、PC の再起動で直ります")


@_locked_fetch
def fetch_entries(conn: sqlite3.Connection, date: str, odds: bool = True, raw_root: str | Path = RAW_DIR,
                  python32: str = PYTHON32, now: dt.datetime | None = None, race_ids: list[str] | None = None) -> int:
    """その日(YYYY-MM-DD)の出馬表と単勝オッズを JRA-VAN から取って保存する。保存したレース数を返す。

    出馬表はその日の全レースぶんをまとめて取る。race_ids を渡すと、オッズはそのレースだけ取る(1レースだけ取得するとき)。

    木曜〜金曜の出走馬名表・出馬表は「今週のデータ」(RACE, option 2)に、当日の出馬表は速報(0B15)に出る。両方取る。
    当日の天候・馬場の発表と変更は速報(0B14)に出る。あれば出馬表の天候・馬場をそれで置き換える(取り直すたびに新しくなる)。
    馬番は枠順の発表(金曜)のあとに入るので、それより前は 0 を返す。
    """
    from .jventries import import_entries, race_keys, read_entries

    raw = Path(raw_root) / "entries"  # 成績の取り込みが使う RACE.txt と混ぜない
    ymd = date.replace("-", "")
    week = (dt.date.fromisoformat(date) - dt.timedelta(days=7)).strftime("%Y%m%d")
    export(raw, week, ["RACE"], option=2, rt=[f"0B15:{ymd}", f"0B14:{ymd}"], python32=python32)
    _unlink(raw / "RT_0B31.txt")  # 前に取った別の日のオッズを混ぜない
    numbered = {p.race.race_id for p in read_entries(raw, date)}  # 馬番の入ったレースだけ(枠順の発表前はオッズもない)
    keys = [k for k in race_keys(raw, date) if k[:4] + k[8:] in numbered and (race_ids is None or k[:4] + k[8:] in race_ids)]
    if keys and odds:
        try:
            export(raw, ymd, [], rt=[f"0B31:{k}" for k in keys], python32=python32)
        except JVError:
            print("オッズの書き出しに失敗しました。出馬表だけ保存します")
    now = now or dt.datetime.now()
    return import_entries(conn, raw, date, now.isoformat(timespec="seconds"))


TRAINING_DAYS = 14  # 調教タイムは、DB にある最新の日からこの日数ぶん戻って取り直す(速報の後から入る分を拾う)


@_locked_fetch
def fetch_training(conn: sqlite3.Connection, raw_root: str | Path = RAW_DIR, python32: str = PYTHON32,
                   today: dt.date | None = None) -> int:
    """坂路・ウッドの調教タイム(SLOP・WOOD)を、DB にある最新の日の少し前から取って入れる。入れた件数を返す。

    出馬表を取るときに一緒に呼ぶ(`jv --what training` を別に実行しなくても、出走馬の直近の調教が入る)。
    """
    from .jvtraining import import_raw

    today = today or dt.date.today()
    try:
        last = conn.execute("SELECT MAX(date) FROM jv_training").fetchone()[0]
    except sqlite3.Error:
        last = None
    start = (dt.date.fromisoformat(last) if last else today) - dt.timedelta(days=TRAINING_DAYS)
    raw = Path(raw_root) / "training"  # 成績・血統の書き出し(raw_root 直下)と混ぜない
    export(raw, f"{start:%Y%m%d}", ["SLOP", "WOOD"], option=1, python32=python32)
    return import_raw(conn, raw)


@_locked_fetch
def fetch_results(conn: sqlite3.Connection, start: dt.date, end: dt.date, raw_root: str | Path = RAW_DIR,
                  python32: str = PYTHON32, today: dt.date | None = None) -> tuple[int, int]:
    """start〜end の成績を JRA-VAN から取って入れる。(レース数, 出走数) を返す。

    確定成績は通常データ(RACE の差分)に、開催直後の速報成績は速報(0B15)に出る。直近の日は両方取り、
    確定成績が出たら速報の内容と入れ替える。
    """
    from .jvrace import import_races

    today = today or dt.date.today()
    raw = Path(raw_root) / "recent"  # セットアップの全件の書き出し(raw_root/RACE.txt)は上書きしない
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    # 速報が残っているのは直近だけ。開催のある土日と月曜(祝日開催)だけ開く(開く回数を減らすため)
    recent = [d for d in days if today - dt.timedelta(days=7) <= d <= today and (d.weekday() >= 5 or d.weekday() == 0)]
    export(raw, f"{start - dt.timedelta(days=1):%Y%m%d}", ["RACE"], option=1,
           rt=[f"0B15:{d:%Y%m%d}" for d in recent], python32=python32)
    refresh = max(start, today - dt.timedelta(days=RECENT_DAYS)).isoformat()
    out = import_races(conn, raw, start.isoformat(), refresh_since=refresh)
    try:  # 地方・海外の成績(海外帰り・地方帰りの判定に使う)。取れなくても中央の成績は使える
        from .jvother import import_other
        print(f"地方・海外の成績: {import_other(conn, [raw])} 件")
    except Exception as e:
        print(f"地方・海外の成績を入れられませんでした: {e}")
    return out


# 速報オッズの種別(JVRTOpen のデータ種別)→ 入っているレコード。2026-10-16 に実機で、10/17 の前日発売のオッズが
# 6種別とも1レース1件ずつ取れることを確かめた(レコードの形は確定オッズ O1〜O6 と同じ)
RT_ODDS_SPECS = {"0B31": "O1", "0B32": "O2", "0B33": "O3", "0B34": "O4", "0B35": "O5", "0B36": "O6"}
# jvodds の券種 → 買い目(betting)の券種名。枠連は枠番の組なので使わない。複勝・ワイドは安全側に下限を使う
BET_KINDS = {"win": "単勝", "place": "複勝", "quinella": "馬連", "wide": "ワイド", "exacta": "馬単",
             "trio": "三連複", "trifecta": "三連単"}


def race_key(conn: sqlite3.Connection, race_id: str) -> str:
    """race_id(12桁)を JV-Link のレースのキー(年+月日+場+回+日+R の16桁)にする。月日は出馬表(entry_races)から。"""
    row = (conn.execute("SELECT date FROM entry_races WHERE race_id = ?", (race_id,)).fetchone()
           or conn.execute("SELECT date FROM races WHERE race_id = ?", (race_id,)).fetchone())  # 終わったレースは成績のほう
    if not row or not row[0]:
        raise ValueError(f"{race_id} の出馬表がありません。先に entries で取得してください")
    return race_id[:4] + row[0][5:7] + row[0][8:10] + race_id[4:]


def announced_at(rec: str) -> str | None:
    """オッズのレコードの発表月日時分(27-34)を ISO の日時にする。年はレースの年。"""
    s = rec[27:35]
    if not s.isdigit() or s == "00000000":
        return None
    return f"{rec[11:15]}-{s[:2]}-{s[2:4]}T{s[4:6]}:{s[6:8]}:00"


def _rt_records(path: Path):
    if path.exists():
        with _open_read(path) as f:
            for line in f:
                _, _, rec = line.rstrip("\n").partition("\t")
                if rec:
                    yield rec


def read_bet_odds(raw_dir: str | Path, key: str) -> tuple[dict, str | None]:
    """書き出した速報オッズ(RT_0B31〜0B36.txt)から、そのレースの {券種: {馬番の組: オッズ}} と発表時刻(単勝の)を作る。"""
    from .jvodds import ODDS, parse_odds

    out: dict = {}
    announced = None
    for spec, code in RT_ODDS_SPECS.items():
        for rec in _rt_records(Path(raw_dir) / f"RT_{spec}.txt"):
            if rec[:2] != code or rec[11:27] != key:
                continue
            if code == "O1":
                announced = announced_at(rec)
            for kind, name in BET_KINDS.items():
                if ODDS[kind][0] != code:
                    continue
                tail = len(ODDS[kind][6]) + 1  # 組のあとに、オッズ(複勝・ワイドは下限と上限)と人気
                table = {r[:-tail]: r[-tail] for r in parse_odds(rec, kind)}
                if table:
                    out[name] = table
    return out, announced


@_locked_fetch
def fetch_bet_odds(conn: sqlite3.Connection, race_id: str, raw_root: str | Path = RAW_DIR,
                   python32: str = PYTHON32) -> dict:
    """1レースの全券種の速報オッズを JRA-VAN から取って {券種: {馬番の組: オッズ}} にする(netkeiba の6回の取得の代わり)。

    JV-Link を種別ごとに1回ずつ、間をあけて開く(6種別で30秒ほど)。発売前などで取れない券種は入らない。
    """
    key = race_key(conn, race_id)
    raw = Path(raw_root) / "bet_odds"
    export(raw, key[:8], [], rt=[f"{s}:{key}" for s in RT_ODDS_SPECS], python32=python32)
    return read_bet_odds(raw, key)[0]


def read_pool_odds(raw_dir: str | Path, keys: dict[str, str]) -> dict[str, tuple[dict, dict]]:
    """書き出した速報オッズ(pools.RT_SPECS の種別)から、レースごとの ({券種: {組: オッズ}}, {券種: {組: 上限}})。

    keys は JV-Link のレースのキー → race_id。単勝(券種と同じ時点の市場)も入れる。複勝は下限をオッズ、上限を2つ目の表に入れる。
    三連複は ★の特徴量には使わないが、仕様 8-25 のペーパートレード(paper.py)用に記録へ残す。
    """
    from .jvodds import ODDS, parse_odds
    from .pools import RT_SPECS

    out: dict[str, tuple[dict, dict]] = {}
    for spec, code in RT_SPECS.items():
        for rec in _rt_records(Path(raw_dir) / f"RT_{spec}.txt"):
            rid = keys.get(rec[11:27])
            if rec[:2] != code or not rid:
                continue
            odds, hi = out.setdefault(rid, ({}, {}))
            for kind, name in BET_KINDS.items():
                if ODDS[kind][0] != code or kind not in ("win", "place", "exacta", "trio", "trifecta"):
                    continue
                n = ODDS[kind][2]
                rows = parse_odds(rec, kind)
                if rows:
                    odds[name] = {r[:n]: r[n] for r in rows}
                    if len(ODDS[kind][6]) == 2:
                        hi[name] = {r[:n]: r[n + 1] for r in rows if r[n + 1]}
    return out


@_locked_fetch
def snapshot_pool_odds(conn: sqlite3.Connection, date: str, race_ids: list[str] | None = None,
                       now: dt.datetime | None = None, raw_root: str | Path = RAW_DIR,
                       python32: str = PYTHON32) -> int:
    """date の各レースの複勝・馬単・三連複・三連単の速報オッズを JRA-VAN から取り、bet_odds_snapshots に入れる。入れたレース数を返す。

    ★・期待値のモデルの券種の特徴量(pools.py)に使う。三連複は仕様 8-25 のペーパートレード(paper.py)用。
    JV-Link は1回だけ起動し、種別・レースごとに間をあけて開く(24レースで約8分)。発売前のレースは飛ばす。
    """
    from . import db
    from .pools import RT_SPECS

    stamp = (now or dt.datetime.now()).isoformat(timespec="seconds")
    # 単勝オッズが出ていない(発売前の)レースは開かない。開くたびに5秒あけるので、無駄に待たない
    on_sale = {r[0] for r in conn.execute(
        "SELECT DISTINCT e.race_id FROM entries e JOIN entry_races c USING (race_id) WHERE c.date = ? AND e.win_odds > 0",
        (date,))}
    ids = race_ids or [r[0] for r in conn.execute("SELECT race_id FROM entry_races WHERE date = ? ORDER BY race_id", (date,))]
    keys = {}
    for rid in ids:
        if rid not in on_sale:
            continue
        try:
            keys[race_key(conn, rid)] = rid
        except ValueError as e:
            print(e)
    if not keys:
        return 0
    raw = Path(raw_root) / "pools"
    # 固まったときに週次の処理ごと止まらないよう、1回の開き直しあたり30秒を上限にする
    export(raw, date.replace("-", ""), [], rt=[f"{s}:{k}" for s in RT_SPECS for k in keys], python32=python32,
           timeout=60 + 30 * len(RT_SPECS) * len(keys))
    n = 0
    for rid, (odds, hi) in read_pool_odds(raw, keys).items():
        if db.save_bet_odds_snapshot(conn, rid, odds, stamp, hi):
            n += 1
    return n


@_locked_fetch
def snapshot_odds(conn: sqlite3.Connection, date: str, race_ids: list[str] | None = None,
                  now: dt.datetime | None = None, raw_root: str | Path = RAW_DIR, python32: str = PYTHON32) -> int:
    """date(YYYY-MM-DD)の各レースの単勝の速報オッズを JRA-VAN から取り、odds_snapshots に追記する。記録したレース数を返す。

    snapped_at は取った時刻、official_at は JRA-VAN のオッズの発表時刻。レースは出馬表を取り込み済みのもの(entry_races)。
    JV-Link は1回だけ起動し、レースごとに間をあけて開く。オッズのないレースは飛ばす。
    """
    from .jventries import parse_o1
    from .jvrace import to_bytes

    stamp = (now or dt.datetime.now()).isoformat(timespec="seconds")
    ids = race_ids or [r[0] for r in conn.execute("SELECT race_id FROM entry_races WHERE date = ? ORDER BY race_id", (date,))]
    keys = {}
    for rid in ids:
        try:
            keys[race_key(conn, rid)] = rid
        except ValueError as e:
            print(e)
    if not keys:
        return 0
    raw = Path(raw_root) / "snapshot"
    export(raw, date.replace("-", ""), [], rt=[f"0B31:{k}" for k in keys], python32=python32)
    n = 0
    seen = set()
    for rec in _rt_records(raw / "RT_0B31.txt"):
        o = parse_o1(to_bytes(rec))
        rid = keys.get(rec[11:27])
        if not o or not rid or rid in seen:
            continue
        rows = [(rid, num, stamp, announced_at(rec), win, pop) for num, (win, pop) in o[1].items() if win is not None]
        if not rows:
            print(f"skip {rid}: オッズがまだありません")
            continue
        with conn:
            conn.executemany("INSERT OR REPLACE INTO odds_snapshots (race_id, number, snapped_at, official_at, win_odds, "
                             "popularity) VALUES (?,?,?,?,?,?)", rows)
        seen.add(rid)
        n += 1
    return n


def store_odds_history(conn: sqlite3.Connection, path: Path, wanted: set[str]) -> int:
    """書き出した時系列オッズ(RT_0B41.txt)の単勝を、発表時刻つきで odds_snapshots に入れる。入れたレース数を返す。

    1行が1回の発表(レコードは確定オッズ O1 と同じ形)。発売前でオッズが 0 の行と、wanted にないレースは入れない。
    JV-Link は、渡したキーの「日」が違っても同じ日付・場・R の別のレースを返すことがあるので、レコードのレース欄で照合する。
    """
    from .jventries import parse_o1
    from .jvrace import to_bytes

    seen = set()
    for rec in _rt_records(path):
        o = parse_o1(to_bytes(rec))
        at = announced_at(rec)
        if not o or o[0] not in wanted or not at:
            continue
        rows = [(o[0], num, at, at, win, pop, "ts") for num, (win, pop) in o[1].items() if win is not None]
        if not rows:
            continue
        with conn:
            conn.executemany("INSERT OR REPLACE INTO odds_snapshots (race_id, number, snapped_at, official_at, win_odds, "
                             "popularity, source) VALUES (?,?,?,?,?,?,?)", rows)
        seen.add(o[0])
    return len(seen)


@_locked_fetch
def import_odds_history(conn: sqlite3.Connection, race_ids: list[str], raw_root: str | Path = RAW_DIR,
                        python32: str = PYTHON32, batch: int = 50) -> int:
    """race_ids の時系列オッズ(JV-Link 0B41。発売開始から締め切りまでの単勝オッズの推移)を取って odds_snapshots に入れる。

    レースが終わったあとでも取れる(1年前でも確認済み)。すでに取り込んだレース(source='ts')は飛ばすので、途中で止めても続きから。
    batch レースごとに書き出して入れる(JV-Link は1レース5秒ほどかかる)。入れたレース数を返す。
    """
    done = {r[0] for r in conn.execute("SELECT DISTINCT race_id FROM odds_snapshots WHERE source = 'ts'")}
    keys = {}
    for rid in race_ids:
        if rid in done:
            continue
        try:
            keys[race_key(conn, rid)] = rid
        except ValueError as e:
            print(e)
    raw = Path(raw_root) / "odds_history"
    items = list(keys.items())
    n = 0
    for i in range(0, len(items), batch):
        chunk = dict(items[i:i + batch])
        export(raw, next(iter(chunk))[:8], [], rt=[f"0B41:{k}" for k in chunk], python32=python32)
        n += store_odds_history(conn, raw / "RT_0B41.txt", set(chunk.values()))
        print(f"{min(i + batch, len(items))}/{len(items)} レース", flush=True)
    return n
