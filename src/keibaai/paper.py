"""仕様 8-25 の買い方を、実際には買わずに追って確かめる(ペーパートレード)。

8-25 では、単勝オッズだけから出した市場の勝率とヘネリー版ハーヴィルで組合せの確率を作り、
期待値と的中確率の下限を上げた買い目(期待値≧1.2・的中確率≧10%、三連単は≧3%)が、馬単・三連複・三連単で
回収率の95%区間が100%を上回った。ただし数字はすべて確定オッズで、ルールは約40通りの組み合わせから
選んだものなので、そのままでは信じない。ここでは同じルールを固定したまま、予想の保存と同じタイミングで
記録した発走前のオッズ(bet_odds_snapshots)から買い目を残し、払戻(jravan_odds.db の payouts)で精算して、
買える時点のオッズでも効くかを前向きに確かめる。お金は動かさない。

確率の作り方は 8-25 に合わせる: 勝率は単勝オッズの比例方式(devig.proportional)、並びの確率は
betting のヘネリー版ハーヴィル(GAMMA_2ND・GAMMA_3RD)。対象の3券種は POOL_MIX の対象外なので、
買い方の画面と同じ部品(betting.candidates)でも券種オッズとの混合は入らない。能力モデルは混ぜない。
8-25 と違うのは、オッズが確定ではなく発走前の速報であること(それを確かめるのが追試の目的)。

ルールの定数(EV_MIN・MIN_HIT)は仕様 8-25 の追試用に固定する。変更したら検証が無効になるので、
変えるときは別の追試として分かるようにすること。
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .betting import GAMMA_2ND, GAMMA_3RD, Bet, candidates
from .devig import proportional
from .live import _is_late
from .pools import ODDS_DB

# --- 仕様 8-25 の追試用に固定したルール(変更したら検証が無効になる) ---
KINDS = ("馬単", "三連複", "三連単")
EV_MIN = 1.2  # 期待値の下限
MIN_HIT = {"馬単": 0.10, "三連複": 0.10, "三連単": 0.03}  # 的中確率の下限(三連単だけ確率が小さいので 3%)
STAKE = 100  # 1点の金額(円)
# ---------------------------------------------------------------

PAYOUT_KIND = {"馬単": "exacta", "三連複": "trio", "三連単": "trifecta"}  # jravan_odds.db の payouts の券種名
MIN_JUDGE_BETS = 100  # 精算できた点数がこれ未満の券種は「判定には不足」と出す目安(8-25 は8年で348〜1,049点)


def bets_for_race(numbers: list[int], win_odds, odds: dict[str, dict[tuple[int, ...], float]]) -> list[Bet]:
    """1レースの、8-25 のルールに当たる買い目。numbers は馬番、win_odds はその単勝オッズ、odds は券種ごとのオッズ表。

    勝率は単勝オッズの比例方式。オッズのある組だけが候補になる(betting.candidates)。
    """
    p = proportional(np.asarray(win_odds, dtype=float))
    cands, *_ = candidates(numbers, p, odds, ev_min=EV_MIN, g2=GAMMA_2ND, g3=GAMMA_3RD, min_hit=0.0)
    return [b for b in cands if b.kind in MIN_HIT and b.prob >= MIN_HIT[b.kind]]


def _combo(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split("-"))


def _start_times(conn: sqlite3.Connection, date: str) -> dict[str, str | None]:
    """その日の {race_id: 発走時刻}。発走後の保存・オッズを除く判定(live._is_late)に使う。"""
    try:
        return dict(conn.execute("SELECT race_id, start_time FROM races WHERE date = ?", (date,)).fetchall())
    except sqlite3.Error:
        return {}


def _snapshot_odds(conn: sqlite3.Connection, race_id: str, saved_at: str, date: str,
                   start_time: str | None) -> tuple[str, dict] | None:
    """予想の保存(saved_at)以前でいちばん新しい、その日・発走前のスナップショットの ({券種: {組: オッズ}})。単勝がなければ None。

    保存より後や発走後に取ったオッズで買い目を作ると、買えたはずのない情報が混ざるので使わない。
    数日前の古いスナップショットも使わない(券種オッズの取得に失敗した日の買い目が、古いオッズで作られるのを防ぐ)。
    前日に保存する形もあるので、開催日の前日以降のものは使う。
    """
    limit = saved_at
    if start_time:  # 発走時刻より後のスナップショットは、締め切り後のオッズなので使わない(live._is_late と同じ比べ方)
        h, _, m = start_time.partition(":")
        limit = min(limit, f"{date}T{h.zfill(2)}:{m}")
    earliest = (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat()
    row = conn.execute(
        "SELECT MAX(snapped_at) FROM bet_odds_snapshots "
        "WHERE race_id = ? AND snapped_at <= ? AND substr(snapped_at, 1, 10) >= ?",
        (race_id, limit, earliest)).fetchone()
    if not row or not row[0]:
        return None
    snap = row[0]
    odds: dict[str, dict] = {}
    for kind, combo, o in conn.execute(
            "SELECT kind, combo, odds FROM bet_odds_snapshots WHERE race_id = ? AND snapped_at = ? AND odds > 0",
            (race_id, snap)):
        odds.setdefault(kind, {})[_combo(combo)] = float(o)
    if not odds.get("単勝"):
        return None
    return snap, odds


def _pre_race_predictions(conn: sqlite3.Connection, date: str) -> dict[str, str]:
    """その日の、発走前に保存した予想の {race_id: saved_at}。同じレースは最初の保存を採る。

    結果のあとに保存した予想は除く: after_result の印と、保存時刻が発走を過ぎたもの(live._is_late と同じ見分け方)。
    """
    starts = _start_times(conn, date)
    out: dict[str, str] = {}
    for rid, saved_at, after in conn.execute(
            "SELECT race_id, saved_at, after_result FROM predictions WHERE date = ? ORDER BY saved_at", (date,)):
        if rid in out or after == 1 or saved_at is None:
            continue
        if _is_late(saved_at, date, starts.get(rid)):
            continue
        out[rid] = saved_at
    return out


def record_day(conn: sqlite3.Connection, date: str) -> int:
    """その日の発走前に保存した予想について、保存以前のスナップショットのオッズから買い目を計算し、paper_bets に残す。

    記録は予想1回につき1回だけ: 記録を済ませたレースは paper_recorded に残し(買い目が0点でも)、
    後から別の時点のスナップショットで買い目を足さない(2時点の和集合になると検証が崩れる)。
    スナップショットがまだ無いレースは記録済みにせず、次に呼ばれたときに回す。新規に記録した買い目数を返す。
    """
    done = {r[0] for r in conn.execute("SELECT race_id FROM paper_recorded WHERE date = ?", (date,))}
    starts = _start_times(conn, date)
    total = 0
    for rid, saved_at in _pre_race_predictions(conn, date).items():
        if rid in done:
            continue
        try:
            got = _snapshot_odds(conn, rid, saved_at, date, starts.get(rid))
            if got is None:
                continue
            snap, odds = got
            missing = [k for k in KINDS if not odds.get(k)]
            if missing:  # 券種の取得が一部欠けたスナップショットは、その券種の買い目が静かに抜けるので知らせる
                print(f"ペーパートレード: {rid} のスナップショット({snap})に {'・'.join(missing)} のオッズがありません")
            win = odds["単勝"]
            numbers = sorted(num for (num,) in win)
            bets = bets_for_race(numbers, [win[(num,)] for num in numbers], odds)
            with conn:
                for b in bets:
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO paper_bets (race_id, date, kind, combo, odds, prob, ev, stake, snapped_at, saved_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (rid, date, b.kind, "-".join(f"{x:02d}" for x in b.numbers), b.odds, b.prob, b.ev, STAKE, snap, saved_at))
                    total += cur.rowcount
                conn.execute("INSERT OR IGNORE INTO paper_recorded (race_id, date, saved_at) VALUES (?,?,?)",
                             (rid, date, saved_at))
        except Exception as e:  # 1レースの異常で、その日の残りのレースまで止めない
            print(f"ペーパートレードの記録を飛ばしました({rid}): {e}")
    return total


def _odds_db_path(conn: sqlite3.Connection, odds_db: str | Path | None) -> Path | None:
    if odds_db is not None:
        return Path(odds_db)
    main = next((f for _, name, f in conn.execute("PRAGMA database_list") if name == "main"), "")
    return Path(main).parent / ODDS_DB if main else None


def settle(conn: sqlite3.Connection, odds_db: str | Path | None = None) -> int:
    """払戻(jravan_odds.db の payouts)が入った買い目を精算し、精算した件数を返す。

    精算するのは、そのレース・その券種の払戻が1件以上入った買い目だけ。単勝・複勝の払戻だけ先に入った段階で
    馬単・三連複・三連単を外れ(0)で確定させてしまうと、後から直らないので、その券種の払戻がそろうまで未精算のまま残す。
    払戻は100円あたりの円で、1点100円なので的中はそのまま、外れは 0 を入れる。payouts の組の形は
    jvodds.parse_payouts が作る '01-02-03' の形で、bet_odds_snapshots の組と同じ。
    """
    path = _odds_db_path(conn, odds_db)
    if path is None or not path.exists():
        return 0
    todo = pd.read_sql_query("SELECT race_id, kind, combo FROM paper_bets WHERE payout IS NULL", conn)
    if todo.empty:
        return 0
    oc = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        pays = pd.read_sql_query("SELECT race_id, kind, combo, payout FROM payouts", oc)
    finally:
        oc.close()
    by_race: dict[str, dict[tuple[str, str], int]] = {}
    ready: set[tuple[str, str]] = set()  # (race_id, payouts の券種名) で払戻が入ったもの
    for r in pays.itertuples():
        by_race.setdefault(r.race_id, {})[(r.kind, r.combo)] = int(r.payout)
        ready.add((r.race_id, r.kind))
    rows = [(by_race[b.race_id].get((PAYOUT_KIND[b.kind], b.combo), 0), b.race_id, b.kind, b.combo)
            for b in todo.itertuples() if (b.race_id, PAYOUT_KIND[b.kind]) in ready]
    with conn:
        conn.executemany(
            "UPDATE paper_bets SET payout = ? WHERE race_id = ? AND kind = ? AND combo = ? AND payout IS NULL", rows)
    return len(rows)


def settled_bets(conn: sqlite3.Connection) -> pd.DataFrame:
    """精算済みの買い目(的中は payout > 0、外れは payout = 0)。"""
    return pd.read_sql_query("SELECT * FROM paper_bets WHERE payout IS NOT NULL ORDER BY date, race_id, kind, combo", conn)


def day_bootstrap_roi(bets: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """開催日単位で引き直した回収率の95%区間(仕様 8-25 と同じ物差し)。開催日が2日未満なら (nan, nan)。"""
    days = bets.groupby("date").agg(stake=("stake", "sum"), payout=("payout", "sum"))
    if len(days) < 2 or not days["stake"].sum():
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), size=(n_boot, len(days)))
    rois = days["payout"].to_numpy(dtype=float)[idx].sum(axis=1) / days["stake"].to_numpy(dtype=float)[idx].sum(axis=1) * 100
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return float(lo), float(hi)


def summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """券種ごとの、精算済みの点数・的中・回収率と95%区間(開催日単位)。週次評価に載せる形。"""
    b = settled_bets(conn)
    cols = ["券種", "点数", "的中", "的中率(%)", "投資(円)", "払戻(円)", "回収率(%)", "区間下限(%)", "区間上限(%)"]
    if b.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for kind in KINDS:
        g = b[b["kind"] == kind]
        if g.empty:
            continue
        lo, hi = day_bootstrap_roi(g)
        rows.append({"券種": kind, "点数": len(g), "的中": int((g["payout"] > 0).sum()),
                     "的中率(%)": (g["payout"] > 0).mean() * 100, "投資(円)": int(g["stake"].sum()),
                     "払戻(円)": int(g["payout"].sum()), "回収率(%)": g["payout"].sum() / g["stake"].sum() * 100,
                     "区間下限(%)": lo, "区間上限(%)": hi})
    return pd.DataFrame(rows, columns=cols)


def trend(conn: sqlite3.Connection) -> pd.DataFrame:
    """開催日ごとの、その日までの累計の点数と回収率(全券種の合計)。週次評価に載せる形。"""
    b = settled_bets(conn)
    cols = ["日付", "点数", "累計点数", "累計回収率(%)"]
    if b.empty:
        return pd.DataFrame(columns=cols)
    days = b.groupby("date").agg(n=("stake", "size"), stake=("stake", "sum"), payout=("payout", "sum")).reset_index()
    days["累計点数"] = days["n"].cumsum()
    days["累計回収率(%)"] = days["payout"].cumsum() / days["stake"].cumsum() * 100
    return days.rename(columns={"date": "日付", "n": "点数"})[cols]
