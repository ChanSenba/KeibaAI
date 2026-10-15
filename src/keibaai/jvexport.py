"""JV-Link(JRA-VAN Data Lab)から調教・成績・馬の生データを書き出す。32bit版 Python で動かす小さなスクリプト。

JV-Link は 32bit の COM 部品なので、64bit の Python からは呼べない。このファイルは標準ライブラリと pywin32 だけを使い、
JV-Link から読んだレコードを1行1レコードのテキストに書き出すだけにする。中身を読んで DB に入れるのは 64bit 側
(調教は jvtraining.py、成績と血統は jvrace.py)。データ種別は --specs で選ぶ(SLOP・WOOD=調教、RACE=成績、DIFN=馬のマスタ、BLDN=血統)。

日本語を含むレコード(RACE・DIFN)は、JV-Link が Shift_JIS の1バイトを1文字として返すので、そのまま書き出す。
文字に戻すのは読む側(jvrace.to_bytes)。

    C:\\Python311-32\\jvenv\\Scripts\\python.exe -m keibaai.jvexport --from 20231001 --out data\\jv_raw

出力は <out>\\<データ種別>.txt。1行は「ファイル名<タブ>レコード」。実行のたびに書き直す(取り込みは上書きなので、重ねて取り込んでも増えない)。利用キーはこのスクリプトでは扱わない(JVLink設定で入れたものを JV-Link が使う)。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SPECS = ("SLOP", "WOOD")  # 坂路調教・ウッドチップ調教
BUFFER = 110000
# データ種別ごとに、書き出すレコード種別。ここにない種別は全部書く。
# RACE にはオッズや票数(1件が数万文字)も入っているので、成績(RA・SE)だけにして残りのファイルは読み飛ばす。
KEEP = {"RACE": ("RA", "SE"), "DIFN": ("UM",), "BLDN": ("HN", "SK", "BT")}
# 読み取りの大きさ。大きすぎると1件ごとの読み取りが極端に遅くなる(RA 1272・SE 555・UM 1609 文字)
SMALL_BUFFER = 4000


def export_spec(jv, spec: str, from_date: str, option: int, out: Path) -> int:
    """1種類のデータを読んで out に書き出す。書いたレコード数を返す。"""
    keep = KEEP.get(spec)
    buffer = BUFFERS.get(spec) or (SMALL_BUFFER if keep else BUFFER)
    ret = jv.JVOpen(spec, f"{from_date}000000", option, 0, 0, "")
    if isinstance(ret, tuple) and ret[0] == -2:  # 最初の1回だけ失敗することがあった。もう一度だけ試す
        jv.JVClose()
        ret = jv.JVOpen(spec, f"{from_date}000000", option, 0, 0, "")
    code = ret[0] if isinstance(ret, tuple) else ret
    print(f"{spec}: JVOpen -> {ret}", flush=True)
    if code < 0:
        if code == -1:
            print(f"{spec}: 該当データなし", flush=True)
            return 0
        raise SystemExit(f"{spec}: JVOpen が失敗しました({code})")
    n = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        while True:
            r, buff, _size, fname = jv.JVRead("", buffer, "")
            if r == 0:  # 全部読んだ
                break
            if r == -1:  # ファイルの切り替わり
                continue
            if r == -3:  # ダウンロード中
                time.sleep(1)
                continue
            if r < 0:
                jv.JVClose()
                raise SystemExit(f"{spec}: JVRead が失敗しました({r})")
            if keep and buff[:2] not in keep:  # 要らない種別のファイルは、先頭の1件だけ見て飛ばす
                jv.JVSkip()
                continue
            f.write(f"{fname}\t{buff.rstrip(chr(13) + chr(10))}\n")
            n += 1
            if n % 20000 == 0:
                print(f"{spec}: {n} 件", flush=True)
    jv.JVClose()
    return n


def export_rt(jv, spec: str, keys: list[str], out: Path) -> int:
    """速報系のデータ(JVRTOpen)を、キー(日付 YYYYMMDD やレース)ごとに読んで out に書き出す。書いたレコード数を返す。

    速報は開催からおよそ1週間ぶんだけ取れる。データのない日は飛ばす。
    JV-Link を短い間隔で何度も開き直すと内部のサービスが落ちることがあったので、開くたびに間をあける。
    """
    n = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for i, key in enumerate(keys):
            if i:
                time.sleep(RT_INTERVAL)
            code = jv.JVRTOpen(spec, key)
            code = code[0] if isinstance(code, tuple) else code
            if code < 0:  # -1 はデータなし。ほかのエラーも、その日を飛ばして続ける
                print(f"{spec} {key}: JVRTOpen -> {code}(飛ばします)", flush=True)
                jv.JVClose()
                continue
            m = 0
            while True:
                r, buff, _size, fname = jv.JVRead("", BUFFER, "")
                if r == 0:
                    break
                if r == -1:
                    continue
                if r == -3:
                    time.sleep(1)
                    continue
                if r < 0:
                    print(f"{spec} {key}: JVRead -> {r}(ここまでにします)", flush=True)
                    break
                f.write(f"{fname or key}\t{buff.rstrip(chr(13) + chr(10))}\n")
                m += 1
            jv.JVClose()
            print(f"{spec} {key}: {m} 件", flush=True)
            n += m
    return n


RT_INTERVAL = 5  # 速報を続けて開くときの間隔(秒)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="JV-Link から調教・成績・馬の生データを書き出す(32bit Python 用)")
    ap.add_argument("--from", dest="from_date", required=True, help="取得を始める日 YYYYMMDD")
    ap.add_argument("--out", default="data/jv_raw", help="出力先フォルダ")
    ap.add_argument("--option", type=int, default=4, help="JVOpen のオプション(1=通常の差分、4=セットアップ(ダイアログなし))")
    ap.add_argument("--specs", nargs="*", default=list(SPECS))
    ap.add_argument("--rt", nargs="*", default=[], metavar="種別:キー",
                    help="速報系のデータ。例: 0B15:20261012(その日の速報成績と払戻)。出力は RT_<種別>.txt")
    args = ap.parse_args(argv)
    import win32com.client as w  # 32bit 側にだけ入れる

    jv = w.Dispatch("JVDTLab.JVLink")
    r = jv.JVInit("UNKNOWN")
    if r != 0:
        raise SystemExit(f"JVInit が失敗しました({r})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, spec in enumerate(args.specs):
        if i:
            time.sleep(RT_INTERVAL)  # 続けて開き直すと JV-Link の内部サービスが落ちることがあった
        n = export_spec(jv, spec, args.from_date, args.option, out / f"{spec}.txt")
        print(f"{spec}: {n} 件を書き出しました", flush=True)
    rt: dict[str, list[str]] = {}
    for item in args.rt:
        spec, _, key = item.partition(":")
        rt.setdefault(spec, []).append(key)
    for spec, keys in rt.items():
        if args.specs:
            time.sleep(RT_INTERVAL)
        n = export_rt(jv, spec, keys, out / f"RT_{spec}.txt")
        print(f"{spec}(速報): {n} 件を書き出しました", flush=True)


if __name__ == "__main__":
    sys.exit(main())
