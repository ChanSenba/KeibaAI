"""JV-Link(JRA-VAN Data Lab)から調教の生データを書き出す。32bit版 Python で動かす小さなスクリプト。

JV-Link は 32bit の COM 部品なので、64bit の Python からは呼べない。このファイルは標準ライブラリと pywin32 だけを使い、
JV-Link から読んだレコードを1行1レコードのテキストに書き出すだけにする。中身を読んで DB に入れるのは 64bit 側(jvtraining.py)。

    C:\\Python311-32\\jvenv\\Scripts\\python.exe -m keibaai.jvexport --from 20231001 --out data\\jv_raw

出力は <out>\\<データ種別>.txt。1行は「ファイル名<タブ>レコード」。追記なので、途中で止めても続きから足せる
(続きは --from に最後の日付を渡す)。利用キーはこのスクリプトでは扱わない(JVLink設定で入れたものを JV-Link が使う)。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SPECS = ("SLOP", "WOOD")  # 坂路調教・ウッドチップ調教
BUFFER = 110000


def export_spec(jv, spec: str, from_date: str, option: int, out: Path) -> int:
    """1種類のデータを読んで out に追記する。書いたレコード数を返す。"""
    ret = jv.JVOpen(spec, f"{from_date}000000", option, 0, 0, "")
    code = ret[0] if isinstance(ret, tuple) else ret
    print(f"{spec}: JVOpen -> {ret}", flush=True)
    if code < 0:
        if code == -1:
            print(f"{spec}: 該当データなし", flush=True)
            return 0
        raise SystemExit(f"{spec}: JVOpen が失敗しました({code})")
    n = 0
    with out.open("a", encoding="utf-8", newline="\n") as f:
        while True:
            r, buff, _size, fname = jv.JVRead("", BUFFER, "")
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
            f.write(f"{fname}\t{buff.rstrip(chr(13) + chr(10))}\n")
            n += 1
            if n % 20000 == 0:
                print(f"{spec}: {n} 件", flush=True)
    jv.JVClose()
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="JV-Link から調教の生データを書き出す(32bit Python 用)")
    ap.add_argument("--from", dest="from_date", required=True, help="取得を始める日 YYYYMMDD")
    ap.add_argument("--out", default="data/jv_raw", help="出力先フォルダ")
    ap.add_argument("--option", type=int, default=1, help="JVOpen のオプション(1=通常、3=今週、4=セットアップ(ダイアログなし))")
    ap.add_argument("--specs", nargs="+", default=list(SPECS))
    args = ap.parse_args(argv)
    import win32com.client as w  # 32bit 側にだけ入れる

    jv = w.Dispatch("JVDTLab.JVLink")
    r = jv.JVInit("UNKNOWN")
    if r != 0:
        raise SystemExit(f"JVInit が失敗しました({r})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.specs:
        n = export_spec(jv, spec, args.from_date, args.option, out / f"{spec}.txt")
        print(f"{spec}: {n} 件を書き出しました", flush=True)


if __name__ == "__main__":
    sys.exit(main())
