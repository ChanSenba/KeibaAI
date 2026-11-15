"""DB から読んで作るのに時間のかかる表を、DB の隣(model_cache)に保存して使い回す。

学習に使う表が変わると model_revision が進むので、それとコードの指紋が同じ間だけ使う。
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import db

CACHE_DIR = "model_cache"


def _db_file(conn: sqlite3.Connection) -> Path | None:
    return next((Path(f) for _, name, f in conn.execute("PRAGMA database_list") if name == "main" and f), None)


def cached(conn: sqlite3.Connection, kind: str, compute):
    """compute() の結果を <DB のフォルダ>/model_cache/<kind>-<指紋>.pkl に保存し、指紋が同じなら次からそれを返す。

    メモリ上の DB や更新検知(model_revision)のない DB では保存しない。同じ kind の古い保存は消す。
    """
    file, revision = _db_file(conn), db.model_revision(conn)
    if file is None or revision is None:
        return compute()
    key = hashlib.sha256(repr((kind, revision, pd.__version__, np.__version__)).encode())
    for src in sorted(Path(__file__).parent.glob("*.py")):
        key.update(src.name.encode())
        key.update(src.read_bytes())
    folder = file.parent / CACHE_DIR
    path = folder / f"{kind}-{key.hexdigest()[:20]}.pkl"
    if path.exists():
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception:  # 壊れた保存は捨てて作り直す
            path.unlink(missing_ok=True)
    out = compute()
    # 画面と定期実行が同時に動くことがあるので、保存や古い保存の削除に失敗しても計算結果はそのまま返す
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob(f"{kind}-*.pkl"):
            try:
                old.unlink(missing_ok=True)
            except OSError:
                pass
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with tmp.open("wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    except OSError:
        pass
    return out
