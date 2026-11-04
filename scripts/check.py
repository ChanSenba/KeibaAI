"""変更を出す前の確認を1回で行う（人も Claude も CI も同じものを使う）。

    python scripts/check.py          ruff → pytest
    python scripts/check.py --fast   ruff だけ
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(name: str, args: list[str]) -> bool:
    print(f"== {name}: {' '.join(args)}", flush=True)
    ok = subprocess.run([sys.executable, "-m", *args], cwd=ROOT).returncode == 0
    print(f"== {name}: {'OK' if ok else '失敗'}\n", flush=True)
    return ok


def main() -> int:
    steps = [("ruff", ["ruff", "check", "src", "tests", "scripts"])]
    if "--fast" not in sys.argv[1:]:
        steps.append(("pytest", ["pytest", "-q"]))
    failed = [name for name, args in steps if not run(name, args)]
    if failed:
        print("失敗: " + ", ".join(failed))
        return 1
    print("すべて通りました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
