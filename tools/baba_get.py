"""jra.go.jp のページを1つ取ってキャッシュする(3秒以上あける。キャッシュがあれば取らない)。
使い方: python baba_get.py <url> [<url> ...]
"""
import hashlib
import sys
import time
from pathlib import Path

import requests

from keibaai.fetch import USER_AGENT

CACHE = Path(r"D:\KeibaAI\pr1-check\data\cache\jra_baba")
CACHE.mkdir(parents=True, exist_ok=True)
last = 0.0
for url in sys.argv[1:]:
    name = url.split("jra.go.jp/", 1)[-1].replace("/", "_") or "index"
    if name.endswith("_"):
        name += "index.html"
    path = CACHE / name
    if path.exists():
        print("cached", url, path.stat().st_size)
        continue
    time.sleep(max(0.0, last + 3.5 - time.time()))
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30, allow_redirects=False)
    last = time.time()
    print(r.status_code, url, r.headers.get("Content-Type"), len(r.content), r.headers.get("Location") or "")
    if r.status_code == 200:
        path.write_bytes(r.content)
