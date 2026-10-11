# KeibaAI

JRA（中央競馬）のレース結果を集めて、着順・勝率を予測するための個人用ツール。

## 使い方（Windows / Python 3.10+）

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
keibaai collect --from 2024-11-23 --to 2024-11-24
pytest
```

取得したページは `data/cache/`、結果は `data/keiba.db`（SQLite）に保存されます。

## netkeiba 利用上の注意

- 個人利用のみ。取得データは公開しない（`data/` は `.gitignore` 済み）
- アクセスは3秒以上あけ、取得済みページは再取得しない
- 429/503 が返ったら待って打ち切る。連続実行は避ける

## 状態

- 結果ページの解析は、実ページ1行ぶんの情報に基づく手作りサンプルでテスト済み。
  実際のページでの動作は、手元のPCでの取得確認がまだ必要です。
