# KeibaAI（Claude 向けの決まりごと）

JRA の成績・血統・調教タイムから各馬の勝率を予測し、買い方を出す個人用ツール。尚さんが Windows の PC で使う。
詳しい使い方は README.md、仕様と検証結果は docs/spec.md、bat ファイルは docs/bat.md。

## 確認のしかた

変更したら、PR を出す前に必ずこれを通す（CI でも同じものが動く）。

```
python scripts/check.py          # ruff（明らかなバグ） → pytest の順に実行
python scripts/check.py --fast   # ruff だけ
```

pytest は全部で約3分かかる。途中は `python -m pytest tests/test_xxx.py` で絞ってよいが、最後は全部通す。

## 構成

- `src/keibaai/cli.py`：`keibaai` コマンドの入口。GUI も同じ関数を呼ぶ
- `gui.py` は CLI の関数を呼ぶ薄い層、画面そのものは `gui_app.py`（Streamlit）
- 特徴量 `features.py`、学習 `model.py` / `nn_model.py`、予想 `predict.py`、買い方 `betting.py` / `bettypes.py`、印 `marks.py`
- JRA-VAN の取り込みは `jv*.py`（JV-Link は 32bit の Windows 部品なので、このコンテナでは動かない）

## 守ること

- 特徴量は、そのレースより前の結果だけから作る（当日以後の情報が混ざらないことをテストで確かめている。新しい特徴量にもテストを足す）
- 取得データ（`data/`、`*.db`）はコミットしない。テスト用の HTML は `tests/fixtures/` の小さなサンプルだけ
- netkeiba へのアクセスは3〜5秒あけ、取得済みのページは取り直さない。ログイン情報や JRA-VAN の利用キーは扱わない
- 取得系のコマンドを同時に動かさない（SQLite がふさがる）
- 画面・メッセージ・ドキュメントは日本語。尚さんへの報告も日本語
- PR のマージは尚さんが行う
