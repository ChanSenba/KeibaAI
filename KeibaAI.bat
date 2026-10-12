@echo off
chcp 65001 >nul
rem ダブルクリックで KeibaAI の操作画面をブラウザで開く
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv が見つかりません。README の手順で先にセットアップしてください。
  pause
  exit /b 1
)
rem このフォルダの src を直接使う(取得中で pip install -e . ができないときも起動できるように)
set "PYTHONPATH=%~dp0src"
rem 画面用のライブラリだけ入れる。keibaai.exe は置き換えないので、取得の実行中でも失敗しない
.venv\Scripts\python.exe -c "import streamlit" 2>nul || .venv\Scripts\python.exe -m pip install "streamlit>=1.50"
.venv\Scripts\python.exe -m keibaai.gui
if errorlevel 1 pause
