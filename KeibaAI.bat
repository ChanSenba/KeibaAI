@echo off
chcp 65001 >nul
rem ダブルクリックで KeibaAI の操作画面をブラウザで開く
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv が見つかりません。README の手順で先にセットアップしてください。
  pause
  exit /b 1
)
.venv\Scripts\python.exe -c "import streamlit" 2>nul || .venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m keibaai.gui
if errorlevel 1 pause
