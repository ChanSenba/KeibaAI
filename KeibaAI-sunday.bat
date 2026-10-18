@echo off
chcp 65001 >nul
rem タスクスケジューラ用: 日曜の最終レース後に、今日の回顧と学習・評価を実行する
cd /d "%~dp0"
if not exist data mkdir data
set "PYTHONPATH=%~dp0src"
.venv\Scripts\python.exe -c "from keibaai.cli import main; main()" weekly --after-races >> data\weekly.log 2>&1
