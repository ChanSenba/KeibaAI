@echo off
chcp 65001 >nul
rem タスクスケジューラ用: 毎週の処理(前の週の回顧と今週末の予想)を実行する
cd /d "%~dp0"
if not exist data mkdir data
set "PYTHONPATH=%~dp0src"
.venv\Scripts\python.exe -c "from keibaai.cli import main; main()" weekly >> data\weekly.log 2>&1
