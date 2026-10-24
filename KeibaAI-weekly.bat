@echo off
chcp 65001 >nul
rem タスクスケジューラ用: 毎週の処理(前の週の回顧と今週末の予想)を実行する
cd /d "%~dp0"
if not exist data mkdir data
set "PYTHONPATH=%~dp0src"
rem 予想のニューラルネット用(PyTorch、CPU 版。初回だけ入れる)
.venv\Scripts\python.exe -c "import torch" 2>nul || .venv\Scripts\python.exe -m pip install "torch>=2.2" >> data\weekly.log 2>&1
.venv\Scripts\python.exe -c "from keibaai.cli import main; main()" weekly >> data\weekly.log 2>&1
