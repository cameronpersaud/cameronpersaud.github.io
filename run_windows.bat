@echo off
REM Double-click to start BrainStack 3D (first run installs the libraries it needs).
cd /d "%~dp0"
if not exist .venv (
    echo Setting up for the first time, this takes a few minutes...
    python -m venv .venv || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python main.py
if errorlevel 1 pause
