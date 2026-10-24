@echo off
setlocal

echo ============================================
echo  Marathi-English Speech Translator Backend
echo ============================================
echo.

REM ── Check if .env exists ─────────────────────────────────────
if not exist .env (
    echo [WARNING] .env not found. Copying .env.example...
    copy .env.example .env >nul
    echo.
    echo [!] Open backend\.env and set your HF_TOKEN, then run again.
    echo.
    pause
    exit /b 1
)

REM ── Create fresh virtual environment (if not exists) ────────
if not exist venv (
    echo Creating Python 3.12 virtual environment...
    py -3.12 -m venv venv
)
if errorlevel 1 (
    echo [ERROR] 'python' not found. Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM ── Activate venv ────────────────────────────────────────────
call venv\Scripts\activate.bat

REM ── Upgrade pip ──────────────────────────────────────────────
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM ── Install core dependencies ─────────────────────────────────
echo Installing dependencies (this may take a minute)...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency install failed. See errors above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Backend ready at http://localhost:8000
echo  API docs at   http://localhost:8000/docs
echo  Press Ctrl+C to stop.
echo ============================================
echo.

REM ── Start FastAPI ─────────────────────────────────────────────
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

endlocal
