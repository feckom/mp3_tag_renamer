@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ---- Config ----
set VENV_DIR=.venv
set PYTHON_EXE=python

REM ---- Create venv if missing ----
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Creating venv...
    %PYTHON_EXE% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERR ] Failed to create virtualenv.
        exit /b 1
    )
)

REM ---- Activate venv ----
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERR ] Failed to activate venv.
    exit /b 1
)

REM ---- Upgrade pip ----
python -m pip install --upgrade pip

REM ---- Install deps ----
REM shazamio uses aiohttp; mutagen for ID3; requests not required (we use aiohttp)
pip install "shazamio>=0.6" "mutagen>=1.47" "aiohttp>=3.9" "mutagen" "pip-system-certs; platform_system=='Windows'"

REM ---- Optional: having ffmpeg on PATH improves robustness (not installed here)
echo [INFO] If recognition fails often, consider installing ffmpeg and adding it to PATH.

REM ---- Run script (DRY-RUN, verbose) ----
python mp3_tag_renamer.py --apply
echo.
echo [INFO] Dry-run complete. To actually write/rename/lyrics, run:
echo        mp3_tag_renamer.py --apply
echo        (add --recursive for subfolders, --force to overwrite tags/lyrics)

endlocal
