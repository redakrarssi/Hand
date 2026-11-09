@echo off
echo ══════════════════════════════════════════════════
echo   HandScroller - One-Click Setup
echo ══════════════════════════════════════════════════
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Install Python 3.9+ from python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [1/2] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo WARNING: Some packages failed to install.
    echo Try: pip install mediapipe opencv-python numpy
    pause
)

echo.
echo [2/2] Setup complete!
echo.
echo ══════════════════════════════════════════════════
echo   Run HandScroller with: python main.py
echo   Or double-click: run.bat
echo ══════════════════════════════════════════════════
echo.
pause
