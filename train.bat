@echo off
echo ========================================================
echo    🖐️  HandScroller Training Station
echo    Record your hand movements to tune the perfect code
echo ========================================================
echo.
cd /d "%~dp0"
python training_station.py %*
if errorlevel 1 (
    echo.
    echo Training Station exited with an error.
    echo Make sure python and dependencies are installed.
    pause
)
