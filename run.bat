@echo off
echo Starting HandScroller...
echo.
cd /d "%~dp0"
python main.py %*
if errorlevel 1 (
    echo.
    echo HandScroller exited with an error.
    echo Run install.bat first if you haven't installed dependencies.
    pause
)
