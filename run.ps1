# HandScroller PowerShell Launcher
Write-Host "🖐️ Starting HandScroller..." -ForegroundColor Cyan
Set-Location $PSScriptRoot
python main.py $args
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nHandScroller exited with an error (code: $LASTEXITCODE)." -ForegroundColor Red
    Write-Host "Make sure Python is installed and dependencies are satisfied." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}
