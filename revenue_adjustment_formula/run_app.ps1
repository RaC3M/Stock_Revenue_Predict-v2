$ErrorActionPreference = "Stop"
$systemDir = $PSScriptRoot
$python = Join-Path $systemDir "..\sarima_forecast\.venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

Set-Location $systemDir
& $python start_app.py
