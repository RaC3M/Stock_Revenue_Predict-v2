$ErrorActionPreference = "Stop"
$systemDir = $PSScriptRoot
$python = Join-Path $systemDir "..\sarima_forecast\.venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "SARIMA virtual environment not found."
}

Set-Location $systemDir
& $python start_app.py
