$ErrorActionPreference = "Stop"
$systemDir = $PSScriptRoot
$python = Join-Path $systemDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "SARIMA virtual environment not found. See README.md."
}

Set-Location $systemDir
& $python start_app.py

