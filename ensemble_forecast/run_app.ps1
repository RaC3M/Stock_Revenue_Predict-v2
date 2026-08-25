$ErrorActionPreference = "Stop"
$systemDir = $PSScriptRoot
$python = Join-Path $systemDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "找不到 Ensemble 虛擬環境。請先執行 README.md 的環境建立指令。"
}

Set-Location $systemDir
& $python start_app.py
