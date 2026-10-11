param([string]$BuildId = (Get-Date -Format 'yyyyMMdd-HHmmss'))
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $env:UV_PROJECT_ENVIRONMENT = 'build/phase-f-env'
    uv sync --locked --extra desktop --extra yolo --python .venv/Scripts/python.exe --cache-dir .uv-cache
    if ($LASTEXITCODE -ne 0) { throw 'Locked runtime environment setup failed' }
    uv pip install --python build/phase-f-env/Scripts/python.exe --require-hashes -r packaging/requirements.lock --cache-dir .uv-cache
    if ($LASTEXITCODE -ne 0) { throw 'Locked build tools setup failed' }
    & build/phase-f-env/Scripts/python.exe scripts/build_windows.py --build-id $BuildId
    if ($LASTEXITCODE -ne 0) { throw 'Candidate build failed; see artifacts/phase-f' }
} finally {
    Pop-Location
}
