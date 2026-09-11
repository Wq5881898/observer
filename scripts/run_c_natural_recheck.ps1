param([ValidateRange(30,3600)][int]$Seconds = 300, [switch]$RecordVideo)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$run = Join-Path 'data/natural/c-closeout' ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8))
$arguments = @('-m', 'observer', 'vision-smoke', '--seconds', "$Seconds",
    '--models-dir', 'models', '--profile', 'data/calibration/default.json',
    '--detector-backend', 'yolo', '--yolo-model',
    'models/yolo/observer-phone-s304-e2b.onnx',
    '--report', "$run/report.json", '--evidence-dir', "$run/evidence")
if ($RecordVideo) { $arguments += @('--debug-video', "$run/session.mp4") }
& ./.venv/Scripts/python.exe @arguments
if ($LASTEXITCODE -ne 0) { throw "Observer exited with code $LASTEXITCODE" }
