$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$run = Join-Path 'data/formal/c-closeout/minimal-holdout' ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8))
$video = Join-Path $run 'session.mp4'
$trace = "$video.trace.jsonl"
$report = Join-Path $run 'report.json'
$audit = Join-Path $run 'holdout-audit.json'
$evidence = Join-Path $run 'evidence'

& ./.venv/Scripts/python.exe -m observer vision-smoke `
    --models-dir models `
    --profile data/calibration/default.json `
    --detector-backend yolo `
    --yolo-model models/yolo/observer-phone-s304-e2b.onnx `
    --guided-debug-plan c-closeout-holdout `
    --debug-video $video `
    --evidence-dir $evidence `
    --report $report
if ($LASTEXITCODE -ne 0) {
    throw "Minimal holdout capture exited with code $LASTEXITCODE"
}

& ./.venv/Scripts/python.exe scripts/audit_c_closeout_holdout.py `
    --trace $trace `
    --output $audit
$auditExitCode = $LASTEXITCODE

Write-Output (@{
    batch_directory = (Resolve-Path $run).Path
    report = (Resolve-Path $report).Path
    audit = (Resolve-Path $audit).Path
    temporary_video = (Resolve-Path $video).Path
    automated_checks_passed = ($auditExitCode -eq 0)
    manual_truth_confirmation_required = $true
} | ConvertTo-Json)

if ($auditExitCode -ne 0) {
    exit $auditExitCode
}
