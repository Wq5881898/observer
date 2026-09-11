$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = './.venv/Scripts/python.exe'
$model = 'models/yolo/observer-phone-s304-e2b.onnx'

& $python -m observer validate-away-events `
    --models-dir models `
    --profile data/calibration/default.json `
    --role tuning `
    --suite full `
    --output-root data/formal/c-closeout/away-events-yolo `
    --detector-backend yolo `
    --yolo-model $model
if ($LASTEXITCODE -ne 0) {
    throw "Away event validation exited with code $LASTEXITCODE; review its report before continuing."
}

& $python -m observer validate-phone-events `
    --models-dir models `
    --profile data/calibration/default.json `
    --role tuning `
    --suite positive `
    --output-root data/formal/c-closeout/phone-events-yolo `
    --detector-backend yolo `
    --yolo-model $model `
    --phone-score-threshold 0.25 `
    --phone-retain-seconds 2.0 `
    --phone-center-distance 0.18
if ($LASTEXITCODE -ne 0) {
    throw "Phone event validation exited with code $LASTEXITCODE; keep both reports for review."
}
