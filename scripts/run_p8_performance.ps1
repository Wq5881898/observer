$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
$model = 'data/training/runs/yolo11n-phone-s304-contrast-e2b-nowarm-20260909/weights/epoch1.onnx'
if (!(Test-Path -LiteralPath $python) -or !(Test-Path -LiteralPath $model)) {
    throw 'Python environment or S304 model missing; run from the Observer project.'
}
$scenarios = 'phone_edge_e01_bottom_center,phone_edge_e01_bottom_center_recovery,phone_contrast_f00_clear,phone_contrast_f01_powerbank_center,phone_contrast_f02_phone_center,phone_contrast_f03_clear,phone_contrast_f04_powerbank_left,phone_contrast_f05_phone_left,phone_contrast_f06_clear,phone_contrast_f07_powerbank_right,phone_contrast_f08_phone_right,phone_contrast_f09_final_clear'
& $python -m observer validate-live --models-dir models --profile data/calibration/default.json --role tuning --signals phone_held --scenarios $scenarios --output-root data/formal/phone-contrast/P8-s304-performance-manual --debug-trial-videos --phone-score-threshold 0.25 --phone-retain-seconds 1.5 --phone-center-distance 0.18 --detector-backend yolo --yolo-model $model
if ($LASTEXITCODE -ne 0) { throw "Observer exited with code $LASTEXITCODE" }
