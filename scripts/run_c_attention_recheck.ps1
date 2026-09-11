$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$scenarios = 'attention_compensation_d01_center,attention_compensation_d02_head_left_eyes_center,attention_compensation_d03_outside_left,attention_compensation_d04_recovery,attention_compensation_d05_head_right_eyes_center,attention_compensation_d06_outside_right,attention_compensation_d07_recovery'
& ./.venv/Scripts/python.exe -m observer validate-live --models-dir models --profile data/calibration/default.json --role tuning --signals attention_away --scenarios $scenarios --output-root data/formal/c-closeout/attention-coarse --debug-trial-videos --detector-backend yolo --yolo-model models/yolo/observer-phone-s304-e2b.onnx
if ($LASTEXITCODE -ne 0) { throw "Observer exited with code $LASTEXITCODE; keep the report and review before further testing." }
