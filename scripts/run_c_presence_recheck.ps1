$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
& ./.venv/Scripts/python.exe -m observer validate-live --models-dir models --profile data/calibration/default.json --role tuning --signals away --scenarios presence_yolo_01_seated,presence_yolo_02_face_hidden,presence_yolo_03_empty,presence_yolo_04_return --output-root data/formal/c-closeout/presence-yolo --debug-trial-videos --detector-backend yolo --yolo-model models/yolo/observer-phone-s304-e2b.onnx
if ($LASTEXITCODE -ne 0) { throw "Observer exited with code $LASTEXITCODE" }
