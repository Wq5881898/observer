# YOLO models

This directory is intentionally separate from the production EfficientDet
files in `models/`. Place the approved COCO-trained YOLO11n ONNX export here
as `yolo11n.onnx` after recording its source, export settings, license, and
SHA-256 in the evaluation report.

Readiness check:

```powershell
.\.venv\Scripts\python.exe -m observer yolo-doctor --model models/yolo/yolo11n.onnx
```

`yolo11n.onnx` is the untouched COCO reference model used for baseline
comparisons. `observer-phone-s304-e2b.onnx` is the Stage C frozen phone
candidate. It is copied byte-for-byte from:

`data/training/runs/yolo11n-phone-s304-contrast-e2b-nowarm-20260909/weights/epoch1.onnx`

Frozen candidate SHA-256:

`C4A8A791E227780B334E075D671D4347ED9D5FEE7474D6F017659366709C486F`

The candidate remains a single-class `cell_phone` detector. Observer runs the
existing EfficientDet person-only probe separately when this model is active.
Its accepted scope and known edge limitations are recorded in
`docs/phase-c-closeout-audit-20260909.md`.
