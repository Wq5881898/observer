"""Compare path/buffer loading of one unchanged model without a camera."""
from pathlib import Path
import json
import os
import shutil

root = Path(__file__).resolve().parents[1]
cache = root/'build/phase-f-mp-unicode/mpl-cache'
cache.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(cache)
import mediapipe as mp
from mediapipe.tasks.python import vision

ascii_path = root/'models/face_landmarker.task'
unicode_path = root/'build/phase-f-mp-unicode/中文 模型/face_landmarker.task'
unicode_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(ascii_path, unicode_path)
results = []
for label, kwargs in [('ascii-path', {'model_asset_path': str(ascii_path)}),
                      ('unicode-path', {'model_asset_path': str(unicode_path)}),
                      ('unicode-buffer', {'model_asset_buffer': unicode_path.read_bytes()})]:
    try:
        with vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(**kwargs))) as detector:
            results.append({'case': label, 'status': 'passed'})
    except Exception as exc:
        results.append({'case': label, 'status': 'failed', 'error': str(exc)})
(root/'artifacts/phase-f/mediapipe-unicode-probe.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(results, ensure_ascii=False))
