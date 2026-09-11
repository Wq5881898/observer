"""Compare pre-optimization decoding/spinning on identical P8 saved frames."""
import inspect
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import observer.object_detection as detection


def main():
    batch = Path('data/formal/phone-contrast/P8-s304-performance-manual/tuning-20260909T220554Z-584d4c1c')
    model = 'data/training/runs/yolo11n-phone-s304-contrast-e2b-nowarm-20260909/weights/epoch1.onnx'
    video = json.loads((batch/'trial-videos.jsonl').read_text().splitlines()[0])
    source = textwrap.dedent(inspect.getsource(detection.YoloOnnxBackend._decode))
    start = source.index('    if array.shape[1] == 5:')
    end = source.index('    for row in array:', start)
    namespace = dict(vars(detection))
    exec(source[:start] + source[end:], namespace)
    legacy_decode = namespace['_decode']
    sessions = []
    for spinning in (True, False):
        opts = ort.SessionOptions()
        if not spinning:
            opts.add_session_config_entry('session.intra_op.allow_spinning', '0')
            opts.add_session_config_entry('session.inter_op.allow_spinning', '0')
        sessions.append(ort.InferenceSession(model, sess_options=opts, providers=['CPUExecutionProvider']))
    backend = detection.YoloOnnxBackend(model, session=sessions[1], single_class_name='cell_phone')
    times = json.loads(Path(video['path']+'.timestamps.json').read_text())['frame_elapsed_seconds']
    cap = cv2.VideoCapture(video['path'])
    rows = []
    try:
        for i, t in enumerate(times):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Missing frame {i}')
            blob, scale, px, py = detection._letterbox(frame, 640)
            outputs = [s.run(None, {s.get_inputs()[0].name: blob})[0] for s in sessions]
            args = (frame.shape[1], frame.shape[0], scale, px, py)
            old = legacy_decode(backend, outputs[0], *args)
            new = backend._decode(outputs[1], *args)
            rows.append(dict(frame=i, seconds=t, equal=old == new,
                             max_raw_difference=float(np.max(np.abs(outputs[0]-outputs[1]))),
                             max_phone_score=float(np.max(outputs[1][0, 4])),
                             detections=len(new)))
            if i % 40 == 0:
                print(f'{i}/{len(times)} frames', flush=True)
    finally:
        cap.release()
    report = dict(model=model, frames=len(rows), equal_frames=sum(r['equal'] for r in rows),
                  detected_frames=sum(r['detections'] > 0 for r in rows), rows=rows,
                  limitation='Compressed scored video only; excludes preparation, live scheduling and tracker state.')
    (batch/'decoder-ab-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
