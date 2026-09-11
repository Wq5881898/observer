"""Paced replay of face, pose, object and fusion; excludes UI/camera/encoding."""
import argparse
import json
import time
from pathlib import Path
import cv2
import onnxruntime as ort
import psutil
from observer.config import load_config
from observer.models import verify_model_directory
from observer.object_detection import YoloOnnxBackend
from observer.vision import MediapipeVisionProbe


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--model', required=True)
    args = p.parse_args()
    models = verify_model_directory('models')
    rates = load_config().inference
    selected = {'phone_edge_e01_bottom_center', 'phone_contrast_f01_powerbank_center'}
    videos = [json.loads(s) for s in (args.batch/'trial-videos.jsonl').read_text().splitlines()
              if json.loads(s)['scenario_id'] in selected]
    results = []
    for spinning in (True, False):
        for video in videos:
            cap = cv2.VideoCapture(video['path'])
            times = json.loads(Path(video['path']+'.timestamps.json').read_text())['frame_elapsed_seconds']
            frames = []
            for t in times:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Missing video frame')
                frames.append((t, frame))
            cap.release()
            opts = ort.SessionOptions()
            if not spinning:
                opts.add_session_config_entry('session.intra_op.allow_spinning','0')
                opts.add_session_config_entry('session.inter_op.allow_spinning','0')
            session = ort.InferenceSession(args.model, sess_options=opts, providers=['CPUExecutionProvider'])
            backend = YoloOnnxBackend(args.model, session=session, single_class_name='cell_phone')
            probe = MediapipeVisionProbe(models, rates, object_backend=backend)
            process = psutil.Process()
            cpu0 = process.cpu_times()
            start = time.perf_counter()
            states, lateness = [], []
            try:
                for t, frame in frames:
                    time.sleep(max(0, start+t-time.perf_counter()))
                    lateness.append(max(0, time.perf_counter()-start-t))
                    probe.process_due(frame, 1000+t)
                    states.append(probe.observation(1000+t).signal('phone_held').state.value)
                elapsed = time.perf_counter()-start
                cpu1 = process.cpu_times()
                cpu = cpu1.user+cpu1.system-cpu0.user-cpu0.system
                row = dict(spinning=spinning, scenario=video['scenario_id'], elapsed=elapsed,
                           cpu_seconds=cpu, cpu_percent=cpu/elapsed/psutil.cpu_count()*100,
                           max_lateness=max(lateness), states=states, measurements=probe.summary())
                results.append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('states','measurements')}),flush=True)
            finally:
                probe.close()
                del probe, backend, session
    (args.batch/'probe-thread-benchmark.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
