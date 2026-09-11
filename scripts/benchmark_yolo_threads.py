"""Compare CPU thread policies on identical saved frames without camera access."""
import argparse
import json
import time
from pathlib import Path

import cv2
import onnxruntime as ort
import psutil

from observer.object_detection import YoloOnnxBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    frames = [cv2.imread(p) for p in args.images]
    if any(f is None for f in frames):
        raise ValueError('Unreadable image')
    process = psutil.Process()
    results = []
    for threads in (0, -1, 1, 2):
        options = ort.SessionOptions()
        if threads > 0:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        if threads != 0:
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        session = ort.InferenceSession(args.model, sess_options=options,
                                       providers=['CPUExecutionProvider'])
        backend = YoloOnnxBackend(args.model, session=session, single_class_name='cell_phone')
        backend.detect(frames[0], 0)
        cpu_start = process.cpu_times()
        start = time.perf_counter()
        detections = []
        for repeat in range(4):
            for frame in frames:
                detections.append([(d.class_name, round(d.confidence, 5))
                                   for d in backend.detect(frame, 0)])
                time.sleep(0.1)
        elapsed = time.perf_counter() - start
        cpu_end = process.cpu_times()
        cpu_seconds = cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system
        row = dict(threads=threads, elapsed_seconds=elapsed,
                   cpu_seconds=cpu_seconds,
                   cpu_percent=100 * cpu_seconds / elapsed / psutil.cpu_count(),
                   detections=detections)
        results.append(row)
        print(json.dumps(row), flush=True)
        backend.close()
        del backend, session
    Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
