from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract exact elapsed-time frames from an Observer debug video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    timestamp_path = args.video.with_suffix(args.video.suffix + ".timestamps.json")
    timestamps = json.loads(timestamp_path.read_text(encoding="utf-8"))["frame_elapsed_seconds"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")
    try:
        for requested in args.times:
            frame_index = min(
                range(len(timestamps)),
                key=lambda index: abs(timestamps[index] - requested),
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"could not read frame {frame_index}")
            actual = timestamps[frame_index]
            output = args.output_dir / f"t-{actual:08.3f}-frame-{frame_index:05d}.jpg"
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError(f"could not write frame: {output}")
            print(output.resolve())
    finally:
        capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
