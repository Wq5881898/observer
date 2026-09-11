from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def sample_indices(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0 or count <= 0:
        raise ValueError("frame_count and count must be positive")
    count = min(frame_count, count)
    if count == 1:
        return [frame_count // 2]
    return sorted(
        {
            round((0.05 + 0.90 * index / (count - 1)) * (frame_count - 1))
            for index in range(count)
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a reviewed negative-only YOLO phone replay pack"
    )
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-video", type=int, default=12)
    parser.add_argument("--source-run", default="HARD_NEG_TRAIN")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"output already exists: {output_root}")
    image_root = output_root / "images"
    label_root = output_root / "labels-voc"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)

    records: list[dict[str, object]] = []
    for video_arg in args.video:
        video = video_arg.resolve()
        if not video.is_file():
            raise SystemExit(f"video not found: {video}")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise SystemExit(f"could not open video: {video}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        video_id = video.stem.rsplit("-", 1)[-1][:12]
        try:
            for frame_index in sample_indices(frame_count, args.samples_per_video):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"could not read frame {frame_index}: {video}")
                file_name = (
                    f"hardneg_{video_id}_{video.stem.rsplit('-', 1)[0]}_"
                    f"frame-{frame_index:06d}.jpg"
                )
                image_path = image_root / file_name
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(f"could not write image: {image_path}")
                (label_root / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
                records.append(
                    {
                        "file_name": file_name,
                        "source_run": args.source_run,
                        "scenario": "phone_baseline_b03_non_phone_object",
                        "source_video": video.as_posix(),
                        "frame_index": frame_index,
                        "width": width,
                        "height": height,
                        "objects": 0,
                        "review_status": "controlled_scenario_negative",
                    }
                )
        finally:
            capture.release()

    manifest = {
        "schema_version": 1,
        "purpose": "phone hard negatives: controlled hand-held non-phone object",
        "negative_only": True,
        "source_run": args.source_run,
        "sample_policy": {
            "method": "uniform_interior",
            "samples_per_video": args.samples_per_video,
            "range_fraction": [0.05, 0.95],
        },
        "images": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# YOLO phone hard negatives\n\n"
        "Controlled B03 clips contain a hand-held non-phone object. All labels are intentionally empty.\n"
        "An independent B03 clip must remain outside training for the regression gate.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output_root": str(output_root), "images": len(records)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
