from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import cv2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _maximum_phone_confidence(frame_record: dict[str, object]) -> float:
    detections = frame_record.get("detections", [])
    if not isinstance(detections, list):
        return 0.0
    scores = [
        float(item.get("confidence", 0.0))
        for item in detections
        if isinstance(item, dict) and item.get("class_name") == "cell_phone"
    ]
    return max(scores, default=0.0)


def _uniform_candidates(
    candidates: list[dict[str, object]], sample_count: int
) -> list[dict[str, object]]:
    if len(candidates) < sample_count:
        raise ValueError(
            f"only {len(candidates)} low-confidence frames available for {sample_count} samples"
        )
    if sample_count == 1:
        return [candidates[len(candidates) // 2]]
    positions = [
        round(index * (len(candidates) - 1) / (sample_count - 1))
        for index in range(sample_count)
    ]
    return [candidates[position] for position in positions]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a compact manual-label pack from low-confidence YOLO frames"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--interior-start", type=float, default=0.05)
    parser.add_argument("--interior-end", type=float, default=0.95)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--source-run", default="B06_TRAIN")
    parser.add_argument(
        "--purpose",
        default="Lightweight B06 hard-positive patch for low-confidence phone frames",
    )
    parser.add_argument("--stem-prefix", default="b06_lowconf")
    args = parser.parse_args()

    video = args.video.resolve()
    replay_report = args.replay_report.resolve()
    output_root = args.output_root.resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    if not replay_report.is_file():
        raise SystemExit(f"replay report not found: {replay_report}")
    if output_root.exists():
        raise SystemExit(f"output already exists: {output_root}")
    if args.samples < 2:
        raise SystemExit("--samples must be at least 2")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if not 0.0 <= args.interior_start < args.interior_end <= 1.0:
        raise SystemExit("interior range must satisfy 0 <= start < end <= 1")

    replay = json.loads(replay_report.read_text(encoding="utf-8"))
    frames = replay.get("frames")
    if not isinstance(frames, list) or not frames:
        raise SystemExit("replay report contains no frames")
    maximum_index = max(int(item["frame_index"]) for item in frames)
    lower = round(maximum_index * args.interior_start)
    upper = round(maximum_index * args.interior_end)
    candidates = [
        item
        for item in frames
        if lower <= int(item["frame_index"]) <= upper
        and _maximum_phone_confidence(item) < args.threshold
    ]
    selected = _uniform_candidates(candidates, args.samples)

    image_root = output_root / "images"
    label_root = output_root / "labels-voc"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    records: list[dict[str, object]] = []
    try:
        for item in selected:
            frame_index = int(item["frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"could not read frame {frame_index}")
            stem = f"{args.stem_prefix}_frame-{frame_index:06d}"
            image_path = image_root / f"{stem}.jpg"
            encoded_ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            if not encoded_ok:
                raise RuntimeError(f"could not encode {image_path}")
            image_path.write_bytes(encoded.tobytes())
            records.append(
                {
                    "file_name": f"images/{image_path.name}",
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "source_video": video.as_posix(),
                    "frame_index": frame_index,
                    "timestamp_seconds": (
                        frame_index / fps if fps > 0.0 else item.get("timestamp_seconds")
                    ),
                    "maximum_phone_confidence": _maximum_phone_confidence(item),
                    "selection_threshold": args.threshold,
                    "expected_scene": "visible_phone_hard_positive",
                    "annotation_status": "pending_manual_yolo",
                }
            )
    finally:
        capture.release()

    manifest = {
        "schema_version": 1,
        "created_date": date.today().isoformat(),
        "purpose": args.purpose,
        "source_run": args.source_run.upper(),
        "source_video": video.as_posix(),
        "source_video_sha256": _sha256(video),
        "replay_report": replay_report.as_posix(),
        "replay_report_sha256": _sha256(replay_report),
        "selection": {
            "samples": len(records),
            "maximum_phone_confidence_below": args.threshold,
            "interior_fraction": [args.interior_start, args.interior_end],
            "method": "uniform quantiles over low-confidence candidate frames",
        },
        "labels": [{"name": "cell_phone", "meaning": "Visible phone extent"}],
        "annotation_rules": {
            "phone": "Draw one cell_phone box around only the visible phone extent.",
            "occlusion": "Do not infer hidden pixels behind either hand.",
            "boundary": "If the visible phone touches an image edge, let the box touch it.",
        },
        "images": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "classes.txt").write_text("cell_phone\n", encoding="utf-8")
    (output_root / "README.md").write_text(
        "# YOLO 手机低置信困难正例标注包\n\n"
        f"本包包含当前 YOLO 置信度低于 {args.threshold:.2f} 的 {len(records)} 张代表帧。\n"
        "使用 LabelImg 的 YOLO 格式，在每张图中仅标注手机可见部分，标签固定为 `cell_phone`。\n"
        "两只手遮挡的区域不要猜测；框不包含手。全部标注完成后再进行轻量微调。\n"
        "本段录像在加入训练后不得再作为独立验收证据；后续重新拍摄 B06 作为 holdout。\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "images": len(records),
                "candidate_frames": len(candidates),
                "selected_indices": [record["frame_index"] for record in records],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
