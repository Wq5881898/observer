from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import cv2

from observer.object_detection import YoloOnnxBackend


POSITIVE_SCENARIOS = (
    "phone_baseline_b04_one_hand_phone",
)
NEGATIVE_SCENARIOS = (
    "phone_baseline_b05_recovery",
)


def _scenario_for(video: Path) -> str | None:
    for scenario in POSITIVE_SCENARIOS + NEGATIVE_SCENARIOS:
        if video.name.startswith(scenario + "-"):
            return scenario
    return None


def _sample_indices(frame_count: int, samples: int) -> list[int]:
    """Return interior temporal quantiles, avoiding transition-heavy endpoints."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if frame_count < samples:
        raise ValueError(f"video has only {frame_count} frames for {samples} samples")
    return [
        round((0.10 + 0.80 * index / max(samples - 1, 1)) * (frame_count - 1))
        for index in range(samples)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a reproducible complete-phone retention pack using the COCO YOLO teacher"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/formal/phone-baseline-postfix"),
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="source batch directory name; repeat for multiple whole sessions",
    )
    parser.add_argument("--teacher-model", type=Path, default=Path("models/yolo/yolo11n.onnx"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-video", type=int, default=6)
    parser.add_argument("--teacher-threshold", type=float, default=0.50)
    parser.add_argument(
        "--minimum-box-area",
        type=float,
        default=0.02,
        help="minimum normalized phone box area; rejects tiny teacher false positives",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the explicitly named generated output directory",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    teacher_model = args.teacher_model.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"source root not found: {source_root}")
    if not teacher_model.is_file():
        raise SystemExit(f"teacher model not found: {teacher_model}")
    if args.samples_per_video < 2:
        raise SystemExit("--samples-per-video must be at least 2")
    if not 0.0 <= args.teacher_threshold <= 1.0:
        raise SystemExit("--teacher-threshold must be between 0 and 1")
    if not 0.0 <= args.minimum_box_area <= 1.0:
        raise SystemExit("--minimum-box-area must be between 0 and 1")
    if output_root.exists():
        if not args.force:
            raise SystemExit(f"output already exists; pass --force to regenerate: {output_root}")
        shutil.rmtree(output_root)

    videos: list[tuple[str, str, Path]] = []
    for run in args.run:
        video_root = source_root / run / "trial-videos"
        if not video_root.is_dir():
            raise SystemExit(f"run video directory not found: {video_root}")
        found: dict[str, Path] = {}
        for video in sorted(video_root.glob("*.mp4")):
            scenario = _scenario_for(video)
            if scenario is not None:
                if scenario in found:
                    raise SystemExit(f"duplicate scenario {scenario} in {video_root}")
                found[scenario] = video
        expected = set(POSITIVE_SCENARIOS + NEGATIVE_SCENARIOS)
        missing = expected - set(found)
        if missing:
            raise SystemExit(f"run {run} is missing scenarios: {sorted(missing)}")
        videos.extend((run, scenario, found[scenario]) for scenario in sorted(expected))

    image_root = output_root / "images"
    label_root = output_root / "labels-voc"
    review_root = output_root / "review"
    for directory in (image_root, label_root, review_root):
        directory.mkdir(parents=True, exist_ok=True)

    teacher = YoloOnnxBackend(teacher_model, score_threshold=args.teacher_threshold)
    images: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    counts = {"positive": 0, "negative": 0}
    try:
        for run, scenario, video in videos:
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise RuntimeError(f"could not open video: {video}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            indices = _sample_indices(frame_count, args.samples_per_video)
            video_id = video.stem.rsplit("-", 1)[-1][:12]
            source_records = 0
            used_indices: set[int] = set()
            try:
                for target_index in indices:
                    positive = scenario in POSITIVE_SCENARIOS
                    offsets = [0]
                    offsets.extend(
                        offset
                        for distance in range(3, 31, 3)
                        for offset in (-distance, distance)
                    )
                    selected: tuple[int, object, list[object]] | None = None
                    for offset in offsets:
                        frame_index = target_index + offset
                        if (
                            frame_index < 0
                            or frame_index >= frame_count
                            or frame_index in used_indices
                        ):
                            continue
                        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ok, frame = capture.read()
                        if not ok or frame is None:
                            continue
                        detections = [
                            detection
                            for detection in teacher.detect(
                                frame,
                                round(frame_index / fps * 1000) if fps > 0 else frame_index,
                            )
                            if detection.class_name == "cell phone"
                            and detection.box.width * detection.box.height
                            >= args.minimum_box_area
                        ]
                        if positive and not detections:
                            continue
                        if not positive and detections:
                            continue
                        if positive:
                            # These controlled clips contain exactly one phone. Keeping only
                            # the strongest COCO phone box avoids copying rare duplicate or
                            # spurious teacher candidates into the one-class student set.
                            detections = [
                                max(detections, key=lambda item: item.confidence)
                            ]
                        selected = (frame_index, frame, detections)
                        break
                    if selected is None:
                        expected = "complete-phone" if positive else "phone-free recovery"
                        raise RuntimeError(
                            f"teacher found no nearby {expected} frame at threshold "
                            f"{args.teacher_threshold:.2f}: {video.name} target {target_index}"
                        )
                    frame_index, frame, detections = selected
                    used_indices.add(frame_index)
                    height, width = frame.shape[:2]

                    run_token = run.replace("tuning-", "").replace("Z-", "-")[:20]
                    stem = f"fulltrain_{run_token}_{scenario}_{video_id}_frame-{frame_index:06d}"
                    image_name = stem + ".jpg"
                    image_path = image_root / image_name
                    encoded_ok, encoded = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                    )
                    if not encoded_ok:
                        raise RuntimeError(f"could not encode {image_path}")
                    image_path.write_bytes(encoded.tobytes())

                    label_lines: list[str] = []
                    teacher_boxes: list[dict[str, float]] = []
                    review = frame.copy()
                    for detection in detections:
                        box = detection.box
                        center_x = box.x + box.width / 2.0
                        center_y = box.y + box.height / 2.0
                        label_lines.append(
                            f"0 {center_x:.8f} {center_y:.8f} {box.width:.8f} {box.height:.8f}"
                        )
                        teacher_boxes.append(
                            {
                                "x": box.x,
                                "y": box.y,
                                "width": box.width,
                                "height": box.height,
                                "confidence": detection.confidence,
                            }
                        )
                        x1, y1 = int(box.x * width), int(box.y * height)
                        x2 = int((box.x + box.width) * width)
                        y2 = int((box.y + box.height) * height)
                        cv2.rectangle(review, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            review,
                            f"cell_phone {detection.confidence:.2f}",
                            (max(0, x1), max(18, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    (label_root / (stem + ".txt")).write_text(
                        "\n".join(label_lines) + ("\n" if label_lines else ""),
                        encoding="utf-8",
                    )
                    cv2.imwrite(str(review_root / image_name), review)

                    counts["positive" if positive else "negative"] += 1
                    source_records += 1
                    images.append(
                        {
                            "file_name": f"images/{image_name}",
                            "width": width,
                            "height": height,
                            "source_run": "FULL_TRAIN",
                            "source_batch": run,
                            "scenario": scenario,
                            "source_video": video.as_posix(),
                            "frame_index": frame_index,
                            "target_frame_index": target_index,
                            "timestamp_seconds": frame_index / fps if fps > 0 else None,
                            "expected_scene": "phone_candidate" if positive else "recovery",
                            "annotation_status": "teacher_pseudo_label" if positive else "teacher_verified_empty",
                            "teacher_boxes": teacher_boxes,
                        }
                    )
            finally:
                capture.release()
            sources.append(
                {
                    "source_batch": run,
                    "scenario": scenario,
                    "path": video.as_posix(),
                    "frame_count": frame_count,
                    "fps": fps,
                    "sampled_frames": source_records,
                }
            )
    finally:
        teacher.close()

    manifest = {
        "schema_version": 1,
        "created_date": date.today().isoformat(),
        "purpose": "Complete-phone retention replay for joint edge/full-phone fine-tuning",
        "source_run": "FULL_TRAIN",
        "teacher": {
            "model": teacher_model.as_posix(),
            "sha256": _sha256(teacher_model),
            "score_threshold": args.teacher_threshold,
            "minimum_box_area": args.minimum_box_area,
            "class": "COCO cell phone",
        },
        "sampling": {
            "selection": "interior temporal quantiles from 10% through 90%, with nearby teacher-confirmed substitution",
            "samples_per_video": args.samples_per_video,
            "source_videos": len(videos),
            "total_images": len(images),
        },
        "counts": counts,
        "source_videos": sources,
        "images": images,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "classes.txt").write_text("cell_phone\n", encoding="utf-8")
    (output_root / "README.md").write_text(
        "# YOLO 完整手机能力保留包\n\n"
        "本目录由历史正式测试录像生成，用于在学习边缘手机时保留完整手机能力。\n"
        "完整手机框来自原始 YOLO11n COCO 教师模型；恢复帧由同一教师检查为无手机。\n"
        "`review/` 保存带框复核图，`manifest.json` 保存每张图的来源、置信度和教师哈希。\n"
        "这些样本属于训练 replay，不得再作为独立泛化证据。\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "videos": len(videos),
                "images": len(images),
                "counts": counts,
                "teacher_sha256": manifest["teacher"]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
