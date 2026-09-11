from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import cv2


SCENARIOS = (
    "phone_edge_e01_bottom_center",
    "phone_edge_e01_bottom_center_recovery",
    "phone_edge_e02_bottom_left",
    "phone_edge_e02_bottom_left_recovery",
    "phone_edge_e03_bottom_right",
    "phone_edge_e03_bottom_right_recovery",
    "phone_edge_e04_top_edge",
    "phone_edge_e04_top_edge_recovery",
)


def scenario_for(video: Path) -> str | None:
    for scenario in SCENARIOS:
        if video.name.startswith(scenario + "-"):
            return scenario
    return None


def run_for(video: Path) -> str:
    for part in video.parts:
        match = re.match(r"^(P\d+)-", part, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return "unknown"


def timestamps_for(video: Path) -> list[float] | None:
    sidecar = video.with_suffix(video.suffix + ".timestamps.json")
    if not sidecar.exists():
        return None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    values = payload.get("frame_elapsed_seconds")
    return [float(value) for value in values] if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a compact phone annotation sample")
    parser.add_argument("--source-root", type=Path, default=Path("data/formal/phone-edge"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/annotations/yolo-phone-gt-20260905-s96"),
    )
    parser.add_argument("--samples-per-video", type=int, default=6)
    parser.add_argument(
        "--expected-videos",
        type=int,
        default=None,
        help="fail unless this many matching videos are found",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    if args.samples_per_video < 2:
        raise SystemExit("--samples-per-video must be >= 2")

    image_dir = args.output_root / "images"
    labels_dir = args.output_root / "labels-voc"
    metadata_dir = args.output_root / "metadata"
    for directory in (image_dir, labels_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    videos: list[tuple[str, Path]] = []
    for video in sorted(args.source_root.rglob("*.mp4")):
        scenario = scenario_for(video)
        if scenario is not None:
            videos.append((scenario, video))
    if not videos:
        raise SystemExit(f"No phone-edge videos found under {args.source_root}")
    if args.expected_videos is not None and len(videos) != args.expected_videos:
        raise SystemExit(
            f"Expected {args.expected_videos} phone-edge videos, found {len(videos)}"
        )

    source_videos: list[dict[str, object]] = []
    images: list[dict[str, object]] = []
    for scenario, video in videos:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {video}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        timestamps = timestamps_for(video)
        run = run_for(video)
        video_id = video.stem.rsplit("-", 1)[-1]
        stem = f"{run.lower()}_{scenario}_{video_id}"
        if frame_count <= 0:
            capture.release()
            raise RuntimeError(f"video has no frames: {video}")
        indices = {
            round(i * (frame_count - 1) / (args.samples_per_video - 1))
            for i in range(args.samples_per_video)
        }
        saved = 0
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if index in indices:
                    image_name = f"{stem}_frame-{index:06d}.jpg"
                    output = image_dir / image_name
                    ok_write, encoded = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                    )
                    if not ok_write:
                        raise RuntimeError(f"could not encode frame: {output}")
                    output.write_bytes(encoded.tobytes())
                    timestamp = (
                        timestamps[index]
                        if timestamps is not None and index < len(timestamps)
                        else (index / fps if fps > 0 else None)
                    )
                    images.append(
                        {
                            "file_name": f"images/{image_name}",
                            "width": int(frame.shape[1]),
                            "height": int(frame.shape[0]),
                            "source_run": run,
                            "scenario": scenario,
                            "source_video": str(video).replace("\\", "/"),
                            "frame_index": index,
                            "timestamp_seconds": timestamp,
                            "sample_percent": round(index / (frame_count - 1) * 100, 2),
                            "expected_scene": "recovery" if scenario.endswith("_recovery") else "phone_candidate",
                            "annotation_status": "pending",
                        }
                    )
                    saved += 1
                index += 1
        finally:
            capture.release()
        source_videos.append(
            {
                "run": run,
                "scenario": scenario,
                "path": str(video).replace("\\", "/"),
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "sampled_frames": saved,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_date": date.today().isoformat(),
        "purpose": "Compact model-blind ground-truth sample for YOLO phone edge screening",
        "sampling": {
            "samples_per_video": args.samples_per_video,
            "selection": "uniform temporal quantiles including first and last frame",
            "source_videos": len(source_videos),
            "total_images": len(images),
        },
        "labels": [
            {"name": "cell_phone", "meaning": "Clearly identifiable phone."},
            {"name": "cell_phone_ignore", "meaning": "Possibly a phone but too ambiguous for primary metrics."},
        ],
        "annotation_rules": {
            "no_phone": "Leave the image without boxes and save it.",
            "occlusion": "Do not guess hidden pixels; box only visible phone extent.",
            "truncation": "Let the box touch the image boundary when the phone is clipped by the frame.",
            "similar_objects": "Do not label remotes, calculators, cameras, hands, or other rectangular objects.",
        },
        "source_videos": source_videos,
        "images": images,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_root / "classes.txt").write_text(
        "cell_phone\ncell_phone_ignore\n", encoding="utf-8"
    )
    (args.output_root / "annotation-instructions.md").write_text(
        """# Observer 手机目标检测精简标注说明

只标注手机，不标注人、手或其他物体。明确有手机时画 `cell_phone`；
无法可靠确认时画 `cell_phone_ignore`；明确无手机时不画框，点击 Verify Image
勾号或按 Space 完成核验。仅按 Ctrl+S 在没有任何框时可能没有反应。

框只覆盖手机可见区域。被手遮挡时不要猜测隐藏区域；被图像边界截断时
框贴到图像边界。遥控器、计算器、摄像机和其他矩形物体不要标注。

LabelImg 中使用 PascalVOC 格式：Open Dir 选择 `images`，Change Save Dir
选择 `labels-voc`，用 W 或 Create RectBox 画框，Ctrl+S 保存，D 进入下一张。
不要裁剪、缩放、旋转原图，也不要参考模型预测框。
""",
        encoding="utf-8",
    )
    (args.output_root / "README.md").write_text(
        "# YOLO 手机边缘精简真值标注包\n\n"
        f"本包包含 {len(source_videos)} 段录像、每段 {args.samples_per_video} 张、"
        f"共 {len(images)} 张图片。标注完成后保留整个目录，Observer 将\n"
        "检查 Pascal VOC XML 并转换为 COCO/YOLO 格式。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(args.output_root.resolve()), "videos": len(source_videos), "images": len(images)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
