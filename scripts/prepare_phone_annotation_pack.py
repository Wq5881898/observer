from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


SCENARIO_ORDER = (
    "phone_edge_e01_bottom_center",
    "phone_edge_e01_bottom_center_recovery",
    "phone_edge_e02_bottom_left",
    "phone_edge_e02_bottom_left_recovery",
    "phone_edge_e03_bottom_right",
    "phone_edge_e03_bottom_right_recovery",
    "phone_edge_e04_top_edge",
    "phone_edge_e04_top_edge_recovery",
)


def _scenario_from_name(name: str) -> str | None:
    for scenario in SCENARIO_ORDER:
        if name.startswith(scenario + "-"):
            return scenario
    return None


def _run_name(video: Path) -> str:
    for part in video.parts:
        if part.startswith("P0-"):
            return "P0"
        if part.startswith("P1-"):
            return "P1"
    return "unknown"


def _timestamps_for(video: Path) -> list[float] | None:
    sidecar = video.with_suffix(video.suffix + ".timestamps.json")
    if not sidecar.exists():
        return None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    values = payload.get("frame_elapsed_seconds")
    return [float(value) for value in values] if values else None


def _safe_stem(run: str, scenario: str, video: Path) -> str:
    video_id = video.stem.rsplit("-", 1)[-1]
    return f"{run.lower()}_{scenario}_{video_id}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a frame-sampled, model-blind phone annotation pack"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/formal/phone-edge"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/annotations/yolo-phone-gt-20260905"),
    )
    parser.add_argument("--every-n-frames", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    if args.every_n_frames < 1:
        raise SystemExit("--every-n-frames must be >= 1")

    image_dir = args.output_root / "images"
    labels_dir = args.output_root / "labels-voc"
    metadata_dir = args.output_root / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    videos: list[tuple[str, Path]] = []
    for video in sorted(args.source_root.rglob("*.mp4")):
        scenario = _scenario_from_name(video.name)
        if scenario is not None:
            videos.append((scenario, video))
    if not videos:
        raise SystemExit(f"No phone-edge trial videos found under {args.source_root}")

    images: list[dict[str, object]] = []
    source_videos: list[dict[str, object]] = []
    for scenario, video in videos:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {video}")
        timestamps = _timestamps_for(video)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        run = _run_name(video)
        stem = _safe_stem(run, scenario, video)
        sampled_indices = set(range(0, max(frame_count, 1), args.every_n_frames))
        if frame_count:
            sampled_indices.add(frame_count - 1)
        saved_count = 0
        try:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if frame_index in sampled_indices:
                    image_name = f"{stem}_frame-{frame_index:06d}.jpg"
                    output = image_dir / image_name
                    if not output.exists():
                        ok_write, encoded = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                        )
                        if not ok_write:
                            raise RuntimeError(f"could not encode frame: {output}")
                        output.write_bytes(encoded.tobytes())
                    timestamp = (
                        timestamps[frame_index]
                        if timestamps is not None and frame_index < len(timestamps)
                        else (frame_index / fps if fps > 0 else None)
                    )
                    images.append(
                        {
                            "file_name": f"images/{image_name}",
                            "width": int(frame.shape[1]),
                            "height": int(frame.shape[0]),
                            "source_run": run,
                            "scenario": scenario,
                            "source_video": str(video).replace("\\", "/"),
                            "frame_index": frame_index,
                            "timestamp_seconds": timestamp,
                            "expected_scene": "recovery" if scenario.endswith("_recovery") else "phone_candidate",
                            "annotation_status": "pending",
                        }
                    )
                    saved_count += 1
                frame_index += 1
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
                "sampled_frames": saved_count,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_date": "2026-09-05",
        "purpose": "Model-blind ground-truth annotation for YOLO phone edge diagnostics",
        "sampling": {
            "every_n_frames": args.every_n_frames,
            "includes_last_frame": True,
            "source_videos": len(source_videos),
        },
        "labels": [
            {
                "name": "cell_phone",
                "meaning": "Clearly identifiable phone; draw a tight box around the visible phone area.",
            },
            {
                "name": "cell_phone_ignore",
                "meaning": "Possibly a phone but too ambiguous or occluded for primary metrics; exclude from primary scoring.",
            },
        ],
        "annotation_rules": {
            "no_phone": "Leave the image without boxes and save it.",
            "occlusion": "Do not guess hidden pixels; box only the visible phone extent.",
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
    print(json.dumps({"output_root": str(args.output_root.resolve()), "images": len(images), "videos": len(source_videos)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
