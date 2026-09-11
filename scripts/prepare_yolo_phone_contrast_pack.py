from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import cv2


PHONE_SCENARIOS = (
    "phone_contrast_f02_phone_center",
    "phone_contrast_f05_phone_left",
    "phone_contrast_f08_phone_right",
)
POWERBANK_SCENARIOS = (
    "phone_contrast_f01_powerbank_center",
    "phone_contrast_f04_powerbank_left",
    "phone_contrast_f07_powerbank_right",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sample_indices(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0 or count <= 0:
        raise ValueError("frame_count and count must be positive")
    count = min(frame_count, count)
    if count == 1:
        return [frame_count // 2]
    return sorted(
        {
            round((0.10 + 0.80 * index / (count - 1)) * (frame_count - 1))
            for index in range(count)
        }
    )


def find_video(video_root: Path, scenario: str) -> Path:
    matches = sorted(video_root.glob(f"{scenario}-*.mp4"))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one video for {scenario}, found {len(matches)}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare paired phone/power-bank YOLO annotations from an F batch"
    )
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-video", type=int, default=8)
    parser.add_argument("--source-run", default="CONTRAST_TRAIN")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    batch_dir = args.batch_dir.resolve()
    video_root = batch_dir / "trial-videos"
    output_root = args.output_root.resolve()
    if not video_root.is_dir():
        raise SystemExit(f"trial video directory not found: {video_root}")
    if output_root.exists():
        raise SystemExit(f"output already exists: {output_root}")
    if args.samples_per_video < 2:
        raise SystemExit("--samples-per-video must be at least 2")

    image_root = output_root / "images"
    phone_pending_root = output_root / "images-phone-pending"
    label_root = output_root / "labels-voc"
    image_root.mkdir(parents=True)
    phone_pending_root.mkdir(parents=True)
    label_root.mkdir(parents=True)

    records: list[dict[str, object]] = []
    videos: list[dict[str, object]] = []
    scenarios = POWERBANK_SCENARIOS + PHONE_SCENARIOS
    for scenario in scenarios:
        video = find_video(video_root, scenario)
        is_phone = scenario in PHONE_SCENARIOS
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise SystemExit(f"could not open video: {video}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        indices = sample_indices(frame_count, args.samples_per_video)
        videos.append(
            {
                "scenario": scenario,
                "source_video": video.as_posix(),
                "source_video_sha256": sha256(video),
                "expected_scene": "visible_phone" if is_phone else "power_bank_no_phone",
                "selected_indices": indices,
            }
        )
        try:
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"could not read frame {frame_index}: {video}")
                stem = f"{scenario}_frame-{frame_index:06d}"
                image_path = image_root / f"{stem}.jpg"
                encoded_ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                )
                if not encoded_ok:
                    raise RuntimeError(f"could not encode image: {image_path}")
                image_path.write_bytes(encoded.tobytes())
                if is_phone:
                    (phone_pending_root / image_path.name).write_bytes(encoded.tobytes())
                # Controlled power-bank frames are reviewed empty negatives.
                # Phone files start empty only to make LabelImg navigation and
                # saving predictable; audit must reject them until boxed.
                (label_root / f"{stem}.txt").write_text("", encoding="utf-8")
                records.append(
                    {
                        "file_name": image_path.name,
                        "source_run": args.source_run.upper(),
                        "scenario": scenario,
                        "source_video": video.as_posix(),
                        "frame_index": frame_index,
                        "timestamp_seconds": frame_index / fps if fps > 0 else None,
                        "width": width,
                        "height": height,
                        "objects": 1 if is_phone else 0,
                        "expected_scene": (
                            "visible_phone" if is_phone else "power_bank_no_phone"
                        ),
                        "annotation_status": (
                            "pending_manual_yolo"
                            if is_phone
                            else "controlled_scenario_negative"
                        ),
                    }
                )
        finally:
            capture.release()

    manifest = {
        "schema_version": 1,
        "created_date": date.today().isoformat(),
        "purpose": "paired phone versus power-bank hard-negative fine-tuning",
        "source_run": args.source_run.upper(),
        "source_batch": batch_dir.as_posix(),
        "sample_policy": {
            "method": "uniform_interior",
            "samples_per_video": args.samples_per_video,
            "range_fraction": [0.10, 0.90],
        },
        "labels": [{"name": "cell_phone", "meaning": "Visible phone extent"}],
        "annotation_rules": {
            "phone": "Draw one cell_phone box around only the visible phone.",
            "power_bank": "Leave empty; the controlled scenario contains no phone.",
            "occlusion": "Do not infer pixels hidden by the hand.",
        },
        "videos": videos,
        "images": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "classes.txt").write_text("cell_phone\n", encoding="utf-8")
    # LabelImg resolves classes.txt relative to the YOLO label directory when
    # a matching .txt annotation already exists (including an empty negative).
    # Keep a copy there so opening the prepared pack cannot fail at startup.
    (label_root / "classes.txt").write_text("cell_phone\n", encoding="utf-8")
    (output_root / "README.md").write_text(
        "# 手机/充电宝成对标注包\n\n"
        "共 48 张：24 张手机正例、24 张充电宝受控负例。\n"
        "充电宝的空标签已经创建，不需要打开或保存。LabelImg 只需打开 "
        "images-phone-pending，其中只有需要画框的 24 张手机图片。\n"
        "标签固定为 cell_phone，框只覆盖手机实际可见部分，不包含手。\n"
        "所有手机图片必须恰好有一个框；训练前由脚本审计，空手机标签会被拒绝。\n"
        "本批进入训练后只可做拟合检查；P5 新充电宝录像继续作为独立 holdout。\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "images": len(records),
                "phone_pending": sum(
                    1 for record in records if record["expected_scene"] == "visible_phone"
                ),
                "power_bank_empty": sum(
                    1
                    for record in records
                    if record["expected_scene"] == "power_bank_no_phone"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
