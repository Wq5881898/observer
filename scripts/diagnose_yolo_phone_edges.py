from __future__ import annotations

import argparse
import json
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2

from observer.object_detection import YoloOnnxBackend


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def parse_xml(path: Path):
    root = ET.parse(path).getroot()
    filename = root.findtext("filename") or ""
    objects = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        coords = tuple(float(box.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
        objects.append((obj.findtext("name") or "", coords))
    return filename, objects


def pixel_box(detection, width: int, height: int) -> tuple[float, float, float, float]:
    box = detection.box
    return (box.x * width, box.y * height, (box.x + box.width) * width, (box.y + box.height) * height)


def draw_box(frame, box, color, label):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (max(0, x1), max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose YOLO phone edge confidence, boxes, and preprocessing")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-threshold", type=float, default=0.01)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--single-class-name",
        default=None,
        help="class name for a single-class YOLO export, e.g. cell_phone",
    )
    args = parser.parse_args()

    manifest = json.loads((args.annotations / "manifest.json").read_text(encoding="utf-8"))
    image_root = args.annotations / "images"
    xml_root = args.annotations / "labels-voc"
    image_records = {Path(str(x["file_name"])).name: x for x in manifest["images"]}
    truth = {}
    for xml_path in xml_root.glob("*.xml"):
        filename, objects = parse_xml(xml_path)
        truth[filename] = objects

    args.output_root.mkdir(parents=True, exist_ok=True)
    evidence_dir = args.output_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    backend = YoloOnnxBackend(
        args.model,
        score_threshold=args.candidate_threshold,
        single_class_name=args.single_class_name,
    )
    records = []
    try:
        for filename in sorted(image_records):
            frame = cv2.imread(str(image_root / filename))
            if frame is None:
                raise RuntimeError(f"could not read {filename}")
            height, width = frame.shape[:2]
            detections = [d for d in backend.detect(frame, 0) if d.class_name in {"cell phone", "cell_phone"}]
            gt = [box for name, box in truth.get(filename, []) if name == "cell_phone"]
            ignore = [box for name, box in truth.get(filename, []) if name == "cell_phone_ignore"]
            best_iou = max((iou(gt_box, pixel_box(d, width, height)) for gt_box in gt for d in detections), default=0.0)
            best_conf = max((float(d.confidence) for d in detections), default=0.0)
            best_pred = max(detections, key=lambda d: d.confidence, default=None)
            candidate_boxes = [pixel_box(d, width, height) for d in detections]
            record = {
                "filename": filename,
                "scenario": image_records[filename].get("scenario"),
                "source_run": image_records[filename].get("source_run"),
                "frame_index": image_records[filename].get("frame_index"),
                "ground_truth_boxes": gt,
                "ignore_boxes": ignore,
                "gt_area_fraction": (((gt[0][2] - gt[0][0]) * (gt[0][3] - gt[0][1])) / (width * height)) if gt else 0.0,
                "phone_candidates": [
                    {"confidence": float(d.confidence), "box": box, "box_area_fraction": ((box[2] - box[0]) * (box[3] - box[1])) / (width * height)}
                    for d, box in zip(detections, candidate_boxes)
                ],
                "best_confidence": best_conf,
                "best_iou": best_iou,
                "detected_at_candidate_threshold": bool(detections),
                "best_prediction": pixel_box(best_pred, width, height) if best_pred else None,
            }
            records.append(record)

            if gt and (best_iou < args.iou_threshold or best_conf < 0.25):
                annotated = frame.copy()
                for gt_box in gt:
                    draw_box(annotated, gt_box, (0, 255, 0), "GT phone")
                for d in detections:
                    draw_box(annotated, pixel_box(d, width, height), (0, 0, 255), f"YOLO {d.confidence:.3f}")
                cv2.imwrite(str(evidence_dir / filename), annotated)
    finally:
        backend.close()

    by_scenario = defaultdict(list)
    for record in records:
        by_scenario[record["scenario"]].append(record)
    scenario_summary = {}
    for scenario, items in sorted(by_scenario.items()):
        positive = [x for x in items if x["ground_truth_boxes"]]
        negative = [x for x in items if not x["ground_truth_boxes"]]
        scenario_summary[scenario] = {
            "images": len(items),
            "positive_images": len(positive),
            "negative_images": len(negative),
            "positive_candidate_detection_rate": sum(x["detected_at_candidate_threshold"] for x in positive) / len(positive) if positive else None,
            "positive_iou50_rate": sum(x["best_iou"] >= args.iou_threshold for x in positive) / len(positive) if positive else None,
            "positive_best_conf_median": statistics.median(x["best_confidence"] for x in positive) if positive else None,
            "positive_gt_area_median": statistics.median(x["gt_area_fraction"] for x in positive) if positive else None,
            "negative_any_candidate_rate": sum(x["detected_at_candidate_threshold"] for x in negative) / len(negative) if negative else None,
            "negative_candidate_box_count": sum(len(x["phone_candidates"]) for x in negative),
        }

    report = {
        "annotations": str(args.annotations.resolve()),
        "model": str(args.model.resolve()),
        "candidate_threshold": args.candidate_threshold,
        "iou_threshold": args.iou_threshold,
        "images": len(records),
        "evidence_images": len(list(evidence_dir.glob("*.jpg"))),
        "by_scenario": scenario_summary,
        "records": records,
    }
    (args.output_root / "diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str((args.output_root / 'diagnostic.json').resolve()), "images": len(records), "evidence_images": len(list(evidence_dir.glob('*.jpg')))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
