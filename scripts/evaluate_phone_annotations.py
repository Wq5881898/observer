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
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def parse_xml(path: Path):
    root = ET.parse(path).getroot()
    filename = root.findtext("filename") or ""
    width = float(root.findtext("size/width") or 0)
    height = float(root.findtext("size/height") or 0)
    objects = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        coords = tuple(float(box.findtext(key) or 0) for key in ("xmin", "ymin", "xmax", "ymax"))
        if not (0 <= coords[0] < coords[2] <= width and 0 <= coords[1] < coords[3] <= height):
            raise ValueError(f"invalid box in {path}: {coords}")
        objects.append((obj.findtext("name") or "", coords))
    return filename, objects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.15, 0.10])
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--single-class-name",
        default=None,
        help="class name for a single-class YOLO export, e.g. cell_phone",
    )
    args = parser.parse_args()

    manifest = json.loads((args.annotations / "manifest.json").read_text(encoding="utf-8"))
    image_root = args.annotations / "images"
    image_records = {str(item["file_name"]).split("/", 1)[-1]: item for item in manifest["images"]}
    xml_records = {}
    for xml_path in sorted((args.annotations / "labels-voc").glob("*.xml")):
        filename, objects = parse_xml(xml_path)
        xml_records[filename] = objects

    completed = sorted(xml_records)
    missing = sorted(set(image_records) - set(completed))
    by_scenario = defaultdict(lambda: {"images": 0, "phone": 0, "ignore": 0, "empty": 0})
    for filename, objects in xml_records.items():
        scenario = str(image_records.get(filename, {}).get("scenario", "unknown"))
        by_scenario[scenario]["images"] += 1
        by_scenario[scenario]["phone"] += sum(name == "cell_phone" for name, _ in objects)
        by_scenario[scenario]["ignore"] += sum(name == "cell_phone_ignore" for name, _ in objects)
        by_scenario[scenario]["empty"] += int(not objects)

    reports = []
    for threshold in args.thresholds:
        backend = YoloOnnxBackend(
            args.model,
            score_threshold=threshold,
            single_class_name=args.single_class_name,
        )
        tp = fn = fp = 0
        presence_tp = presence_fn = presence_fp = presence_tn = 0
        latencies = []
        scenarios = defaultdict(lambda: {"gt": 0, "tp": 0, "fn": 0, "fp": 0})
        try:
            for filename in completed:
                record = image_records[filename]
                scenario = str(record.get("scenario", "unknown"))
                frame = cv2.imread(str(image_root / filename))
                if frame is None:
                    raise RuntimeError(f"could not read {filename}")
                detections = [d for d in backend.detect(frame, 0) if d.class_name in {"cell phone", "cell_phone"}]
                latencies.append(float(backend.last_latency_ms))
                ground_truth = [box for name, box in xml_records[filename] if name == "cell_phone"]
                ignored = [box for name, box in xml_records[filename] if name == "cell_phone_ignore"]
                stats = scenarios[scenario]
                stats["gt"] += len(ground_truth)
                if ground_truth:
                    if detections:
                        presence_tp += 1
                    else:
                        presence_fn += 1
                elif detections:
                    presence_fp += 1
                else:
                    presence_tn += 1
                used = set()
                for gt_box in ground_truth:
                    best_index = None
                    best_overlap = 0.0
                    for index, detection in enumerate(detections):
                        if index in used:
                            continue
                        b = detection.box
                        pred_box = (b.x * frame.shape[1], b.y * frame.shape[0], (b.x + b.width) * frame.shape[1], (b.y + b.height) * frame.shape[0])
                        overlap = iou(gt_box, pred_box)
                        if overlap > best_overlap:
                            best_index, best_overlap = index, overlap
                    if best_index is not None and best_overlap >= args.iou_threshold:
                        used.add(best_index)
                        stats["tp"] += 1
                        tp += 1
                    else:
                        stats["fn"] += 1
                        fn += 1
                for index, detection in enumerate(detections):
                    if index in used:
                        continue
                    b = detection.box
                    pred_box = (b.x * frame.shape[1], b.y * frame.shape[0], (b.x + b.width) * frame.shape[1], (b.y + b.height) * frame.shape[0])
                    if any(iou(pred_box, ignored_box) >= args.iou_threshold for ignored_box in ignored):
                        continue
                    stats["fp"] += 1
                    fp += 1
        finally:
            backend.close()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        presence_precision = (
            presence_tp / (presence_tp + presence_fp)
            if presence_tp + presence_fp
            else 0.0
        )
        presence_recall = (
            presence_tp / (presence_tp + presence_fn)
            if presence_tp + presence_fn
            else 0.0
        )
        negative_false_positive_rate = (
            presence_fp / (presence_fp + presence_tn)
            if presence_fp + presence_tn
            else 0.0
        )
        reports.append({"threshold": threshold, "images_evaluated": len(completed), "tp": tp, "fn": fn, "fp": fp, "precision": precision, "recall": recall, "f1": f1, "presence": {"tp_images": presence_tp, "fn_images": presence_fn, "fp_images": presence_fp, "tn_images": presence_tn, "precision": presence_precision, "recall": presence_recall, "negative_false_positive_rate": negative_false_positive_rate}, "latency_ms_mean": statistics.mean(latencies) if latencies else None, "by_scenario": dict(scenarios)})

    report = {"annotations": str(args.annotations.resolve()), "manifest_images": len(image_records), "xml_images": len(completed), "missing_xml_images": len(missing), "missing_xml_examples": missing[:20], "annotation_coverage": len(completed) / len(image_records) if image_records else 0.0, "by_scenario_completed": dict(by_scenario), "metrics_scope": "completed XML only; missing XML is excluded", "iou_threshold": args.iou_threshold, "thresholds": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "xml_images": len(completed), "manifest_images": len(image_records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
