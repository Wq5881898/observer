from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_xml(path: Path) -> tuple[str, int, int, list[tuple[str, float, float, float, float]]]:
    root = ET.parse(path).getroot()
    filename = root.findtext("filename") or path.stem + ".jpg"
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing <size> in {path}")
    width = int(size.findtext("width") or 0)
    height = int(size.findtext("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size in {path}")
    objects: list[tuple[str, float, float, float, float]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name") or ""
        box = obj.find("bndbox")
        if box is None or name != "cell_phone":
            continue
        xmin = float(box.findtext("xmin") or 0)
        ymin = float(box.findtext("ymin") or 0)
        xmax = float(box.findtext("xmax") or 0)
        ymax = float(box.findtext("ymax") or 0)
        xmin = min(max(xmin, 0.0), float(width))
        ymin = min(max(ymin, 0.0), float(height))
        xmax = min(max(xmax, 0.0), float(width))
        ymax = min(max(ymax, 0.0), float(height))
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"invalid cell_phone box in {path}")
        objects.append((name, xmin, ymin, xmax, ymax))
    return filename, width, height, objects


def parse_yolo(
    path: Path, filename: str, width: int, height: int
) -> tuple[str, int, int, list[tuple[str, float, float, float, float]]]:
    objects: list[tuple[str, float, float, float, float]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid YOLO field count in {path}:{line_number}")
        class_id = int(fields[0])
        if class_id != 0:
            raise ValueError(f"unsupported YOLO class {class_id} in {path}:{line_number}")
        center_x, center_y, box_width, box_height = map(float, fields[1:])
        if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, box_width, box_height)):
            raise ValueError(f"YOLO coordinates outside [0,1] in {path}:{line_number}")
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError(f"empty YOLO box in {path}:{line_number}")
        xmin = (center_x - box_width / 2.0) * width
        ymin = (center_y - box_height / 2.0) * height
        xmax = (center_x + box_width / 2.0) * width
        ymax = (center_y + box_height / 2.0) * height
        tolerance = 1e-3
        if xmin < -tolerance or ymin < -tolerance or xmax > width + tolerance or ymax > height + tolerance:
            raise ValueError(f"YOLO box crosses image boundary in {path}:{line_number}")
        objects.append(
            (
                "cell_phone",
                min(max(xmin, 0.0), float(width)),
                min(max(ymin, 0.0), float(height)),
                min(max(xmax, 0.0), float(width)),
                min(max(ymax, 0.0), float(height)),
            )
        )
    return filename, width, height, objects


def parse_runs(value: str) -> set[str]:
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def group_for(
    record: dict[str, object], train_runs: set[str], val_runs: set[str]
) -> str:
    run = str(record.get("source_run") or "unknown").upper()
    if run in train_runs:
        return "train"
    if run in val_runs:
        return "val"
    raise ValueError(f"unsupported source_run for {record.get('file_name')}: {run}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a grouped YOLO phone fine-tuning dataset")
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/yolo-phone-gt-20260905-s96"))
    parser.add_argument(
        "--additional-annotations",
        type=Path,
        action="append",
        default=[],
        help="additional annotation root; may be repeated",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/training/yolo-phone-s96-runholdout"))
    parser.add_argument("--train-runs", default="P0")
    parser.add_argument("--val-runs", default="P1")
    parser.add_argument("--force", action="store_true", help="replace only this generated output directory")
    args = parser.parse_args()

    annotation_roots = [args.annotations.resolve()]
    annotation_roots.extend(path.resolve() for path in args.additional_annotations)
    output = args.output_root.resolve()
    for annotations in annotation_roots:
        if not annotations.is_dir():
            raise SystemExit(f"annotation directory not found: {annotations}")
    train_runs = parse_runs(args.train_runs)
    val_runs = parse_runs(args.val_runs)
    if not train_runs or not val_runs:
        raise SystemExit("--train-runs and --val-runs must each contain at least one run")
    overlap = train_runs & val_runs
    if overlap:
        raise SystemExit(f"runs cannot be in both train and val: {sorted(overlap)}")
    if output.exists():
        if not args.force:
            raise SystemExit(f"output already exists; pass --force to regenerate: {output}")
        # This is intentionally limited to the explicitly named generated root.
        shutil.rmtree(output)

    counts = {"train": {"images": 0, "positive": 0, "negative": 0}, "val": {"images": 0, "positive": 0, "negative": 0}}
    split_records: list[dict[str, object]] = []
    seen_names: set[str] = set()

    for annotations in annotation_roots:
        manifest = json.loads((annotations / "manifest.json").read_text(encoding="utf-8"))
        image_root = annotations / "images"
        xml_root = annotations / "labels-voc"
        for source_record in manifest.get("images", []):
            record = dict(source_record)
            record.setdefault("source_run", manifest.get("source_run"))
            file_name = Path(str(record["file_name"])).name
            if file_name in seen_names:
                raise SystemExit(f"duplicate image name across annotation roots: {file_name}")
            seen_names.add(file_name)
            image_path = image_root / file_name
            xml_path = xml_root / f"{Path(file_name).stem}.xml"
            yolo_path = xml_root / f"{Path(file_name).stem}.txt"
            if not image_path.is_file():
                raise SystemExit(f"missing image: {image_path}")
            split = group_for(record, train_runs, val_runs)
            if xml_path.is_file():
                filename, width, height, objects = parse_xml(xml_path)
            elif yolo_path.is_file():
                width = int(record.get("width") or 0)
                height = int(record.get("height") or 0)
                if width <= 0 or height <= 0:
                    raise SystemExit(f"missing image dimensions in manifest for {file_name}")
                filename, width, height, objects = parse_yolo(
                    yolo_path, file_name, width, height
                )
            else:
                raise SystemExit(f"missing XML or YOLO label for: {file_name}")
            if Path(filename).name != file_name:
                raise SystemExit(f"label filename mismatch: {filename}, manifest says {file_name}")

            image_out = output / "images" / split / file_name
            label_out = output / "labels" / split / f"{Path(file_name).stem}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_out)

            lines: list[str] = []
            for _, xmin, ymin, xmax, ymax in objects:
                center_x = ((xmin + xmax) / 2.0) / width
                center_y = ((ymin + ymax) / 2.0) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height
                lines.append(f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}")
            label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            counts[split]["images"] += 1
            counts[split]["positive" if objects else "negative"] += 1
            split_records.append(
                {
                    "file_name": file_name,
                    "split": split,
                    "source_annotation_root": str(annotations),
                    "source_run": record.get("source_run"),
                    "scenario": record.get("scenario"),
                    "source_video": record.get("source_video"),
                    "objects": len(objects),
                    "image_width": width,
                    "image_height": height,
                }
            )

    dataset_yaml = (
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: cell_phone\n"
    )
    (output / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    (output / "split.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_annotations": [str(path) for path in annotation_roots],
                "split_policy": {
                    "train_runs": sorted(train_runs),
                    "val_runs": sorted(val_runs),
                },
                "classes": ["cell_phone"],
                "counts": counts,
                "records": split_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# YOLO 手机微调数据集\n\n"
        "由 Observer 人工确认的 VOC 数据生成。只使用 `cell_phone` 一个类别；\n"
        "没有手机的图像保留为空 `.txt` 标签。数据按 source_run 整组分配，\n"
        "避免同一录像的相邻帧同时出现在训练和验证中。具体来源和划分见 split.json。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
