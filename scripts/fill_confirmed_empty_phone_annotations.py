from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def empty_annotation(filename: str, image_path: Path, width: int, height: int) -> ET.Element:
    root = ET.Element("annotation", verified="yes")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "path").text = str(image_path.resolve())
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Observer user-confirmed no-phone"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="Create empty VOC annotations for user-confirmed no-phone images")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("auto", "voc", "yolo"),
        default="auto",
        help="empty-label format; auto follows existing image-matched labels",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.annotations / "manifest.json").read_text(encoding="utf-8"))
    image_root = args.annotations / "images"
    xml_root = args.annotations / "labels-voc"
    metadata_root = args.annotations / "metadata"
    xml_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    existing_by_filename: dict[str, Path] = {}
    for xml_path in xml_root.glob("*.xml"):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename")
        if filename:
            existing_by_filename[filename] = xml_path

    image_stems = {Path(str(item["file_name"])).stem for item in manifest["images"]}
    existing_yolo_by_stem = {
        path.stem: path
        for path in xml_root.glob("*.txt")
        if path.stem in image_stems
    }
    output_format = args.format
    if output_format == "auto":
        output_format = "yolo" if existing_yolo_by_stem else "voc"

    created: list[str] = []
    skipped: list[str] = []
    for item in manifest["images"]:
        filename = Path(str(item["file_name"])).name
        image_path = image_root / filename
        stem = Path(filename).stem
        if filename in existing_by_filename or stem in existing_yolo_by_stem:
            skipped.append(filename)
            continue
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        suffix = ".txt" if output_format == "yolo" else ".xml"
        output = xml_root / f"{stem}{suffix}"
        if output.exists() and not args.overwrite:
            skipped.append(filename)
            continue
        if output_format == "yolo":
            output.write_text("", encoding="utf-8")
        else:
            root = empty_annotation(filename, image_path, int(item["width"]), int(item["height"]))
            ET.indent(root, space="\t")
            ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=False)
        created.append(filename)

    confirmation = {
        "schema_version": 1,
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
        "confirmation": "User confirmed that every image without an existing cell_phone annotation contains no phone.",
        "annotation_format": output_format,
        "created_empty_xml": created,
        "existing_xml_preserved": skipped,
    }
    (metadata_root / "no-phone-confirmation.json").write_text(
        json.dumps(confirmation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "annotation_format": output_format,
                "created_empty_labels": len(created),
                "existing_labels_preserved": len(skipped),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
