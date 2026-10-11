from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelFiles:
    face: Path
    pose: Path
    object: Path


def mediapipe_asset_options(path: str | Path) -> dict:
    """Avoid the Windows Tasks native narrow-path loader for Unicode paths."""
    path = Path(path)
    if platform.system() == 'Windows' and not str(path.resolve()).isascii():
        return {'model_asset_buffer': path.read_bytes()}
    return {'model_asset_path': str(path)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_model_directory(directory: str | Path) -> ModelFiles:
    model_directory = Path(directory)
    manifest_path = model_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("models")
    if not isinstance(entries, list):
        raise ValueError("model manifest must contain a models list")

    paths: dict[str, Path] = {}
    errors: list[str] = []
    for entry in entries:
        model_id = str(entry["id"])
        path = model_directory / str(entry["filename"])
        if not path.is_file():
            errors.append(f"missing model: {path.name}")
            continue
        actual = sha256_file(path)
        expected = str(entry["sha256"]).upper()
        if actual != expected:
            errors.append(f"SHA-256 mismatch: {path.name}")
            continue
        paths[model_id] = path.resolve()

    required = {"face", "pose", "object"}
    missing_ids = required - set(paths)
    if missing_ids:
        errors.append(f"manifest is missing valid model ids: {sorted(missing_ids)}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return ModelFiles(face=paths["face"], pose=paths["pose"], object=paths["object"])

