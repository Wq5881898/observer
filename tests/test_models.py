import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from observer.models import verify_model_directory


class ModelManifestTests(unittest.TestCase):
    def test_manifest_hashes_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            entries = []
            for model_id in ("face", "pose", "object"):
                content = model_id.encode("utf-8")
                filename = f"{model_id}.bin"
                (directory / filename).write_bytes(content)
                entries.append(
                    {
                        "id": model_id,
                        "filename": filename,
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
            (directory / "manifest.json").write_text(
                json.dumps({"models": entries}), encoding="utf-8"
            )
            files = verify_model_directory(directory)
            self.assertEqual(files.face.name, "face.bin")

    def test_changed_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            entries = []
            for model_id in ("face", "pose", "object"):
                filename = f"{model_id}.bin"
                (directory / filename).write_bytes(b"actual")
                entries.append({"id": model_id, "filename": filename, "sha256": "00"})
            (directory / "manifest.json").write_text(
                json.dumps({"models": entries}), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                verify_model_directory(directory)


if __name__ == "__main__":
    unittest.main()
