import threading
import unittest
from unittest.mock import patch

from observer.camera import CameraWorker
from observer.config import CameraConfig


class CameraWorkerTests(unittest.TestCase):
    def test_start_reports_timeout(self) -> None:
        release = threading.Event()

        def blocked_run(worker: CameraWorker) -> None:
            release.wait(timeout=1.0)

        camera = CameraWorker(CameraConfig())
        with patch.object(CameraWorker, "_run", blocked_run):
            self.assertFalse(camera.start(wait_seconds=0.01))
            self.assertIn("startup timed out", camera.stats.last_error or "")
            release.set()
            camera.stop()


if __name__ == "__main__":
    unittest.main()
