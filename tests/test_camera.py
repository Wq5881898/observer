import threading
import time
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

    def test_start_can_be_cancelled_without_waiting_for_timeout(self) -> None:
        def blocked_run(worker: CameraWorker) -> None:
            worker._stop_event.wait(timeout=1.0)

        camera = CameraWorker(CameraConfig())
        started = time.monotonic()
        with patch.object(CameraWorker, "_run", blocked_run):
            self.assertFalse(camera.start(wait_seconds=1, cancelled=lambda: True))
        self.assertLess(time.monotonic() - started, .5)
        self.assertEqual(camera.stats.last_error, 'camera startup cancelled')


if __name__ == "__main__":
    unittest.main()
