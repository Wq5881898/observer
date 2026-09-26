import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication
    from observer.desktop_ipc import SingleInstanceWake


@unittest.skipUnless(HAS_QT, 'desktop optional dependency not installed')
class DesktopIpcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = SingleInstanceWake(Path(self.temp.name))
        self.assertTrue(self.server.start())

    def tearDown(self):
        self.server.close()
        self.app.processEvents()
        self.temp.cleanup()

    def run_client(self, root, timeout=800):
        code = (
            "import os,sys; os.environ['QT_QPA_PLATFORM']='offscreen'; "
            "from PySide6.QtWidgets import QApplication; "
            "from observer.desktop_ipc import SingleInstanceWake; QApplication([]); "
            "print(SingleInstanceWake.wake_existing(sys.argv[1], int(sys.argv[2])))"
        )
        process = subprocess.Popen(
            [sys.executable, '-c', code, str(root), str(timeout)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        deadline = time.monotonic() + 3
        while process.poll() is None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        stdout, stderr = process.communicate(timeout=.5)
        self.assertEqual(process.returncode, 0, stderr)
        return stdout.strip() == 'True'

    def test_exact_show_request_activates_once(self):
        activations = []
        self.server.activation_requested.connect(lambda: activations.append(True))
        self.assertTrue(self.run_client(self.temp.name))
        self.app.processEvents()
        self.assertEqual(activations, [True])

    def test_split_show_request_activates_once(self):
        activations = []
        self.server.activation_requested.connect(lambda: activations.append(True))
        code = (
            "import sys,time\n"
            "from PySide6.QtNetwork import QLocalSocket\n"
            "s=QLocalSocket()\n"
            "s.connectToServer(sys.argv[1])\n"
            "assert s.waitForConnected(800)\n"
            "s.write(b'sh')\n"
            "s.flush()\n"
            "if s.bytesToWrite(): assert s.waitForBytesWritten(800)\n"
            "time.sleep(.05)\n"
            "s.write(b'ow\\n')\n"
            "s.flush()\n"
            "if s.bytesToWrite(): assert s.waitForBytesWritten(800)\n"
            "if not s.bytesAvailable(): assert s.waitForReadyRead(800)\n"
            "print(bytes(s.readAll()) == b'ok\\n')\n"
        )
        process = subprocess.Popen(
            [sys.executable, '-c', code, self.server.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 3
        while process.poll() is None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        stdout, stderr = process.communicate(timeout=.5)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(stdout.strip(), 'True')
        self.assertEqual(activations, [True])

    def test_other_data_directory_does_not_connect(self):
        other = Path(self.temp.name) / 'other'
        self.assertFalse(self.run_client(other, 100))

    def test_closed_server_no_longer_accepts_wake(self):
        self.server.close()
        self.assertFalse(self.run_client(self.temp.name, 100))

    def test_wake_name_is_stable_per_resolved_data_directory(self):
        same = SingleInstanceWake(Path(self.temp.name) / '.')
        other = SingleInstanceWake(Path(self.temp.name) / 'other')
        self.assertEqual(self.server.name, same.name)
        self.assertNotEqual(self.server.name, other.name)
        self.assertTrue(
            self.server.server.socketOptions()
            & self.server.server.SocketOption.UserAccessOption)

    def test_successful_wake_returns_before_window_or_worker_creation(self):
        from observer.desktop import launch
        args = SimpleNamespace(data_dir=Path(self.temp.name), demo=True,
                               models_dir=Path('models'), profile=None)
        with patch('observer.desktop.SingleInstanceWake.wake_existing', return_value=True), \
             patch('observer.desktop.ObserverWindow') as window:
            self.assertEqual(launch(args), 0)
            window.assert_not_called()
