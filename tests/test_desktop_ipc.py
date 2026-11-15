import importlib.util
import json
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

    def run_raw_client_until_closed(self, request):
        # Keep the peer alive until the server closes it: exiting the client
        # must not be mistaken for server-side rejection or timeout cleanup.
        code = (
            "import json,sys,time\n"
            "from PySide6.QtCore import QCoreApplication\n"
            "from PySide6.QtNetwork import QLocalSocket\n"
            "app=QCoreApplication([])\n"
            "s=QLocalSocket()\n"
            "s.connectToServer(sys.argv[1])\n"
            "assert s.waitForConnected(800)\n"
            "s.write(bytes.fromhex(sys.argv[2]))\n"
            "s.flush()\n"
            # A rejected oversized request may close the peer before its full
            # payload drains. The rejection/response assertions below still
            # verify server behavior; a clean peer close isn't a send failure.
            "if s.bytesToWrite() and not s.waitForBytesWritten(800):\n"
            "    assert s.state()==QLocalSocket.LocalSocketState.UnconnectedState or s.error()==QLocalSocket.LocalSocketError.PeerClosedError\n"
            "started=time.monotonic()\n"
            "response=b''\n"
            "deadline=started+2.5\n"
            "while s.state()!=QLocalSocket.LocalSocketState.UnconnectedState:\n"
            "    remaining=int((deadline-time.monotonic())*1000)\n"
            "    if remaining<=0: break\n"
            "    if not s.bytesAvailable(): s.waitForReadyRead(remaining)\n"
            "    response+=bytes(s.readAll())\n"
            "print(json.dumps(dict(\n"
            "    closed=s.state()==QLocalSocket.LocalSocketState.UnconnectedState,\n"
            "    response=response.hex(), elapsed=time.monotonic()-started)))\n"
        )
        process = subprocess.Popen(
            [sys.executable, '-c', code, self.server.name, request.hex()],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        observed_requests = []
        deadline = time.monotonic() + 4
        while process.poll() is None and time.monotonic() < deadline:
            self.app.processEvents()
            observed_requests.extend(bytes(value) for value in self.server.requests.values())
            time.sleep(.005)
        stdout, stderr = process.communicate(timeout=.5)
        self.assertEqual(process.returncode, 0, stderr)
        self.app.processEvents()
        return json.loads(stdout), observed_requests

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

    def test_malformed_requests_close_without_activation_or_response(self):
        activations = []
        self.server.activation_requested.connect(lambda: activations.append(True))
        for request in (b'hide\n', b'show\x00', b'sh\xff\n'):
            with self.subTest(request=request):
                result, _ = self.run_raw_client_until_closed(request)
                self.assertTrue(result['closed'], result)
                self.assertEqual(result['response'], '')
                self.assertEqual(activations, [])
                self.assertEqual(self.server.requests, {})
                self.assertEqual(self.server.connections, set())

    def test_show_with_oversized_suffix_is_rejected_without_activation(self):
        activations = []
        self.server.activation_requested.connect(lambda: activations.append(True))
        result, _ = self.run_raw_client_until_closed(b'show\n' + b'x' * 4096)
        self.assertTrue(result['closed'], result)
        self.assertEqual(result['response'], '')
        self.assertEqual(activations, [])
        self.assertEqual(self.server.requests, {})
        self.assertEqual(self.server.connections, set())

    def test_incomplete_request_expires_and_next_valid_client_activates_once(self):
        activations = []
        self.server.activation_requested.connect(lambda: activations.append(True))
        result, observed_requests = self.run_raw_client_until_closed(b'sh')
        self.assertIn(b'sh', observed_requests)
        self.assertTrue(result['closed'], result)
        self.assertEqual(result['response'], '')
        # Real elapsed time verifies the production 1000 ms timer, without
        # mocking it. Allow modest Qt timer and process scheduling jitter.
        self.assertGreaterEqual(result['elapsed'], .85)
        self.assertLess(result['elapsed'], 1.8)
        self.assertEqual(activations, [])
        self.assertEqual(self.server.requests, {})
        self.assertEqual(self.server.connections, set())
        self.assertTrue(self.run_client(self.temp.name))
        self.assertEqual(activations, [True])
        self.assertEqual(self.server.requests, {})
        self.assertEqual(self.server.connections, set())

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
