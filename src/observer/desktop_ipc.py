"""A single narrow same-user wake signal for an existing desktop window."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket


class SingleInstanceWake(QObject):
    activation_requested = Signal()
    REQUEST = b'show\n'
    RESPONSE = b'ok\n'

    def __init__(self, data_dir, parent=None):
        super().__init__(parent)
        root = str(Path(data_dir).resolve()).casefold().encode('utf-8')
        self.name = 'observer-' + hashlib.sha256(root).hexdigest()[:24]
        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self.server.newConnection.connect(self._accept)
        self.connections = set()
        self.requests = {}

    @classmethod
    def wake_existing(cls, data_dir, timeout_ms=800):
        root = str(Path(data_dir).resolve()).casefold().encode('utf-8')
        name = 'observer-' + hashlib.sha256(root).hexdigest()[:24]
        socket = QLocalSocket()
        socket.connectToServer(name)
        if not socket.waitForConnected(timeout_ms):
            socket.abort()
            return False
        socket.write(cls.REQUEST)
        socket.flush()
        if socket.bytesToWrite() and not socket.waitForBytesWritten(timeout_ms):
            socket.abort()
            return False
        deadline = time.monotonic() + timeout_ms / 1000
        response = b''
        while len(response) < len(cls.RESPONSE):
            if socket.bytesAvailable():
                response += bytes(socket.readAll())
                if not cls.RESPONSE.startswith(response):
                    socket.abort()
                    return False
                continue
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0 or not socket.waitForReadyRead(remaining_ms):
                socket.abort()
                return False
        accepted = response == cls.RESPONSE
        socket.disconnectFromServer()
        return accepted

    def start(self):
        if self.server.listen(self.name):
            return True
        # Unix can leave a filesystem endpoint after a crash. Windows uses a
        # named pipe and must never remove a possibly live endpoint here.
        if os.name != 'nt':
            QLocalServer.removeServer(self.name)
            return self.server.listen(self.name)
        return False

    def _accept(self):
        while self.server.hasPendingConnections():
            socket = self.server.nextPendingConnection()
            self.connections.add(socket)
            self.requests[socket] = bytearray()
            socket.readyRead.connect(lambda current=socket: self._read(current))
            socket.disconnected.connect(
                lambda current=socket: self._discard(current))
            QTimer.singleShot(1000, lambda current=socket: self._expire(current))
            if socket.bytesAvailable():
                self._read(socket)

    def _read(self, socket):
        request = self.requests.get(socket)
        if request is None:
            return
        request.extend(bytes(socket.readAll()))
        if len(request) > len(self.REQUEST) or not self.REQUEST.startswith(request):
            socket.abort()
        elif request == self.REQUEST:
            self.requests.pop(socket, None)
            self.activation_requested.emit()
            socket.write(self.RESPONSE)
            socket.flush()
            socket.waitForBytesWritten(200)
            socket.disconnectFromServer()

    def _expire(self, socket):
        if socket in self.requests:
            socket.abort()

    def _discard(self, socket):
        self.connections.discard(socket)
        self.requests.pop(socket, None)
        socket.deleteLater()

    def close(self):
        self.server.close()
        for socket in tuple(self.connections):
            socket.abort()
        self.connections.clear()
        self.requests.clear()
