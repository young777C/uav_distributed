"""Length-prefixed pickle wire protocol for the env↔policy socket bridge.

The CARLA env (cyh-carla, py3.6) and the VLA policy (acot-uav-train, py3.11) cannot
share a process (incompatible python/torch/carla). They talk over TCP on the shared
`cyh-carla-net`: the env sends each tick's frame + GT, the policy returns the action.

pickle protocol 2 so a py3.11 server can read a py3.6 client's frames and vice-versa.
numpy arrays pickle portably across versions. Messages are plain dicts.
"""

from __future__ import annotations

import pickle
import socket
import struct

_PROTO = 2
_HDR = struct.Struct(">Q")     # 8-byte big-endian length prefix


def send_msg(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=_PROTO)
    sock.sendall(_HDR.pack(len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket):
    (n,) = _HDR.unpack(_recv_exact(sock, _HDR.size))
    return pickle.loads(_recv_exact(sock, n))
