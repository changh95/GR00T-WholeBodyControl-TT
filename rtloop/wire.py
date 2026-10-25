"""The Isaac-GR00T PolicyServer wire format, without importing gr00t (which pulls in torch and
transformers, about 1.6 s).

A message is msgpack; an ndarray travels as ``{"__ndarray_class__": True, "as_npy": <np.save
bytes>}``, exactly as ``gr00t.policy.server_client.MsgSerializer`` encodes it, so either side can
talk to the stock server or client. A ``ModalityConfig`` arriving from a stock server is left as
its ``{"__ModalityConfig_class__": True, "as_json": ...}`` dict.
"""

import io

import msgpack
import numpy as np


def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _decode(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def dumps(data) -> bytes:
    return msgpack.packb(data, default=_encode)


def loads(data: bytes):
    return msgpack.unpackb(data, object_hook=_decode)


def request(endpoint: str, data: dict | None = None) -> bytes:
    """A PolicyServer request: ``{"endpoint": ..., "data": {...}}`` (``data`` omitted if None)."""
    msg = {"endpoint": endpoint}
    if data is not None:
        msg["data"] = data
    return dumps(msg)
