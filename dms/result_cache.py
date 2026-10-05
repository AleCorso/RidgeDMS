"""Local, content-addressed cache for trusted notebook results.

Pickles preserve the project's result dataclasses and Matplotlib figures. Only
load cache files produced by this project in a directory you control.
"""
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import pickle
import platform
import tempfile

import numpy as np


def _package_version(name):
    # Record optional libraries when present without making metadata collection
    # require them (the NumPy-only inference workflow does not need SciPy).
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def _atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class ResultCache:
    """Reuse exact inputs, implementation fingerprints, and package versions.

    ``compute=False`` is a cache-only read: it never calls the producer.
    Returned identities include the result digest, so a recomputed result cannot
    accidentally reuse a figure made from different data.
    """

    def __init__(self, directory, *, enabled=True, verbose=True):
        self.directory = Path(directory)
        self.enabled = enabled
        self.verbose = verbose

    def get(self, kind, inputs, producer=None, *, sources=(), implementation=(),
            force=False, compute=True):
        if not kind.replace('_', '').isalnum():
            raise ValueError('Cache kind must contain letters, numbers, or underscores')
        manifest = dict(
            schema=1, kind=kind, inputs=_json_value(inputs),
            sources={str(Path(p).name): sha256(Path(p).read_bytes()).hexdigest() for p in sources},
            implementation=list(implementation),
            versions=dict(python=platform.python_version(),
                          **{p: _package_version(p) for p in ('numpy', 'scipy', 'matplotlib')}),
        )
        encoded = json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
        key = sha256(encoded).hexdigest()
        path = self.directory / kind / (key + '.pkl')
        if self.enabled and not force and path.exists():
            try:
                envelope = pickle.loads(path.read_bytes())
                body = envelope['body']
                digest = sha256(body).hexdigest()
                if envelope['key'] != key or envelope['digest'] != digest:
                    raise ValueError('Cache integrity check failed')
                value = pickle.loads(body)
            except Exception as exc:
                # Interrupted/corrupt/obsolete cache files are misses, not results.
                if self.verbose:
                    print(f'{kind}: unreadable cache ({type(exc).__name__}); ignoring {path.name}')
            else:
                if self.verbose:
                    print(f'{kind}: cache hit {key[:12]}')
                return value, key + ':' + digest
        if not compute or producer is None:
            raise FileNotFoundError(f'{kind}: no matching cached result; run its compute cell first')
        if self.verbose:
            print(f'{kind}: computing {key[:12]}' + (' (forced)' if force else ''))
        value = producer()
        body = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        digest = sha256(body).hexdigest()
        if self.enabled:
            envelope = dict(key=key, digest=digest, body=body)
            _atomic_write(path, pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
            _atomic_write(path.with_suffix('.json'), json.dumps(manifest, indent=2).encode())
        return value, key + ':' + digest
