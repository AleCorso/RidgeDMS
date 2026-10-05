"""Read and write numerical figure data as checksummed JSON + NPZ archives."""
from dataclasses import fields, is_dataclass
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from importlib import import_module

import numpy as np


_RECORD_MODULES = {
    'DMSConfig': 'dms.config',
    'SweepParameters': 'dms.exploration',
    'SweepCase': 'dms.exploration',
    'OptimizedSweepCurve': 'dms.exploration',
    'FixedRSweepCurve': 'dms.parameter_axis',
    'BoundaryRidgeOptimization': 'dms.ridge_diagnostics',
    'RidgeOptimizationResult': 'dms.optimization',
    'GRJobResult': 'dms.jobs',
    'FiniteSizeResult': 'dms.jobs',
    'FeasibilityReport': 'dms.simulation',
}


def assert_settings_match(saved, current):
    """Reject bundled data when numerical controls have been changed."""
    def normalize(value):
        if is_dataclass(value) and not isinstance(value, type):
            return normalize({f.name: getattr(value, f.name) for f in fields(value)})
        if isinstance(value, np.ndarray):
            return normalize(value.tolist())
        if isinstance(value, np.generic):
            return normalize(value.item())
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        if isinstance(value, Path):
            return value.name
        return value
    left = json.dumps(normalize(saved), sort_keys=True, allow_nan=False)
    right = json.dumps(normalize(current), sort_keys=True, allow_nan=False)
    if left != right:
        raise ValueError('Numerical settings differ from the bundled results. '
                         'Set RECOMPUTE=True and rerun the numerical stages, '
                         'or restore the published settings.')


def save_results(stem, results, *, metadata=None):
    """Save a structural JSON record and named, non-object NumPy arrays."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                # Legacy result dictionaries use object arrays for status text.
                if all(isinstance(v, str) for v in value.flat):
                    value = value.astype(str)
                else:
                    return {'type': 'object_array', 'shape': list(value.shape),
                            'items': [encode(v) for v in value.flat]}
            key = f'array_{len(arrays):04d}'
            arrays[key] = value
            return {'type': 'array', 'key': key, 'shape': list(value.shape),
                    'dtype': str(value.dtype)}
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, Fraction):
            return {'type': 'rational', 'numerator': value.numerator,
                    'denominator': value.denominator}
        if is_dataclass(value) and not isinstance(value, type):
            return {'type': 'record', 'name': type(value).__name__,
                    'fields': encode({f.name: getattr(value, f.name) for f in fields(value)})}
        if isinstance(value, dict):
            return {'type': 'dict', 'items': [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {'type': 'tuple' if isinstance(value, tuple) else 'list',
                    'items': [encode(v) for v in value]}
        if isinstance(value, Path):
            # Paths are not needed to reconstruct results; avoid machine-specific paths.
            return {'type': 'path_name', 'value': value.name}
        if isinstance(value, float) and not np.isfinite(value):
            return {'type': 'nonfinite', 'value': str(value)}
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(f'Unsupported portable result type: {type(value).__name__}')

    bundle = Path(__file__).resolve().parents[1]
    document = {'schema': 1, 'results': encode(results), 'metadata': encode(metadata or {})}
    document['utility_sha256'] = {
        p.name: sha256(p.read_bytes()).hexdigest() for p in sorted((bundle/'dms').glob('*.py'))}
    payload = stem.with_suffix('.npz')
    np.savez_compressed(payload, **arrays)
    document['npz_sha256'] = sha256(payload.read_bytes()).hexdigest()
    record = stem.with_suffix('.json')
    record.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    return record, payload


def load_results(stem, *, restore_records=False):
    """Read an exported archive with allow_pickle=False and checksum validation."""
    stem = Path(stem)
    document = json.loads(stem.with_suffix('.json').read_text())
    payload = stem.with_suffix('.npz')
    if document.get('schema') != 1:
        raise ValueError('Unsupported numerical archive schema')
    if sha256(payload.read_bytes()).hexdigest() != document['npz_sha256']:
        raise ValueError('Numerical archive checksum mismatch')
    with np.load(payload, allow_pickle=False) as archive:
        def decode(value):
            if not isinstance(value, dict):
                return value
            kind = value['type']
            if kind == 'array':
                arr = archive[value['key']].copy()
                if list(arr.shape) != value['shape'] or str(arr.dtype) != value['dtype']:
                    raise ValueError('Numerical array metadata mismatch')
                return arr
            if kind == 'dict':
                return {decode(k): decode(v) for k, v in value['items']}
            if kind in ('list', 'tuple'):
                items = [decode(v) for v in value['items']]
                return tuple(items) if kind == 'tuple' else items
            if kind == 'record':
                values = decode(value['fields'])
                if not restore_records:
                    return values
                name = value['name']
                if name not in _RECORD_MODULES:
                    raise ValueError(f'Unsupported numerical record: {name}')
                cls = getattr(import_module(_RECORD_MODULES[name]), name)
                return cls(**values)
            if kind == 'rational':
                return Fraction(value['numerator'], value['denominator'])
            if kind == 'path_name':
                return value['value']
            if kind == 'nonfinite':
                return float(value['value'])
            if kind == 'object_array':
                arr = np.empty(len(value['items']), dtype=object)
                for i, item in enumerate(value['items']):
                    arr[i] = decode(item)
                return arr.reshape(value['shape'])
            raise ValueError(f'Unknown archive entry: {kind}')
        return decode(document['results']), decode(document['metadata'])
