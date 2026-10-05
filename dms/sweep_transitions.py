"""Cached barrier diagnostics for stored parameter-sweep optima."""
from hashlib import sha256
from pathlib import Path
from dataclasses import asdict
import json

from .transition_diagnostics import ridge_transition_evidence


def sweep_transition_signature(curves, options):
    """Bind evidence to the full curve inputs, selected optima, and controls."""
    # Canonical values, not pickle bytes: shared-object memoization and NumPy
    # storage layout can change after a cache round trip without changing data.
    payload = dict(curves=[asdict(c) for c in curves], options=options)
    encoded = json.dumps(payload, sort_keys=True, default=lambda v: v.tolist())
    return sha256(encoded.encode()).hexdigest()


def cached_sweep_transitions(curves, options, cache, *, force=False):
    signature = sweep_transition_signature(curves, options)
    root = Path(__file__).parent
    evidence, identity = cache.get(
        'sweep_transitions', dict(curves_sha256=signature, options=options),
        lambda: tuple(ridge_transition_evidence(c.optimum, c.case.config, **options)
                      for c in curves),
        sources=[root / (name + '.py') for name in (
            'sweep_transitions', 'transition_diagnostics', 'ridge_diagnostics',
            'stieltjes', 'theory', 'config')], force=force,
    )
    return dict(signature=signature, evidence=evidence, identity=identity)
