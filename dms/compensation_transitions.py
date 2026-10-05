"""Separately cached transition evidence for the coupled compensation path.

This is the existing parameter-path risk-barrier diagnostic applied to both
complete neighbouring configurations. It does not classify jumps from curve
steepness, a ridge ratio or boundary status, and never reoptimizes a curve.
Keeping this adapter separate preserves the existing numerical curve caches.
Figure construction and manual presentation overrides belong in the notebook.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .parameter_axis import parameter_transition_evidence
from .reference_compensation import SOURCES as CURVE_SOURCES


DEFAULT_OPTIONS = dict(n_probe=65, barrier_rtol=1e-8)
_SOURCE_DIR = Path(__file__).resolve().parent
SOURCES = CURVE_SOURCES + tuple(_SOURCE_DIR / name for name in (
    'compensation_transitions.py', 'parameter_axis.py',
    'transition_diagnostics.py',
))


def _options(options):
    supplied = {} if options is None else dict(options)
    unknown = set(supplied) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f'Unknown compensation transition options: {sorted(unknown)}')
    controls = dict(DEFAULT_OPTIONS, **supplied)
    count, tolerance = controls['n_probe'], controls['barrier_rtol']
    if not np.isfinite(count) or int(count) != count or count < 5:
        raise ValueError('n_probe must be an integer >= 5')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('barrier_rtol must be finite and nonnegative')
    return dict(n_probe=int(count), barrier_rtol=float(tolerance))


def compensation_transition_evidence(data, *, n_probe=65, barrier_rtol=1e-8):
    """Diagnose adjacent sampled optima without altering their coordinates.

    Both mismatch means vary on this path; each endpoint therefore uses its
    own stored full configuration. Failed/infeasible endpoints leave an
    unresolved interval rather than joining points across the missing sample.

    A resolved risk barrier at both endpoints is numerical evidence for a
    switch between separated wells. It is resolution-dependent, does not
    establish transition order or the exact crossing, and does not certify
    that a finite-cap optimum is infinite. A missing barrier is inconclusive.
    """
    controls = _options(dict(n_probe=n_probe, barrier_rtol=barrier_rtol))
    means = np.asarray(data['mean_values'], dtype=float)
    ridge = np.asarray(data['ridge'], dtype=float)
    status = np.asarray(data['status'])
    contributions = np.asarray(data['contribution'], dtype=float)
    if (means.ndim != 1 or not means.size or np.any(~np.isfinite(means))
            or np.any(np.diff(means) <= 0)):
        raise ValueError('mean_values must be a nonempty finite increasing vector')
    if any(values.shape != means.shape for values in (ridge, status, contributions)):
        raise ValueError('ridge, status and contribution must match mean_values')
    r = float(data['r'])
    if not np.isfinite(r) or r <= 0:
        raise ValueError('r must be finite and positive')
    size = means.size - 1
    jumps = np.zeros(size, dtype=bool)
    barriers = np.full((size, 2), np.nan)
    endpoint_reason = np.full((size, 2), 'unavailable_endpoint', dtype='<U32')
    reasons = np.full(size, 'unavailable_endpoint', dtype='<U32')
    eligible = np.zeros(size, dtype=bool)
    configs = data['configs']
    valid_status = np.isin(status, ('interior', 'zero', 'upper'))
    for index in range(size):
        endpoints = (index, index + 1)
        if not all(point in configs and valid_status[point]
                   and np.isfinite(ridge[point]) and ridge[point] >= 0
                   for point in endpoints):
            continue
        eligible[index] = True
        # Reuse the diagnostic for parameter sweeps; no single-parameter
        # substitution is made, so the coupled mean path remains intact.
        curve = SimpleNamespace(
            r=r,
            cases=tuple(SimpleNamespace(config=configs[point]) for point in endpoints),
            ridge_lambdas=ridge[index:index + 2],
            parameter_values=means[index:index + 2],
        )
        evidence = parameter_transition_evidence(curve, **controls)
        jumps[index] = evidence['jump_mask'][0]
        barriers[index] = evidence['barrier_excess'][0]
        endpoint_reason[index] = evidence['endpoint_reason'][0]
        reasons[index] = evidence['reason'][0]
    return dict(
        jump_mask=jumps, jump_edges=np.flatnonzero(jumps).tolist(),
        barrier_excess=barriers, endpoint_reason=endpoint_reason, reason=reasons,
        eligible=eligible,
        parameter_intervals=np.column_stack((means[:-1], means[1:])),
        contribution_intervals=np.column_stack((contributions[:-1], contributions[1:])),
        fixed_r=r, **controls,
        note=('Numerical barrier evidence at both sampled endpoint configurations; '
              'unresolved intervals are inconclusive. No exact crossing or '
              'transition order is inferred, and finite-cap hits remain cap hits.'),
    )


def cached_compensation_transitions(data, curve_identity, *, options, cache,
                                    force=False, compute=True):
    """Return (evidence, identity) for an existing cached compensation curve.

    The curve identity includes its result digest. The diagnostic cache also
    includes every diagnostic control and relevant source file. Plot-only
    reads use compute=False: they never call the diagnostic producer, evaluate
    risk, convert moments or optimize, and report a missing cache explicitly.
    """
    if not isinstance(curve_identity, str) or not curve_identity:
        raise ValueError('curve_identity must be the cached curve result identity')
    controls = _options(options)
    inputs = dict(curve_identity=curve_identity, options=controls)
    return cache.get(
        'compensation_transitions', inputs,
        lambda: compensation_transition_evidence(data, **controls),
        sources=SOURCES, force=force, compute=compute,
    )
