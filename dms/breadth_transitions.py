"""Two-endpoint risk-barrier evidence for changes in optimal breadth.

Uses saved adaptive profiles to screen intervals and supply interior probes.
Only missing cross-endpoint risks (and explicitly requested extra probes) need
ridge optimization. Original breadth/profile caches are never modified.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from .parameter_planes import SOURCES, cached_parameter_map


TRANSITION_SOURCES = (*SOURCES, Path(__file__).resolve())


def _profile_arrays(row):
    f = np.asarray(row.get('evaluated_f', []), dtype=float)
    risk = np.asarray(row.get('evaluated_risk', []), dtype=float)
    status = np.asarray(row.get('evaluated_status', []), dtype=str)
    if (f.ndim != 1 or f.shape != risk.shape or f.shape != status.shape
            or np.any(~np.isfinite(f)) or np.any(np.diff(f) <= 0)):
        raise ValueError('Profile arrays must align on increasing finite breadths')
    # A cap value, even with an excluded tail, is not an evaluated optimum.
    valid = np.isfinite(risk) & np.isin(status, ['zero', 'interior'])
    return f, risk, valid


def _screen_profile(row, lo, hi, barrier_rtol):
    """Cheap candidate screen only; interpolation is never used as evidence."""
    f, risk, valid = _profile_arrays(row)
    f, risk = f[valid], risk[valid]
    inside = (f > lo) & (f < hi)
    if f.size < 3 or not np.any(inside) or lo < f[0] or hi > f[-1]:
        return False
    endpoints = np.interp([lo, hi], f, risk)
    scale = max(1., float(np.max(np.abs(np.r_[risk[inside], endpoints]))))
    return bool(np.max(risk[inside]) > max(endpoints) + barrier_rtol * scale)


def cached_breadth_transition_evidence(study, study_identity, *, cache, point_cache,
                                      barrier_rtol=1e-8, n_probe=0,
                                      compute=False, force=False):
    """A dashed interval requires a resolved barrier at BOTH test breadths.

    At least one saved profile must suggest an intervening barrier before an
    interval is checked. For a candidate, evaluate the risk at BOTH adjacent
    f_opt values in each endpoint test distribution, then compare with saved
    interior risks. Optional n_probe>=3 adds linear interior probes for these
    candidate intervals. n_probe=0 reuses the saved interior samples only.

    The criterion matches the landscape/sweep barrier check, now applied to
    R_opt(f; ftilde) instead of R(lambda; r). Not resolving a barrier is
    inconclusive: coarse adaptive profiles can miss weak/narrow wells. It is
    not evidence for continuity or a proof/precise location of a discontinuity.
    No steepness, jump-size or boundary-status classification is used.
    """
    if not np.isfinite(barrier_rtol) or barrier_rtol < 0:
        raise ValueError('barrier_rtol must be finite and nonnegative')
    if int(n_probe) != n_probe or (n_probe != 0 and n_probe < 3):
        raise ValueError('n_probe must be 0 or an integer >=3')
    inputs = dict(study_identity=study_identity,
                  barrier_rtol=float(barrier_rtol), n_probe=int(n_probe))

    def produce():
        meta = study['metadata']
        curves = []
        point_ids = []
        for value, rows in study['profiles']:
            count = max(0, len(rows)-1)
            mask = np.zeros(count, dtype=bool)
            screened = np.zeros(count, dtype=bool)
            barriers = np.full((count, 2), np.nan)
            reasons = np.full(count, 'no_cached_barrier_candidate', dtype=object)
            bounds = np.array([[rows[i]['ftilde'], rows[i+1]['ftilde']]
                               for i in range(count)], dtype=float).reshape(count, 2)
            details = []

            def risk_at(row, f):
                fs, risks, valid = _profile_arrays(row)
                known = np.flatnonzero((fs == f) & valid)
                if known.size:
                    return float(risks[known[0]])
                if f == row['f_opt'] and row['ridge_status'] in ('zero', 'interior'):
                    return float(row['risk_opt'])
                data, identity = cached_parameter_map(
                    base=replace(meta['base'], Fneq=float(value), ftilde=float(row['ftilde'])),
                    x_key='f', x_values=[float(f)], y_key='ftilde',
                    y_values=[float(row['ftilde'])], r=float(meta['r']),
                    mean_convention=meta['mean_convention'], reference_L=meta['reference_L'],
                    search=deepcopy(meta['search']), cache=point_cache, compute=True)
                point_ids.append(identity)
                if data['status'][0, 0] not in ('zero', 'interior'):
                    raise ValueError('cross-endpoint risk unresolved: '
                                     + str(data['status'][0, 0]) + '; ' + str(data['reason'][0, 0]))
                result = float(data['risk'][0, 0])
                if not np.isfinite(result):
                    raise ValueError('cross-endpoint risk is nonfinite')
                return result

            for i in range(count):
                pair = rows[i:i+2]
                detail = dict(index=i, endpoint_checks=[])
                details.append(detail)
                if not all(row['complete'] and np.isfinite(row['f_opt']) for row in pair):
                    reasons[i] = 'unresolved_optimum'
                    continue
                lo, hi = sorted(float(row['f_opt']) for row in pair)
                if lo == hi:
                    reasons[i] = 'same_breadth'
                    continue
                if not any(_screen_profile(row, lo, hi, barrier_rtol) for row in pair):
                    continue
                screened[i] = True
                passed = []
                for endpoint, row in enumerate(pair):
                    try:
                        edge_risk = np.array([risk_at(row, lo), risk_at(row, hi)])
                        fs, risks, valid = _profile_arrays(row)
                        interior = risks[valid & (fs > lo) & (fs < hi)]
                        if n_probe:
                            interior = np.r_[interior, [risk_at(row, float(f)) for f in
                                             np.linspace(lo, hi, n_probe)[1:-1]]]
                        if not interior.size:
                            raise ValueError('no resolved interior probes')
                        excess = float(np.max(interior) - np.max(edge_risk))
                        scale = max(1., float(np.max(np.abs(np.r_[edge_risk, interior]))))
                        barriers[i, endpoint] = excess
                        passed.append(excess > barrier_rtol * scale)
                        detail['endpoint_checks'].append(dict(
                            ftilde=float(row['ftilde']), f_bounds=(lo, hi),
                            endpoint_risk=edge_risk, n_interior=int(interior.size),
                            barrier_excess=excess, threshold=barrier_rtol * scale))
                    except (ValueError, FloatingPointError, OverflowError) as exc:
                        passed.append(False)
                        detail['endpoint_checks'].append(dict(
                            ftilde=float(row['ftilde']), reason=str(exc)))
                mask[i] = all(passed)
                reasons[i] = 'barrier_switch' if mask[i] else 'unresolved_two_endpoint_barrier'
            curves.append((value, dict(jump_mask=mask, screened=screened,
                ftilde_intervals=bounds, barrier_excess=barriers, reason=reasons, details=details)))
        return dict(curves=curves, metadata=dict(r=float(meta['r']), **inputs),
                    point_ids=point_ids,
                    note='Two-endpoint numerical barrier evidence; unresolved does not imply continuity.')

    return cache.get('breadth_transition_evidence', inputs, produce,
                     sources=TRANSITION_SOURCES, compute=compute, force=force)
