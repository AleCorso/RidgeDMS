"""Numerical evidence for switches between separated ridge-risk wells.

This is a diagnostic on an existing optimum path, not a second optimizer or a
proof of transition order. It deliberately does not classify from steepness.
"""
import numpy as np

from .ridge_diagnostics import effective_risk, kappa_from_lambda
from .stieltjes import StieltjesEvaluationError
from .theory import TheoryEvaluationError


def ridge_transition_evidence(optimum, cfg, *, n_probe=65, barrier_rtol=1e-8):
    """Find an intervening risk barrier at BOTH ends of each r interval.

    For adjacent chosen penalties a,b, evaluate R(lambda,r) between a and b
    at each endpoint r. A sampled risk above both endpoint risks establishes
    separation by a barrier there. If it occurs at both r values, the selected
    optimum has crossed a persistent barrier rather than just moved steeply.
    Failure to resolve a barrier is inconclusive; narrow/weak barriers can be
    missed. Probe resolution and tolerance are explicit, cache-keyed controls.
    No claim about infinite ridge follows from a finite search-cap result.
    """
    if int(n_probe) != n_probe or n_probe < 5 or not np.isfinite(barrier_rtol) or barrier_rtol < 0:
        raise ValueError('Require n_probe >= 5 and nonnegative barrier_rtol')
    rs = np.asarray(optimum.r_values, float)
    penalties = np.asarray(optimum.ridge_lambdas, float)
    if rs.ndim != 1 or penalties.shape != rs.shape:
        raise ValueError('Optimum r and ridge arrays must have matching shapes')
    jump = np.zeros(max(0, len(rs)-1), bool)
    barriers = np.full((len(jump), 2), np.nan)
    reasons = np.full(len(jump), 'no_resolved_barrier', dtype='<U32')
    for i in range(len(jump)):
        lo, hi = sorted(penalties[i:i+2])
        if not np.isfinite(lo+hi) or lo < 0:
            reasons[i] = 'nonfinite_endpoint'
            continue
        if lo == hi:
            reasons[i] = 'same_penalty'
            continue
        # For a zero endpoint include both logarithmic and linear interior probes.
        # This avoids treating lambda=0 as an arbitrary positive cutoff.
        if lo == 0:
            probes = np.unique(np.r_[0., np.geomspace(max(hi*1e-12, np.nextafter(0., 1.)), hi, n_probe),
                                    np.linspace(0., hi, n_probe)])
        else:
            probes = np.unique(np.r_[lo, np.geomspace(lo, hi, n_probe), hi])
        if len(probes) < 3:
            reasons[i] = 'unresolved_interval'
            continue
        passed = []
        for j, r in enumerate(rs[i:i+2]):
            try:
                risk = np.array([effective_risk(kappa_from_lambda(float(lam), float(r), cfg),
                                               float(r), cfg).risk for lam in probes])
            except (ValueError, FloatingPointError, OverflowError,
                    StieltjesEvaluationError, TheoryEvaluationError):
                reasons[i] = 'unresolved_evaluation'
                passed.append(False)
                continue
            if not np.all(np.isfinite(risk)):
                reasons[i] = 'unresolved_evaluation'
                passed.append(False)
                continue
            barrier = float(np.max(risk[1:-1]) - max(risk[0], risk[-1]))
            barriers[i, j] = barrier
            scale = max(1., float(np.max(np.abs(risk))))
            passed.append(barrier > barrier_rtol*scale)
        jump[i] = all(passed)
        if jump[i]:
            reasons[i] = 'barrier_switch'
    return dict(jump_mask=jump, barrier_excess=barriers, reason=reasons,
                r_intervals=np.column_stack((rs[:-1], rs[1:])),
                n_probe=int(n_probe), barrier_rtol=float(barrier_rtol))
