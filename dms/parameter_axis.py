"""Fixed-r views of the same optimized parameter sweeps used by Figure 2.

Only numerical assembly lives here; figure construction stays in the notebook.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .exploration import SweepCase, optimize_parameter_sweep
from .transition_diagnostics import ridge_transition_evidence


@dataclass(frozen=True)
class FixedRSweepCurve:
    r: float
    cases: tuple[SweepCase, ...]
    ridge_lambdas: np.ndarray
    risks: np.ndarray
    status: np.ndarray
    r_infinity: np.ndarray

    @property
    def parameter_values(self):
        return np.array([case.value for case in self.cases])

    def regularization_values(self, coordinate):
        if coordinate == 'lambda':
            return self.ridge_lambdas.copy()
        if coordinate in ('effective', 'mu'):
            # Each x sample has its own training fraction and hence scaling.
            scales = np.array([(c.config.q - 1) / c.config.f_train for c in self.cases])
            return self.ridge_lambdas * scales
        raise ValueError("coordinate must be 'lambda' or 'effective'")

    def risk_values(self, normalization):
        if normalization == 'R_infinity':
            scale = self.r_infinity
        elif normalization == 'sigma2':
            scale = np.array([c.config.sigma**2 for c in self.cases])
        elif normalization == 'none':
            scale = np.ones(len(self.cases))
        else:
            raise ValueError("normalization must be 'R_infinity', 'sigma2' or 'none'")
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f'cannot normalize by {normalization}')
        return self.risks / scale


def optimize_fixed_r_sweep(theory, cases, fixed_rs, **search):
    """Reuse the Figure 2 optimizer, then transpose parameter/r axes.

    No interpolation of previously optimized curves: ridge is optimized at
    every requested parameter value and every requested fixed r.
    """
    cases = tuple(cases)
    xs = np.array([case.value for case in cases])
    rs = np.asarray(fixed_rs, float)
    if len(xs) < 2 or np.any(~np.isfinite(xs)) or np.any(np.diff(xs) <= 0):
        raise ValueError('parameter values must be finite, increasing, with at least two samples')
    if rs.ndim != 1 or not rs.size or np.any(~np.isfinite(rs)) or np.any(rs <= 0):
        raise ValueError('fixed_rs must be a nonempty positive finite vector')
    if len(np.unique(rs)) != len(rs):
        raise ValueError('fixed_rs must not contain duplicates')
    by_parameter = optimize_parameter_sweep(theory, cases, rs, **search)
    return tuple(
        FixedRSweepCurve(
            r=float(r), cases=cases,
            ridge_lambdas=np.array([c.optimum.ridge_lambdas[j] for c in by_parameter]),
            risks=np.array([c.optimum.risks[j] for c in by_parameter]),
            status=np.array([c.optimum.boundary[j] for c in by_parameter]),
            r_infinity=np.array([c.r_infinity for c in by_parameter]),
        )
        for j, r in enumerate(rs)
    )


def parameter_transition_evidence(curve, *, n_probe=65, barrier_rtol=1e-8):
    """Apply the existing barrier diagnostic at BOTH parameter endpoints.

    r is fixed, but cfg changes along x. Calling the r-path diagnostic on two
    repeated r entries evaluates the two candidate penalties for one cfg; do
    this independently for each endpoint cfg. An unresolved barrier remains
    inconclusive, and a finite search-cap hit remains a cap hit.
    """
    size = len(curve.cases) - 1
    barriers = np.full((size, 2), np.nan)
    endpoint_reasons = np.empty((size, 2), dtype='<U32')
    jumps = np.zeros(size, dtype=bool)
    for i in range(size):
        pair = SimpleNamespace(
            r_values=np.array([curve.r, curve.r]),
            ridge_lambdas=curve.ridge_lambdas[i:i + 2],
        )
        passed = []
        for j, case in enumerate(curve.cases[i:i + 2]):
            evidence = ridge_transition_evidence(
                pair, case.config, n_probe=n_probe, barrier_rtol=barrier_rtol,
            )
            barriers[i, j] = evidence['barrier_excess'][0, 0]
            endpoint_reasons[i, j] = evidence['reason'][0]
            passed.append(bool(evidence['jump_mask'][0]))
        jumps[i] = all(passed)
    xs = curve.parameter_values
    return dict(
        jump_mask=jumps, barrier_excess=barriers,
        endpoint_reason=endpoint_reasons,
        reason=np.where(jumps, 'barrier_switch', 'inconclusive'),
        parameter_intervals=np.column_stack((xs[:-1], xs[1:])),
        fixed_r=curve.r, n_probe=int(n_probe), barrier_rtol=float(barrier_rtol),
    )
