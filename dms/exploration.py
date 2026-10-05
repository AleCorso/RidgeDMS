"""One-coordinate parameter sweeps with explicit constraints and mean conventions.

Research defaults belong in notebooks. These utilities map experimental
coordinates to DMSConfig, check moment feasibility, and collect ridge optima.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Mapping

import numpy as np

from .config import DMSConfig
from .optimization import RidgeOptimizationResult, optimize_ridge_over_r
from .ridge_diagnostics import BoundaryRidgeOptimization, optimize_effective_ridge_over_r
from .simulation import moment_feasibility
from .theory import DMSTTheory


@dataclass(frozen=True)
class SweepParameters:
    q: int
    sigma: float
    G2: float
    f: float
    ftilde: float
    eta: float
    F: float
    FTest: float
    Fneq: float
    m: float
    mneq: float
    mneqtilde: float
    Cneq: float | None = None
    rhoneq: float | None = None

    def to_config(self, *, mean_convention: str, reference_L: int) -> DMSConfig:
        """Map means using raw moments or normalized mean coordinates.

        With gamma=sqrt((q-1)/q), normalized means are gamma*m*sqrt(L*F),
        gamma*mneq*sqrt(eta*L*Fneq), and
        gamma*mneqtilde*sqrt(eta*L*(FTest-F+Fneq)). No clipping.
        reference_L fixes the mapping; it does not vary with r.
        """
        if any(value is not None and not np.isfinite(value) for value in asdict(self).values()):
            raise ValueError('all sweep parameters must be finite')
        if self.G2 <= 0 or not 0 <= self.F <= 1 or not 0 <= self.FTest <= 1:
            raise ValueError('require G2>0 and F,FTest in [0,1]')
        if int(reference_L) != reference_L or reference_L <= 0:
            raise ValueError('reference_L must be a positive integer')
        if self.q != int(self.q):
            raise ValueError('q must be an integer')
        if not 0 <= self.eta <= 1 or self.Fneq < 0 or self.Fneq > self.F:
            raise ValueError('require eta in [0,1] and 0 <= Fneq <= F')
        test_neq = self.FTest - self.F + self.Fneq
        if test_neq < -1e-12:
            raise ValueError('FTest-F+Fneq must be nonnegative')
        if (self.Cneq is None) == (self.rhoneq is None):
            raise ValueError('specify exactly one of Cneq or rhoneq')
        if self.rhoneq is not None:
            if not -1 <= self.rhoneq <= 1:
                raise ValueError('rhoneq must lie in [-1,1]')
            mismatch_scale = np.sqrt(self.Fneq * max(0.0, test_neq))
            if mismatch_scale == 0 and self.rhoneq != 0:
                raise ValueError('rhoneq is undefined for zero mismatch energy; use zero for the constrained case')
            cneq = self.rhoneq * mismatch_scale
        else:
            cneq = self.Cneq
        if mean_convention == 'normalized':
            masses = np.array([reference_L * self.F,
                               self.eta * reference_L * self.Fneq,
                               self.eta * reference_L * max(0.0, test_neq)])
            means = np.array([self.m, self.mneq, self.mneqtilde])
            if np.any((masses == 0) & (means != 0)):
                raise ValueError('a normalized mean is undefined for a zero-energy/site class; set it to zero')
            mt, mn, mnt = np.sqrt((self.q - 1.) / self.q) * means * np.sqrt(masses)
        elif mean_convention == 'raw':
            mt, mn, mnt = self.m, self.mneq, self.mneqtilde
        else:
            raise ValueError("mean_convention must be 'normalized' or 'raw'")
        cfg = DMSConfig(
            q=int(self.q), sigma=self.sigma, signal_norm=float(np.sqrt(self.G2)),
            f_train=self.f, f_test=self.ftilde, eta=self.eta,
            F=self.F, F_test=self.FTest, F_neq=self.Fneq, C_neq=cneq,
            M_tot=float(mt), M_neq=float(mn), M_neq_test=float(mnt),
        )
        cfg.validate_theory()
        return cfg


@dataclass(frozen=True)
class SweepCase:
    parameter: str
    value: float
    parameters: SweepParameters
    config: DMSConfig
    reference_L: int


@dataclass(frozen=True)
class OptimizedSweepCurve:
    case: SweepCase
    optimum: RidgeOptimizationResult | BoundaryRidgeOptimization
    r_infinity: float

    def risk_values(self, normalization: str) -> np.ndarray:
        if normalization == 'R_infinity':
            scale = self.r_infinity
        elif normalization == 'sigma2':
            scale = self.case.config.sigma**2
        elif normalization == 'none':
            scale = 1.0
        else:
            raise ValueError("normalization must be 'R_infinity', 'sigma2' or 'none'")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f'cannot normalize risk by {normalization}={scale}')
        return self.optimum.risks / scale

    def regularization_values(self, coordinate: str) -> np.ndarray:
        if coordinate == 'lambda':
            return self.optimum.ridge_lambdas.copy()
        if coordinate == 'mu':
            cfg = self.case.config
            return self.optimum.ridge_lambdas * (cfg.q - 1) / cfg.f_train
        raise ValueError("coordinate must be 'lambda' (physical) or 'mu' (scaled)")


def parameter_sweep_cases(
    baseline: SweepParameters, parameter: str, values, *,
    mean_convention: str, reference_L: int,
    fixed: Mapping[str, float] | None = None,
    ties: Mapping[str, str] | None = None,
) -> tuple[SweepCase, ...]:
    """Change one independent coordinate, then apply explicitly declared ties.

    Every case is checked at reference_L before raising a combined error listing
    all infeasible values and their reasons. No moments are adjusted to pass.
    Tied targets may not be swept, and chained/cyclic ties are rejected.
    parameter='L' varies the reference length for mean conversion and feasibility,
    holding all SweepParameters fixed. It does not set a simulation size.
    """
    fixed, ties = dict(fixed or {}), dict(ties or {})
    names = {field.name for field in fields(SweepParameters)}
    unknown = (({parameter} - {'L'}) | fixed.keys() | ties.keys() | set(ties.values())) - names
    if unknown:
        raise ValueError(f'unknown parameter names: {sorted(unknown)}')
    if parameter in fixed or parameter in ties:
        raise ValueError(f'{parameter} is fixed or tied, so it cannot be varied independently')
    if ties.keys() & set(ties.values()) or fixed.keys() & ties.keys():
        raise ValueError('ties must be independent, unambiguous and unchained')
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or np.any(~np.isfinite(values)):
        raise ValueError('values must be a nonempty finite vector')
    out, errors = [], []
    for value in values:
        changes = dict(fixed) if parameter == 'L' else {**fixed, parameter: float(value)}
        # Selecting a cross-moment coordinate replaces the alternative coordinate.
        if 'rhoneq' in changes and 'Cneq' not in changes:
            changes['Cneq'] = None
        elif 'Cneq' in changes and 'rhoneq' not in changes:
            changes['rhoneq'] = None
        params = replace(baseline, **changes)
        params = replace(params, **{target: getattr(params, source) for target, source in ties.items()})
        try:
            case_L = float(value) if parameter == 'L' else reference_L
            cfg = params.to_config(mean_convention=mean_convention, reference_L=case_L)
            report = moment_feasibility(int(case_L), cfg)
            if not report.feasible:
                raise ValueError('; '.join(report.messages))
        except ValueError as exc:
            errors.append(f'infeasible {parameter}={value:g}: {exc}')
            continue
        out.append(SweepCase(parameter, float(value), params, cfg, int(case_L)))
    if errors:
        raise ValueError('\n'.join(errors))
    return tuple(out)


def optimize_parameter_sweep(
    theory: DMSTTheory, cases: tuple[SweepCase, ...], r_values,
    *, method: str = 'positive', **search_options,
) -> tuple[OptimizedSweepCurve, ...]:
    """Minimize ridge using a positive search or the exact nonnegative theory.

    method='nonnegative' uses the project's rational risk, includes zero,
    and accepts lambda_max/n_grid. It does not use a custom theory transform.
    method='positive' preserves the supplied theory and legacy search options.
    """
    if method not in ('positive', 'nonnegative'):
        raise ValueError("method must be 'positive' or 'nonnegative'")
    return tuple(
        OptimizedSweepCurve(case,
                            (optimize_effective_ridge_over_r(r_values, case.config, **search_options)
                             if method == 'nonnegative' else
                             optimize_ridge_over_r(theory, r_values, case.config, **search_options)),
                            theory.infinity_risk(case.config))
        for case in cases
    )
