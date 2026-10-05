"""Exact effective-ridge coordinates and diagnostics for zero-ridge optima.

kappa=mu/x, x=1-r E[t/(t+kappa)], mu=kappa*x. The eigenvalue
weights are (q-2)/q at 1, 1/q at q(1-f_train), and 1/q at zero.
The general risk is rational in kappa; no cubic solve is required per point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DMSConfig
from .stieltjes import project_resolvent_state, project_stieltjes_evaluator
from .theory import DMSTTheory, TheoryEvaluationError


@dataclass(frozen=True)
class EffectiveRisk:
    kappa: float
    ridge_lambda: float
    mu: float
    x: float
    denominator: float
    bias: float
    variance: float
    risk: float
    d_bias_d_kappa: float
    d_variance_d_kappa: float
    d_risk_d_lambda: float


def zero_ridge_kappa(r: float, cfg: DMSConfig) -> float:
    """Left endpoint of the physical kappa domain (lambda=0)."""
    cfg.validate_theory()
    if not np.isfinite(r) or r <= 0:
        raise ValueError('r must be finite and positive')
    q, t = cfg.q, cfg.q * (1 - cfg.f_train)
    b = 1 + t - r * (q - 2 + t) / q
    c = t * (1 - r * (q - 1) / q)
    if t == 0:
        return max(0.0, -b)
    if c >= 0:
        return 0.0
    disc = np.hypot(b, 2 * np.sqrt(-c))
    return float(-2 * c / (b + disc) if b >= 0 else (disc - b) / 2)


def kappa_from_lambda(ridge_lambda: float, r: float, cfg: DMSConfig) -> float:
    cfg.validate_theory()
    if not np.isfinite(ridge_lambda) or ridge_lambda < 0:
        raise ValueError('ridge_lambda must be finite and nonnegative')
    if ridge_lambda == 0:
        return zero_ridge_kappa(r, cfg)
    mu = ridge_lambda * (cfg.q - 1) / cfg.f_train
    state = project_resolvent_state(mu, r, cfg.q, cfg.f_train)
    return mu / state.x


def effective_risk(kappa: float, r: float, cfg: DMSConfig) -> EffectiveRisk:
    """Exact current general theory and analytic slopes, including lambda=0.

    At the interpolation threshold, the noisy zero-ridge limit diverges and
    raises; approach it from kappa>0 instead. This function does not extend
    the estimator to negative ridge.
    """
    lower = zero_ridge_kappa(r, cfg)
    if not np.isfinite(kappa) or kappa < lower:
        raise ValueError(f'kappa must be finite and >= {lower}')
    q, t = cfg.q, cfg.q * (1 - cfg.f_train)
    if kappa == 0 and t == 0:
        # The zero eigenvalue carries no response; its signal contribution
        # vanishes because its train/test/mismatch weights also vanish only
        # in special cases. Keep this endpoint outside the diagnostic API.
        raise ValueError('kappa=0 with f_train=1 requires a separate zero-eigenvalue limit')
    u, w = 1 / (1 + kappa), 1 / (t + kappa)
    x = 1 - r / q * ((q - 2) * u + t * w)
    if kappa == lower and lower > 0:
        x = 0.0
    a1 = r / q * ((q - 2) * u**2 + t**2 * w**2)
    da1 = -2 * r / q * ((q - 2) * u**3 + t**2 * w**3)
    denom = 1 - a1
    if denom <= 32 * np.finfo(float).eps:
        raise ValueError('zero-ridge risk diverges at the interpolation threshold')
    cm, cw = DMSTTheory._trace_weights(cfg)
    dm, dw = DMSTTheory._tracedelta_weights(cfg)
    cm += (q - 1) / cfg.f_test * dm
    cw += (q - 1) / cfg.f_test * dw
    a23 = r * (cm * u**2 + t * cw * w**2)
    da23 = -2 * r * (cm * u**3 + t * cw * w**3)
    g2 = cfg.signal_norm**2
    b1 = g2 * ((1 - cfg.F) * u**2 + cfg.F * t * w**2)
    db1 = -2 * g2 * ((1 - cfg.F) * u**3 + cfg.F * t * w**3)
    th = DMSTTheory(project_stieltjes_evaluator())
    test_t = q * (1 - cfg.f_test)
    bw = test_t * th.g000(cfg) + th.g001(cfg)
    bm = th.g111(cfg) + test_t * th.g110(cfg)
    cross = th.g011(cfg) + test_t * th.g010(cfg)
    shift_w, shift_m = th.g00_delta(cfg), th.g11_delta(cfg)
    shift = shift_w * w + shift_m * u
    b23 = bw * w**2 + bm * u**2 + 2 * cross * u * w + (q - 1) / cfg.f_test * shift**2
    db23 = (-2 * bw * w**3 - 2 * bm * u**3 - 2 * cross * (u**2 * w + u * w**2)
            - 2 * (q - 1) / cfg.f_test * shift * (shift_w * w**2 + shift_m * u**2))
    ratio = a23 / denom
    dratio = (da23 * denom + a23 * da1) / denom**2
    c, noise = cfg.f_test / (q - 1), cfg.sigma**2 * cfg.f_test / cfg.f_train
    bias = c * kappa**2 * (b23 + b1 * ratio)
    variance = noise * ratio
    dbias = c * (2 * kappa * (b23 + b1 * ratio)
                 + kappa**2 * (db23 + db1 * ratio + b1 * dratio))
    dvariance = noise * dratio
    mu = kappa * x
    lam = mu * cfg.f_train / (q - 1)
    slope = (dbias + dvariance) / denom * (q - 1) / cfg.f_train
    return EffectiveRisk(kappa, float(lam), float(mu), float(x), float(denom),
                         float(bias), float(variance), float(bias + variance),
                         float(dbias), float(dvariance), float(slope))


def _require_matched(cfg):
    if not (cfg.eta == 0 and cfg.f_train == cfg.f_test and cfg.F == cfg.F_test
            and cfg.F_neq == cfg.C_neq == cfg.M_neq == cfg.M_neq_test == 0):
        raise ValueError('this comparison requires matched train/test distributions and signal moments')


def commuted_matched_risk(ridge_lambda, r, cfg, *, legacy_weight=False):
    """Old scalar-derivative approximation, for comparison only.

    The normalized version uses the same population covariance as the exact
    theory. legacy_weight=True also reproduces the historical strong-mode
    coefficient (q-1)*(1-f) instead of q*(1-f). In old notebooks `norm`/`G`
    in this formula denotes signal_norm**2, not signal_norm.
    """
    _require_matched(cfg)
    k = kappa_from_lambda(ridge_lambda, r, cfg)
    point = effective_risk(k, r, cfg)
    t = cfg.q * (1 - cfg.f_train)
    u, w = 1 / (1 + k), 1 / (t + k)
    dx_dk = r / cfg.q * ((cfg.q - 2) * u**2 + t * w**2)
    dx_dmu = dx_dk / point.denominator
    weight_w = t * ((cfg.q - 1) / cfg.q if legacy_weight else 1)
    bias = (cfg.f_train / (cfg.q - 1) * cfg.signal_norm**2 * k**2
            * (cfg.F * weight_w * w**2 * (1 + t * dx_dmu)
               + (1 - cfg.F) * u**2 * (1 + dx_dmu)))
    return float(bias + point.variance)


@dataclass(frozen=True)
class BoundaryRidgeOptimization:
    r_values: np.ndarray
    ridge_lambdas: np.ndarray
    kappas: np.ndarray
    risks: np.ndarray
    status: np.ndarray  # zero, interior, upper
    zero_slopes: np.ndarray  # dR/d physical lambda at 0+, -inf at divergence

    @property
    def boundary(self):
        """Common status interface for parameter sweeps and exports."""
        return self.status


def optimize_effective_ridge_over_r(r_values, cfg, *, lambda_max=1e3, n_grid=161):
    """Search rational risk with exact zero-ridge candidate and analytic slopes.

    Bracket all slope sign changes resolved by the log-offset grid, bisect
    their stationary points, and compare with both endpoints. Increasing
    n_grid checks resolution; arbitrarily narrow extrema can still be missed.
    No positive lambda_min is imposed. At the critical aspect ratio zero is
    divergent and is excluded. An upper result is a finite search-bound hit.
    """
    rs = np.asarray(r_values, dtype=float)
    if rs.ndim != 1 or not rs.size or np.any(~np.isfinite(rs)) or np.any(rs <= 0):
        raise ValueError('r_values must be a nonempty positive finite vector')
    if not np.isfinite(lambda_max) or lambda_max <= 0 or n_grid < 3:
        raise ValueError('require lambda_max>0 and n_grid>=3')
    bests, statuses, zero_slopes = [], [], []
    for r in rs:
        lo, hi = zero_ridge_kappa(float(r), cfg), kappa_from_lambda(lambda_max, float(r), cfg)
        try:
            zero = effective_risk(lo, float(r), cfg)
        except ValueError:
            if cfg.f_train < 1 and abs(r * (cfg.q - 1) / cfg.q - 1) < 1e-14:
                zero = None
            else:
                raise
        zero_slopes.append(zero.d_risk_d_lambda if zero is not None else -np.inf)
        delta = max(32 * np.spacing(max(1, lo)), 1e-12 * max(1, lo))
        if hi - lo <= delta:
            raise ValueError('lambda_max is too close to zero for the requested floating-point search')
        grid = lo + np.geomspace(delta, hi - lo, n_grid)
        points = [effective_risk(float(k), float(r), cfg) for k in grid]
        if zero is not None:
            points.insert(0, zero)
        candidates = points.copy()
        for left, right in zip(points[:-1], points[1:]):
            if np.signbit(left.d_risk_d_lambda) == np.signbit(right.d_risk_d_lambda):
                continue
            for _ in range(70):
                mid = effective_risk((left.kappa + right.kappa) / 2, float(r), cfg)
                if right.kappa - left.kappa < 1e-11 * max(1, mid.kappa):
                    break
                if np.signbit(mid.d_risk_d_lambda) == np.signbit(left.d_risk_d_lambda):
                    left = mid
                else:
                    right = mid
            candidates.append(mid)
        best = min(candidates, key=lambda point: point.risk)
        if zero is not None and zero.d_risk_d_lambda >= 0 and zero.risk <= best.risk + 1e-12 * max(1, abs(best.risk)):
            best = zero
        if best.risk < -1e-10 or not np.isfinite(best.risk):
            raise TheoryEvaluationError(f'nonphysical effective-ridge optimum at r={r}: {best}')
        bests.append(best)
        statuses.append('zero' if best is zero else 'upper' if np.isclose(best.kappa, hi, rtol=1e-10) else 'interior')
    return BoundaryRidgeOptimization(rs.copy(), np.array([p.ridge_lambda for p in bests]),
                                     np.array([p.kappa for p in bests]), np.array([p.risk for p in bests]),
                                     np.array(statuses), np.array(zero_slopes))


@dataclass(frozen=True)
class GaussianRidgeCheck:
    ridge_lambdas: np.ndarray
    risks: np.ndarray  # conditional on each sampled design, noise averaged analytically
    zero_bias_slopes: np.ndarray
    zero_variance_slopes: np.ndarray
    realized_r: float
    n: int
    p: int

    @property
    def means(self):
        return self.risks.mean(axis=0)

    @property
    def standard_errors(self):
        return self.risks.std(axis=0, ddof=1) / np.sqrt(self.risks.shape[0])


def matched_gaussian_check(cfg, *, L, r, ridge_lambdas, trials, seed):
    """Independent finite Gaussian-design check of the matched covariance model.

    Uses the same two nonzero population eigenvalues/energy fractions as DMS.
    This is a Gaussian surrogate, not a one-hot DMS simulation. Noise variance
    is integrated exactly for each design. Shared designs give paired risk
    differences; returned arrays retain every design's risk and boundary slope.
    The zero-mean population is known and no sample intercept is fitted.
    """
    _require_matched(cfg)
    cfg.validate_theory()
    if L < 1 or int(L) != L or trials < 2 or int(trials) != trials:
        raise ValueError('L must be a positive integer and trials an integer >=2')
    if r <= 0 or not np.isfinite(r):
        raise ValueError('r must be finite and positive')
    lambdas = np.asarray(ridge_lambdas, dtype=float)
    if lambdas.ndim != 1 or not lambdas.size or np.any(~np.isfinite(lambdas)) or np.any(lambdas < 0):
        raise ValueError('ridge_lambdas must be a finite nonnegative vector')
    n, p = max(1, int(round(cfg.q * L / r))), (cfg.q - 1) * L
    if n >= p or cfg.f_train >= 1:
        raise ValueError('this zero-ridge matrix check requires n<p and f_train<1')
    eigenvalues = np.concatenate([np.full(L, cfg.q * (1 - cfg.f_train)),
                                 np.ones((cfg.q - 2) * L)]) * cfg.f_train / (cfg.q - 1)
    beta = np.zeros(p)
    beta[0], beta[L] = cfg.signal_norm * np.sqrt(cfg.F), cfg.signal_norm * np.sqrt(1 - cfg.F)
    rng = np.random.default_rng(seed)
    risks, bias_slopes, variance_slopes = [], [], []
    for _ in range(trials):
        design = rng.normal(size=(n, p)) * np.sqrt(eigenvalues)
        values, vectors = np.linalg.eigh(design @ design.T / n)
        if values[0] <= 0:
            raise RuntimeError('Gaussian Gram matrix is numerically singular')
        v = (design.T @ vectors) / np.sqrt(n * values)
        projection = v.T @ beta
        test_diagonal = (eigenvalues[:, None] * v**2).sum(axis=0)
        factors = values[:, None] / (values[:, None] + lambdas[None, :])
        errors = beta[:, None] - v @ (factors * projection[:, None])
        bias = (eigenvalues[:, None] * errors**2).sum(axis=0)
        variance = cfg.sigma**2 / n * (test_diagonal[:, None] * values[:, None]
                                      / (values[:, None] + lambdas[None, :])**2).sum(axis=0)
        null_beta, plus_beta = beta - v @ projection, v @ (projection / values)
        bias_slopes.append(2 * np.dot(eigenvalues * null_beta, plus_beta))
        variance_slopes.append(-2 * cfg.sigma**2 / n * np.sum(test_diagonal / values**2))
        risks.append(bias + variance)
    return GaussianRidgeCheck(lambdas.copy(), np.array(risks), np.array(bias_slopes),
                              np.array(variance_slopes), cfg.q * L / n, n, p)


@dataclass(frozen=True)
class CommutedMatchedTheory:
    """Adapter for running the existing grid optimizer on the old approximation."""
    legacy_weight: bool = False

    def risk(self, ridge_lambda, r, cfg):
        return commuted_matched_risk(ridge_lambda, r, cfg, legacy_weight=self.legacy_weight)
