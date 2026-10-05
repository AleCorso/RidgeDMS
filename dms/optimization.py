"""Physical-ridge search with local refinement in log(lambda)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DMSConfig
from .theory import DMSTTheory, TheoryEvaluationError


@dataclass(frozen=True)
class RidgeOptimizationResult:
    r_values: np.ndarray
    ridge_lambdas: np.ndarray
    risks: np.ndarray
    boundary: np.ndarray  # 'lower', 'upper', or 'interior'
    evaluations: np.ndarray
    coarse_lambdas: np.ndarray
    coarse_risks: np.ndarray

    def interpolate_lambda(self, r):
        """First-order log-log interpolation; extrapolation is rejected."""
        values = np.asarray(r, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < self.r_values[0]) or np.any(values > self.r_values[-1]):
            raise ValueError("interpolation requires r inside the sampled range")
        out = np.exp(np.interp(np.log(values), np.log(self.r_values), np.log(self.ridge_lambdas)))
        return float(out) if out.ndim == 0 else out


def geometric_ridge_grid(lambda_min=1e-6, lambda_max=1.0, n_grid=41):
    if not (np.isfinite(lambda_min) and np.isfinite(lambda_max) and 0 < lambda_min < lambda_max):
        raise ValueError("require finite 0 < lambda_min < lambda_max")
    if int(n_grid) != n_grid or n_grid < 3:
        raise ValueError("n_grid must be an integer >= 3")
    return np.geomspace(lambda_min, lambda_max, int(n_grid))


def _golden_minimize(function, left, right, tol, max_iter):
    ratio = (np.sqrt(5) - 1) / 2
    c, d = right - ratio * (right - left), left + ratio * (right - left)
    fc, fd = function(c), function(d)
    for _ in range(max_iter):
        if right - left <= tol:
            break
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = function(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = function(d)
    else:
        raise RuntimeError("ridge refinement did not converge; increase max_iter")
    return (c, fc) if fc < fd else (d, fd)


def optimize_ridge_over_r(
    theory: DMSTTheory, r_values, cfg: DMSConfig, *,
    lambda_min=1e-6, lambda_max=1.0, n_grid=41,
    refine=True, log_tolerance=1e-6, max_iter=100,
) -> RidgeOptimizationResult:
    """Grid search, then bounded log-lambda refinement around its best point.

    Boundary grid points are retained as candidates even when refining the
    neighboring interval. The method is a local refinement, not a guarantee
    of the global optimum for arbitrary multimodal custom risks.
    """
    rs = np.asarray(r_values, dtype=float)
    if rs.ndim != 1 or not rs.size or np.any(~np.isfinite(rs)) or np.any(rs <= 0) or np.any(np.diff(rs) <= 0):
        raise ValueError("r_values must be finite, positive and strictly increasing")
    if not np.isfinite(log_tolerance) or log_tolerance <= 0 or max_iter < 1:
        raise ValueError("log_tolerance and max_iter must be positive")
    grid = geometric_ridge_grid(lambda_min, lambda_max, n_grid)
    logs = np.log(grid)
    coarse = np.empty((rs.size, grid.size))
    best_lambdas, best_risks, boundaries, counts = [], [], [], []
    for i, r in enumerate(rs):
        count = 0
        def evaluate(lam):
            nonlocal count
            count += 1
            risk = theory.risk(float(lam), float(r), cfg)
            if not np.isfinite(risk):
                raise TheoryEvaluationError(f"Non-finite optimization risk at lambda={lam}, r={r}")
            return risk
        coarse[i] = [evaluate(lam) for lam in grid]
        k = int(np.argmin(coarse[i]))
        best_lambda, best_risk = grid[k], coarse[i, k]
        if refine:
            u, value = _golden_minimize(
                lambda u: evaluate(np.exp(u)), logs[max(0, k - 1)],
                logs[min(grid.size - 1, k + 1)], log_tolerance, max_iter,
            )
            if value < best_risk:
                best_lambda, best_risk = np.exp(u), value
        boundary = ('lower' if best_lambda <= lambda_min * np.exp(log_tolerance)
                    else 'upper' if best_lambda >= lambda_max * np.exp(-log_tolerance)
                    else 'interior')
        best_lambdas.append(best_lambda)
        best_risks.append(best_risk)
        boundaries.append(boundary)
        counts.append(count)
    return RidgeOptimizationResult(rs.copy(), np.array(best_lambdas), np.array(best_risks),
                                   np.array(boundaries), np.array(counts), grid, coarse)
