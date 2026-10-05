from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter
from typing import Callable

import numpy as np

from .config import DMSConfig
from .simulation import GRModelResult, gr_model, gr_trial
from .theory import DMSTTheory


ProgressCallback = Callable[[dict], None]


@dataclass(frozen=True)
class GRJobResult:
    """Output of a generalized risk simulation job.

    data has shape (n_lambda, n_r, 3), with the last axis
    [realized_r, mean_risk, standard_error].
    """

    data: np.ndarray
    ridge_lambdas: np.ndarray
    config: DMSConfig
    measured_parameters: tuple[dict, ...]
    sizes: np.ndarray
    seconds: float
    requested_r: np.ndarray


@dataclass(frozen=True)
class FiniteSizeResult:
    """Repeated GR jobs over different finite-size budgets for one lambda."""

    data: np.ndarray  # (n_budget, n_r, 3)
    budgets: np.ndarray
    ridge_lambda: float
    config: DMSConfig
    sizes: tuple[np.ndarray, ...]
    measured_parameters: tuple[tuple[dict, ...], ...]
    jobs: tuple[GRJobResult, ...]
    r_infinity: float


def _eta_block(eta: float) -> int:
    if eta == 0.0 or eta == 1.0:
        return 1
    # Matches the intent of Mathematica Rationalize[eta, 10^-9] for ordinary
    # decimal eta values while avoiding very large binary-float denominators.
    frac = Fraction(float(eta)).limit_denominator(10**9)
    return int(frac.denominator)


def gr_job_sizes(r_values, cfg: DMSConfig, budget: int) -> np.ndarray:
    """Planned integer (n, L) rows, for feasibility checks before a job."""
    cfg.validate_basic()
    rs = np.asarray(r_values, dtype=float)
    if rs.ndim != 1 or not rs.size or np.any(~np.isfinite(rs)) or np.any(rs <= 0):
        raise ValueError("r_values must be a nonempty vector of finite positive values")
    if not np.isfinite(budget) or budget <= 0 or int(budget) != budget:
        raise ValueError("budget must be a positive integer")
    block = _eta_block(cfg.eta)
    sizes = np.array([
        (max(2, int(np.floor(budget / np.sqrt(r)))),
         block * int(np.floor(budget * np.sqrt(r) / (cfg.q * block))))
        for r in rs
    ], dtype=int)
    if np.any(sizes[:, 1] == 0):
        raise ValueError(f"increase budget; zero-length model encountered: {sizes}")
    return sizes


def gr_job(
    r_values: np.ndarray | list[float],
    ridge_lambdas: np.ndarray | list[float],
    cfg: DMSConfig,
    budget: int,
    trials: int,
    *,
    seed: int = 17,
    progress: ProgressCallback | None = None,
) -> GRJobResult:
    """Python translation of GRJob from clean.nb."""
    cfg.validate_basic()
    rs = np.asarray(r_values, dtype=float)
    lambdas = np.asarray(ridge_lambdas, dtype=float)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if trials < 2:
        raise ValueError("trials must be at least 2 to estimate a standard error")
    if rs.size == 0 or lambdas.size == 0:
        raise ValueError("r_values and ridge_lambdas must be non-empty")
    if np.any(rs <= 0) or np.any(lambdas <= 0):
        raise ValueError("r_values and ridge_lambdas must be positive")

    rng = np.random.default_rng(seed)
    q = cfg.q
    sizes = gr_job_sizes(rs, cfg, budget)

    start = perf_counter()
    data = np.empty((lambdas.size, rs.size, 3), dtype=float)
    measured: list[dict] = []
    total = rs.size * trials
    done = 0

    for j, (n, L) in enumerate(sizes):
        model: GRModelResult = gr_model(int(L), cfg, rng)
        measured.append(model.measured_parameters)

        risks = np.empty((trials, lambdas.size), dtype=float)
        for t in range(trials):
            risks[t] = gr_trial(int(n), lambdas, model, cfg, rng)
            done += 1
            if progress is not None:
                progress(
                    {
                        "done": done,
                        "total": total,
                        "requested_r": float(rs[j]),
                        "realized_r": float(q * L / n),
                        "n": int(n),
                        "L": int(L),
                        "trial": t + 1,
                    }
                )

        realized_r = float(q * L / n)
        means = risks.mean(axis=0)
        ses = risks.std(axis=0, ddof=1) / np.sqrt(trials)
        data[:, j, 0] = realized_r
        data[:, j, 1] = means
        data[:, j, 2] = ses

    return GRJobResult(
        data=data,
        ridge_lambdas=lambdas.copy(),
        config=cfg,
        measured_parameters=tuple(measured),
        sizes=sizes,
        seconds=float(perf_counter() - start),
        requested_r=rs.copy(),
    )


def finite_size_job(
    r_values: np.ndarray | list[float],
    ridge_lambda: float,
    cfg: DMSConfig,
    n_runs: int,
    budgets: np.ndarray | list[int],
    *,
    seed: int = 17,
    progress: ProgressCallback | None = None,
) -> FiniteSizeResult:
    budgets_arr = np.asarray(budgets, dtype=int)
    if np.any(budgets_arr <= 0):
        raise ValueError("all budgets must be positive")

    jobs: list[GRJobResult] = []
    for i, budget in enumerate(budgets_arr):
        job = gr_job(
            r_values,
            [ridge_lambda],
            cfg,
            int(budget),
            n_runs,
            seed=seed + i,
            progress=progress,
        )
        jobs.append(job)

    data = np.stack([job.data[0] for job in jobs], axis=0)
    return FiniteSizeResult(
        data=data,
        budgets=budgets_arr.copy(),
        ridge_lambda=float(ridge_lambda),
        config=cfg,
        sizes=tuple(job.sizes for job in jobs),
        measured_parameters=tuple(job.measured_parameters for job in jobs),
        jobs=tuple(jobs),
        r_infinity=DMSTTheory.infinity_risk(cfg),
    )


def theory_curves(
    theory: DMSTTheory,
    ridge_lambdas: np.ndarray | list[float],
    r_grid: np.ndarray | list[float],
    cfg: DMSConfig,
    *,
    normalize_by_sigma2: bool = True,
) -> np.ndarray:
    return theory.curves(
        np.asarray(ridge_lambdas, dtype=float),
        np.asarray(r_grid, dtype=float),
        cfg,
        normalize_by_sigma2=normalize_by_sigma2,
    )
