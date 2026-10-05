"""Real-negative Stieltjes branch of the project's defining equation.

The original clean.nb uses a full-q spectrum, including mass 1/q at zero:

    m = -1/(q*z) + 1/(q*A) + (q-2)/(q*B),
    A = q*(1-f)*(1-r-r*z*m)-z, B = 1-r-r*z*m-z.

The implementation brief omitted the zero mass and q in A; Mathematica
parity demonstrates that discrepancy. The cubic is solved in x=1-r-r*z*m. Its physical root is positive and
separated from the negative rational poles, even at tiny ridge and large r.
This affine change of variable avoids nearly coincident roots in m.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

MCallable = Callable[[complex, float, int, float], complex]
DMCallable = MCallable


class StieltjesEvaluationError(RuntimeError):
    """The physical branch could not be resolved reliably."""


def _A(m, z, r, q, f):
    return q * (1.0 - f) * (1.0 - r - r * m * z) - z


def _B(m, z, r):
    return 1.0 - r - r * m * z - z


def _P(m, z, r, q, f):
    a, b = _A(m, z, r, q, f), _B(m, z, r)
    return q * (m + 1 / (q * z)) * a * b - b - (q - 2.0) * a


def _fixed_point_residual(m, z, r, q, f):
    return m + 1 / (q * z) - 1 / (q * _A(m, z, r, q, f)) - (
        (q - 2) / (q * _B(m, z, r))
    )


def _cubic_coefficients(z, r, q, f):
    a0, a1 = q * (1 - f) * (1 - r) - z, q * (1 - f) * r * z
    b0, b1 = 1 - r - z, r * z
    cross = a0 * b1 + a1 * b0
    return np.array([
        q * a1 * b1,
        -q * cross + a1 * b1 / z,
        q * a0 * b0 - cross / z + b1 + (q - 2) * a1,
        a0 * b0 / z - b0 - (q - 2) * a0,
    ], dtype=float)


def _validate(z, r, q, f):
    if np.ndim(z) != 0 or np.iscomplexobj(z) and np.imag(z) != 0:
        raise ValueError("project Stieltjes solver supports only real z < 0")
    z = float(np.real(z))
    if not np.isfinite(z) or z >= 0:
        raise ValueError("project Stieltjes solver requires finite real z < 0")
    if not np.isfinite(r) or r < 0:
        raise ValueError("r must be finite and nonnegative")
    if not np.isfinite(q) or q < 3 or int(q) != q:
        raise ValueError("q must be an integer >= 3")
    if not np.isfinite(f) or not 0 <= f <= 1:
        raise ValueError("f must lie in [0, 1]")
    return z


def _x_roots(z, r, q, f):
    # With s=-z, t=q*(1-f), the defining equation becomes
    # [q*x+(q-1)*r-q]*(t*x+s)*(x+s) - r*s*(x+s)
    #     - r*s*(q-2)*(t*x+s) = 0.
    # Solve in y=x/(1+s) to keep degree detection scale-aware.
    s, t = -z, q * (1 - f)
    k = (q - 1) * r - q
    unit = 1 + s
    coeffs = np.array([
        q * t * unit**3,
        (q * s * (t + 1) + k * t) * unit**2,
        (q * s**2 + s * ((q - 2) * r - q + (r - q) * t)) * unit,
        -q * s**2,
    ], dtype=float)
    scale = np.max(np.abs(coeffs))
    if not np.isfinite(scale) or scale == 0:
        raise StieltjesEvaluationError("Cubic coefficients exceed floating-point range")
    coeffs /= scale
    significant = np.flatnonzero(np.abs(coeffs) > 8 * np.finfo(float).eps)
    if not significant.size or significant[0] == 3:
        raise StieltjesEvaluationError("Degenerate Stieltjes polynomial")
    return np.roots(coeffs[significant[0]:]) * unit


def _all_roots(z, r, q, f):
    roots = _x_roots(z, r, q, f)
    if r == 0:
        return np.array([_m_from_x(1.0, z, q, f)])
    return (roots - 1 + r) / (-r * z)


def _m_from_x(x, z, q, f):
    return -1 / (q * z) + 1 / (q * (q * (1 - f) * x - z)) + (q - 2) / (q * (x - z))


def stieltjes_roots(z, r, q, f):
    """All algebraic roots (including nonphysical roots), for diagnostics."""
    z = _validate(z, r, q, f)
    return _all_roots(z, r, q, f)


def _dP_dm(m, z, r, q, f, *, x=None):
    a, b = (_A(m, z, r, q, f), _B(m, z, r)) if x is None else (q * (1 - f) * x - z, x - z)
    am, bm = -q * (1 - f) * r * z, -r * z
    return q * (a * b + (m + 1 / (q * z)) * (am * b + a * bm)) - bm - (q - 2) * am


def _dP_dz(m, z, r, q, f, *, x=None):
    a, b = (_A(m, z, r, q, f), _B(m, z, r)) if x is None else (q * (1 - f) * x - z, x - z)
    az, bz = -q * (1 - f) * r * m - 1, -r * m - 1
    return -a * b / z**2 + q * (m + 1 / (q * z)) * (az * b + a * bz) - bz - (q - 2) * az


def _valid_states(roots, z, r, q, f):
    states = []
    for root in roots:
        if not np.isfinite(root) or abs(root.imag) > 1e-9 * max(abs(root.real), np.finfo(float).tiny):
            continue
        x = float(root.real)
        if x <= 0:
            continue
        # Polish the monotone original equation in x. Extended precision here
        # protects its cancellation at very large r; no extra cubic solves.
        x, zz, rr = np.longdouble(x), np.longdouble(z), np.longdouble(r)
        t = np.longdouble(q) * (1 - np.longdouble(f))
        for _ in range(3):
            value = -1 / (q * zz) + 1 / (q * (t * x - zz)) + (q - 2) / (q * (x - zz))
            residual = x - 1 + rr + rr * zz * value
            slope = 1 - rr * zz * (t / (q * (t * x - zz)**2) + (q - 2) / (q * (x - zz)**2))
            new_x = x - residual / slope
            if new_x <= 0:
                break
            x = new_x
        x = float(x)
        m = float(_m_from_x(x, z, q, f))
        # This is the original rational equation after the affine substitution,
        # not merely the cleared polynomial, so poles cannot pass this check.
        residual = x - 1 + r + r * z * m
        if abs(residual) > 1e-11 + 1e-9 * max(1, r):
            continue
        if not np.isfinite(m) or not 1 / q - 1e-9 <= -z * m <= 1 + 1e-9:
            continue
        if not any(abs(x - other[1]) <= 1e-10 * abs(x) for other in states):
            states.append((m, x))
    return states


def _continuation_state(z, r, q, f, *, steps=64):
    z = _validate(z, r, q, f)
    start = max(100.0, 10 * abs(z))
    previous = 1.0 / start
    for magnitude in np.geomspace(start, -z, steps):
        current_z = -float(magnitude)
        roots = _x_roots(current_z, r, q, f)
        candidates = _valid_states(roots, current_z, r, q, f)
        if not candidates:
            raise StieltjesEvaluationError(
                f"No physical branch during continuation: z={current_z}, r={r}, q={q}, f={f}, x_roots={roots}"
            )
        state = min(candidates, key=lambda state: abs(state[0] - previous))
        previous = state[0]
    return state


def _continuation_m(z, r, q, f, *, steps=64):
    """Rare fallback: track the asymptotic branch toward a negative target."""
    return _continuation_state(z, r, q, f, steps=steps)[0]


@lru_cache(maxsize=32768)
def _project_state(z, r, q, f):
    z = _validate(z, r, q, f)
    candidates = _valid_states(_x_roots(z, r, q, f), z, r, q, f)
    if len(candidates) == 1:
        return candidates[0]
    return _continuation_state(z, r, q, f)


def project_stieltjes_m(z, r, q, f):
    """Physical m(z,r,q,f) for real z<0; cached by the exact scalar state."""
    return _project_state(z, r, q, f)[0]


project_stieltjes_m.cache_info = _project_state.cache_info
project_stieltjes_m.cache_clear = _project_state.cache_clear


def project_stieltjes_dm_dz(z, r, q, f):
    """Exact implicit derivative -P_z/P_m, with the same cached branch."""
    m, x = _project_state(z, r, q, f)
    z = float(np.real(z))
    pm = _dP_dm(m, z, r, q, f, x=x)
    a, b = q * (1 - f) * x - z, x - z
    am, bm = -q * (1 - f) * r * z, -r * z
    v = m + 1 / (q * z)
    scale = q * (abs(a * b) + abs(v * am * b) + abs(v * a * bm)) + abs(bm) + (q - 2) * abs(am)
    if not np.isfinite(pm) or abs(pm) <= 64 * np.finfo(float).eps * scale:
        raise StieltjesEvaluationError(f"Ill-conditioned implicit derivative: P_m={pm}, z={z}, r={r}, q={q}, f={f}")
    derivative = -_dP_dz(m, z, r, q, f, x=x) / pm
    if not np.isfinite(derivative) or derivative <= 0:
        raise StieltjesEvaluationError(f"Nonphysical dm/dz={derivative} at z={z}, r={r}")
    return float(derivative)


def stieltjes_diagnostics(z, r, q, f):
    """Scalar branch checks, without a second cubic solve."""
    m, x = _project_state(z, r, q, f)
    z = float(np.real(z))
    a, b = q * (1 - f) * x - z, x - z
    return dict(z=z, r=float(r), m=m, x=x,
                residual=m + 1 / (q * z) - 1 / (q * a) - (q - 2) / (q * b),
                affine_residual=x - 1 + r + r * z * m,
                polynomial_residual=q * (m + 1 / (q * z)) * a * b - b - (q - 2) * a,
                dm_dz=project_stieltjes_dm_dz(z, r, q, f),
                abs_P_m=abs(_dP_dm(m, z, r, q, f, x=x)), A=a, B=b)



@dataclass(frozen=True)
class ResolventState:
    """Stable affine resolvent quantities, differentiated with respect to mu."""
    x: float
    dx_dmu: float
    x_minus_mu_dx_dmu: float


def project_resolvent_state(mu, r, q, f) -> ResolventState:
    """Avoid subtracting m-mu*m' or x-mu*x' near the ridgeless limit.

    Differentiating the original rational equation gives
    K=r/q*[q(1-f)/A²+(q-2)/B²], x_mu=x*K/(1+mu*K),
    and x-mu*x_mu=x/(1+mu*K). These are exact identities.
    """
    _m, x = _project_state(-mu, r, q, f)
    t = q * (1 - f)
    a, b = t * x + mu, x + mu
    k = r / q * (t / a**2 + (q - 2) / b**2)
    response = x / (1 + mu * k)
    return ResolventState(x, response * k, response)


@dataclass(frozen=True)
class StieltjesEvaluator:
    """Adapter for custom transforms; exact derivative optional for custom use."""
    m: MCallable
    dm_dz: DMCallable | None = None
    derivative_step: float = 1e-6
    resolvent_state: Callable[[float, float, int, float], ResolventState] | None = None

    def __call__(self, z: complex, r: float, q: int, f: float) -> complex:
        return self.m(z, r, q, f)

    def derivative(self, z: complex, r: float, q: int, f: float) -> complex:
        if self.dm_dz is not None:
            return self.dm_dz(z, r, q, f)
        h = self.derivative_step * max(1.0, abs(z))
        if np.imag(z) == 0 and np.real(z) < 0:
            h = min(h, abs(z) / 2)
        return (self(z + h, r, q, f) - self(z - h, r, q, f)) / (2 * h)


def project_stieltjes_evaluator() -> StieltjesEvaluator:
    return StieltjesEvaluator(project_stieltjes_m, project_stieltjes_dm_dz,
                             resolvent_state=project_resolvent_state)


def placeholder_stieltjes(z, r, q, f):  # pragma: no cover
    """Legacy placeholder; use project_stieltjes_evaluator() for production."""
    raise NotImplementedError("Use project_stieltjes_evaluator(), or provide a custom transform")
