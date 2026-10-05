from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import DMSConfig


class MomentFeasibilityError(ValueError):
    pass


@dataclass(frozen=True)
class FeasibilityReport:
    feasible: bool
    changed_sites: int
    equal_sites: int
    centered_gram_equal: np.ndarray
    centered_gram_changed: np.ndarray
    prescribed_norm2: float
    residual_norm2: float
    residual_dimension: int
    messages: tuple[str, ...]


@dataclass(frozen=True)
class GRModelResult:
    g: np.ndarray
    train_ref: np.ndarray
    test_ref: np.ndarray
    target_error: float
    measured_parameters: dict[str, float | int]

    @property
    def L(self) -> int:
        return int(self.train_ref.size)


def _round_half_up_positive(x: float) -> int:
    return int(np.floor(x + 0.5))


def gr_moments(
    n: int,
    sums: np.ndarray | list[float],
    gram: np.ndarray | list[list[float]],
    rng: np.random.Generator,
    *,
    tol: float = 1e-10,
) -> np.ndarray:
    """Construct n rows with prescribed column sums and Gram matrix.

    This is the NumPy translation of GRMoments.  If X has shape (n, d), the
    constraints are X.sum(axis=0) == sums and X.T @ X == gram.
    """
    sums = np.asarray(sums, dtype=float)
    gram = np.asarray(gram, dtype=float)
    if gram.shape != (sums.size, sums.size):
        raise ValueError("gram must be square with dimension len(sums)")

    if n == 0:
        if max(np.max(np.abs(sums), initial=0.0), np.max(np.abs(gram), initial=0.0)) > tol:
            raise MomentFeasibilityError("nonzero moments requested for an empty site class")
        return np.empty((0, sums.size), dtype=float)
    if n < 0:
        raise ValueError("n must be nonnegative")

    h = gram - np.outer(sums, sums) / n
    h = 0.5 * (h + h.T)
    evals, evecs = np.linalg.eigh(h)
    keep = evals > tol
    rank = int(np.count_nonzero(keep))

    if evals.min(initial=0.0) < -tol or rank > n - 1:
        raise MomentFeasibilityError(
            f"infeasible moments for n={n}; centered Gram eigenvalues={evals}"
        )

    out = np.tile(sums / n, (n, 1))
    if rank == 0:
        return out

    # Build an orthonormal basis Q in the centered site subspace 1^perp.
    # Re-draw in the extremely unlikely event of a rank-deficient random draw.
    for _ in range(8):
        z = rng.normal(size=(n, rank))
        z -= z.mean(axis=0, keepdims=True)
        qmat, rmat = np.linalg.qr(z, mode="reduced")
        if np.min(np.abs(np.diag(rmat))) > 1e-12:
            break
    else:  # pragma: no cover - practically unreachable
        raise RuntimeError("failed to construct centered orthonormal directions")

    vk = evecs[:, keep]                  # d x rank
    dk = np.sqrt(evals[keep])           # rank
    out += qmat @ (dk[:, None] * vk.T)  # n x d
    return out


def moment_feasibility(L: int, cfg: DMSConfig, *, tol: float = 1e-10) -> FeasibilityReport:
    """Check the exact finite-L feasibility conditions used by GRModel."""
    cfg.validate_basic()
    if L <= 0:
        raise ValueError("L must be positive")

    q = cfg.q
    k = _round_half_up_positive(L * cfg.eta)
    n0 = L - k
    gamma2 = (q - 1.0) / q
    kappa = -1.0 / (q - 1.0)
    qb = 1.0 - kappa**2

    f_eq = cfg.F - cfg.F_neq
    f_test_neq = cfg.F_test - f_eq
    m_eq = cfg.M_tot - cfg.M_neq

    msgs: list[str] = []

    if n0 == 0:
        h_eq = np.array([[0.0]])
        if abs(f_eq) > tol or abs(m_eq) > tol:
            msgs.append("equal-site moments are nonzero but there are no equal sites")
    else:
        h_eq = np.array([[f_eq - m_eq**2 / (gamma2 * n0)]], dtype=float)
        if h_eq[0, 0] < -tol:
            msgs.append("equal-site centered Gram is not positive semidefinite")

    if k == 0:
        h_ne = np.zeros((2, 2), dtype=float)
        if max(abs(cfg.F_neq), abs(cfg.C_neq), abs(f_test_neq), abs(cfg.M_neq), abs(cfg.M_neq_test)) > tol:
            msgs.append("changed-site moments are nonzero but there are no changed sites")
    else:
        sums = np.array([cfg.M_neq, cfg.M_neq_test], dtype=float) / np.sqrt(gamma2)
        gram = np.array(
            [[cfg.F_neq, cfg.C_neq], [cfg.C_neq, f_test_neq]], dtype=float
        )
        h_ne = gram - np.outer(sums, sums) / k
        h_ne = 0.5 * (h_ne + h_ne.T)
        evals = np.linalg.eigvalsh(h_ne)
        rank = int(np.count_nonzero(evals > tol))
        if evals.min(initial=0.0) < -tol:
            msgs.append("changed-site centered Gram is not positive semidefinite")
        if rank > k - 1:
            msgs.append("changed-site centered Gram rank exceeds k-1")

    c_energy = (f_test_neq - 2.0 * kappa * cfg.C_neq + kappa**2 * cfg.F_neq) / qb
    prescribed_norm2 = cfg.F + c_energy
    residual_norm2 = 1.0 - prescribed_norm2
    if residual_norm2 < -1e-9:
        msgs.append("prescribed projections require signal norm greater than 1")

    # Per equal site, after zero-sum and u removal: q-2 dimensions.
    # Per changed site, after zero-sum and u,w removal: q-3 dimensions.
    residual_dimension = n0 * (q - 2) + k * (q - 3)
    if residual_norm2 > tol and residual_dimension <= 0:
        msgs.append("positive residual norm requested but no residual direction exists")

    return FeasibilityReport(
        feasible=(len(msgs) == 0),
        changed_sites=k,
        equal_sites=n0,
        centered_gram_equal=h_eq,
        centered_gram_changed=h_ne,
        prescribed_norm2=float(prescribed_norm2),
        residual_norm2=float(residual_norm2),
        residual_dimension=int(residual_dimension),
        messages=tuple(msgs),
    )


def gr_model(
    L: int,
    cfg: DMSConfig,
    rng: np.random.Generator | None = None,
    *,
    tol: float = 1e-9,
) -> GRModelResult:
    """Construct a synthetic signal with the requested low-dimensional moments."""
    cfg.validate_basic()
    if L <= 0:
        raise ValueError("L must be positive")
    rng = np.random.default_rng() if rng is None else rng

    report = moment_feasibility(L, cfg)
    if not report.feasible:
        raise MomentFeasibilityError("; ".join(report.messages))

    q = cfg.q
    k = report.changed_sites
    gamma = np.sqrt((q - 1.0) / q)
    kappa = -1.0 / (q - 1.0)
    qb = 1.0 - kappa**2

    u = np.zeros(q)
    u[0] = 1.0
    u = (u - np.full(q, 1.0 / q)) / gamma

    v = np.zeros(q)
    v[1] = 1.0
    v = (v - np.full(q, 1.0 / q)) / gamma

    w = (v - kappa * u) / np.sqrt(qb)

    f_eq = cfg.F - cfg.F_neq
    f_test_neq = cfg.F_test - f_eq

    eq = gr_moments(
        L - k,
        [(cfg.M_tot - cfg.M_neq) / gamma],
        [[f_eq]],
        rng,
    )
    ne = gr_moments(
        k,
        np.array([cfg.M_neq, cfg.M_neq_test]) / gamma,
        [[cfg.F_neq, cfg.C_neq], [cfg.C_neq, f_test_neq]],
        rng,
    )

    a = np.concatenate([ne[:, 0] if k > 0 else np.empty(0), eq[:, 0]])
    c = np.concatenate(
        [
            (ne[:, 1] - kappa * ne[:, 0]) / np.sqrt(qb)
            if k > 0
            else np.empty(0),
            np.zeros(L - k),
        ]
    )

    g = a[:, None] * u[None, :] + c[:, None] * w[None, :]
    rem = 1.0 - float(np.sum(g * g))
    if rem < -tol:
        raise MomentFeasibilityError(
            f"prescribed projections use norm^2={1.0-rem:.12g} > 1"
        )

    mask = np.concatenate([np.ones(k), np.zeros(L - k)])

    if rem > 1e-10:
        z = rng.normal(size=(L, q))
        z -= z.mean(axis=1, keepdims=True)
        z -= (z @ u)[:, None] * u[None, :]
        z -= (mask * (z @ w))[:, None] * w[None, :]
        znorm = float(np.linalg.norm(z))
        if znorm < 1e-9:
            raise MomentFeasibilityError("no residual direction available")
        g += np.sqrt(rem) * z / znorm

    aa = g @ u
    bb = (1.0 - mask) * aa + mask * (g @ v)
    gnorm = float(np.linalg.norm(g))

    got = np.array(
        [
            gnorm,
            aa @ aa,
            bb @ bb,
            np.sum(mask * aa**2),
            np.sum(mask * aa * bb),
            gamma * np.sum(aa),
            gamma * np.sum(mask * aa),
            gamma * np.sum(mask * bb),
        ],
        dtype=float,
    )
    target = np.array(
        [
            1.0,
            cfg.F,
            cfg.F_test,
            cfg.F_neq,
            cfg.C_neq,
            cfg.M_tot,
            cfg.M_neq,
            cfg.M_neq_test,
        ],
        dtype=float,
    )
    target_error = float(np.max(np.abs(got - target)))
    if target_error > 1e-7:
        raise RuntimeError(f"constructed model missed target moments by {target_error:g}")

    measured: dict[str, float | int] = {
        **cfg.as_dict(),
        "signal_norm": cfg.signal_norm * gnorm,
        "F": got[1] / gnorm**2,
        "F_test": got[2] / gnorm**2,
        "F_neq": got[3] / gnorm**2,
        "C_neq": got[4] / gnorm**2,
        "M_tot": got[5] / gnorm,
        "M_neq": got[6] / gnorm,
        "M_neq_test": got[7] / gnorm,
        "eta": k / L,
        "L": L,
        "changed_sites": k,
    }

    return GRModelResult(
        g=cfg.signal_norm * g.reshape(-1),
        train_ref=np.zeros(L, dtype=int),
        test_ref=np.concatenate([np.ones(k, dtype=int), np.zeros(L - k, dtype=int)]),
        target_error=target_error,
        measured_parameters=measured,
    )


def gr_draw(
    ref: np.ndarray | list[int],
    q: int,
    f: float,
    n: int,
    rng: np.random.Generator | None = None,
    *,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Draw n one-hot genotypes around a sitewise reference sequence.

    State labels are zero-based in Python.  With probability 1-f a site stays
    at its reference state; otherwise one of the q-1 alternatives is chosen
    uniformly.
    """
    rng = np.random.default_rng() if rng is None else rng
    ref = np.asarray(ref, dtype=int)
    L = ref.size
    if n <= 0:
        raise ValueError("n must be positive")
    if q < 2 or np.any((ref < 0) | (ref >= q)):
        raise ValueError("invalid q or reference states")
    if not (0.0 <= f <= 1.0):
        raise ValueError("f must lie in [0,1]")

    mutate = rng.random((n, L)) < f
    alt = rng.integers(0, q - 1, size=(n, L))
    alt_state = alt + (alt >= ref[None, :])
    states = np.where(mutate, alt_state, ref[None, :])

    x = np.zeros((n, L * q), dtype=dtype)
    cols = q * np.arange(L)[None, :] + states
    rows = np.arange(n)[:, None]
    x[rows, cols] = 1.0
    return x


def _test_probabilities(ref: np.ndarray, q: int, f: float) -> np.ndarray:
    probs = np.full((ref.size, q), f / (q - 1.0), dtype=float)
    probs[np.arange(ref.size), ref] = 1.0 - f
    return probs


def gr_trial(
    n: int,
    ridge_lambdas: np.ndarray | list[float],
    model: GRModelResult,
    cfg: DMSConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One training draw and exact test-distribution signal MSE for each lambda."""
    rng = np.random.default_rng() if rng is None else rng
    lambdas = np.asarray(ridge_lambdas, dtype=float)
    if np.any(lambdas <= 0):
        raise ValueError("ridge lambdas must be positive")

    q = cfg.q
    g = model.g
    x = gr_draw(model.train_ref, q, cfg.f_train, n, rng)
    y = x @ g + cfg.sigma * rng.normal(size=n)

    xm = x.mean(axis=0)
    ym = float(y.mean())
    xc = x - xm
    yc = y - ym

    dual = n <= g.size
    if dual:
        gram = xc @ xc.T
        rhs = yc
    else:
        gram = xc.T @ xc
        rhs = xc.T @ yc

    gram = 0.5 * (gram + gram.T)
    d, v = np.linalg.eigh(gram)
    d = np.clip(d, 0.0, None)
    scores = v.T @ rhs

    coeff = scores[:, None] / (d[:, None] + n * lambdas[None, :])
    sol = v @ coeff
    hats = xc.T @ sol if dual else sol  # p x n_lambda
    hats = hats.T                      # n_lambda x p

    probs = _test_probabilities(model.test_ref, q, cfg.f_test)
    L = model.L
    err = (hats - g[None, :]).reshape(lambdas.size, L, q)
    means = np.sum(probs[None, :, :] * err, axis=2)
    off = ym - hats @ xm

    risks = (
        np.sum(probs[None, :, :] * err**2, axis=(1, 2))
        - np.sum(means**2, axis=1)
        + (off + np.sum(means, axis=1)) ** 2
    )
    return risks.astype(float)


def exact_population_infinity_risk(model: GRModelResult, cfg: DMSConfig) -> float:
    """Direct finite-q population check of the r -> infinity formula."""
    q = cfg.q
    L = model.L
    g = model.g.reshape(L, q)
    p_train = _test_probabilities(model.train_ref, q, cfg.f_train)
    p_test = _test_probabilities(model.test_ref, q, cfg.f_test)

    mean_train_sites = np.sum(p_train * g, axis=1)
    mean_test_sites = np.sum(p_test * g, axis=1)
    second_test_sites = np.sum(p_test * g**2, axis=1)

    mean_train = float(np.sum(mean_train_sites))
    mean_test = float(np.sum(mean_test_sites))
    var_test = float(np.sum(second_test_sites - mean_test_sites**2))
    return var_test + (mean_train - mean_test) ** 2
