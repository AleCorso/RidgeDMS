from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DMSConfig
from .stieltjes import StieltjesEvaluator


class TheoryEvaluationError(RuntimeError):
    pass


def qbar(q: int) -> float:
    return 1.0 - 1.0 / (q - 1.0) ** 2


@dataclass(frozen=True)
class DMSTTheory:
    """Generalized out-of-sample deterministic-equivalent theory.

    The public ``risk`` method takes the physical ridge parameter lambda used
    in the numerical fit, i.e. X^T X + n * lambda I. Internally it converts to

        mu = lambda * (q - 1) / f_train,

    which is the scaled regularization variable used by the analytic formulas.
    """

    stieltjes: StieltjesEvaluator
    imag_tol: float = 1e-8

    # ---------- resolvent helpers ----------
    def h(self, t: float, r: float, q: int, z: complex, f: float) -> complex:
        if self.stieltjes.resolvent_state is not None:
            state = self.stieltjes.resolvent_state(-z, r, q, f)
            return 1.0 / (t * state.x - z)
        m = self.stieltjes(z, r, q, f)
        return 1.0 / (t * (1.0 - r - r * z * m) - z)

    def dh_dmu(self, t: float, mu: float, r: float, q: int, f: float) -> complex:
        """d/dmu h(t, r, q, -mu, f)."""
        if self.stieltjes.resolvent_state is not None:
            state = self.stieltjes.resolvent_state(mu, r, q, f)
            return -(1 + t * state.dx_dmu) / (t * state.x + mu)**2
        z = -mu
        m = self.stieltjes(z, r, q, f)
        mp = self.stieltjes.derivative(z, r, q, f)
        hv = self.h(t, r, q, z, f)
        # dh/dz = h^2 [1 + t r (m + z m')], and z=-mu.
        return -(hv**2) * (1.0 + t * r * (m + z * mp))

    def h_response(self, t: float, mu: float, r: float, cfg: DMSConfig) -> complex:
        """h + mu*dh/dmu, evaluated without cancellation for the project solver."""
        if self.stieltjes.resolvent_state is not None:
            state = self.stieltjes.resolvent_state(mu, r, cfg.q, cfg.f_train)
            return t * state.x_minus_mu_dx_dmu / (t * state.x + mu)**2
        return (self.h(t, r, cfg.q, -mu, cfg.f_train)
                + mu * self.dh_dmu(t, mu, r, cfg.q, cfg.f_train))

    # ---------- variance trace pieces ----------
    @staticmethod
    def _trace_weights(cfg: DMSConfig) -> tuple[float, float]:
        q, qb = cfg.q, qbar(cfg.q)
        cm = q - 2.0 - cfg.eta * qb + q * (1.0 - cfg.f_test) * cfg.eta * qb
        cw = q * (1.0 - cfg.f_test) * (1.0 - cfg.eta * qb) + cfg.eta * qb
        return cm / q, cw / q

    @staticmethod
    def _tracedelta_weights(cfg: DMSConfig) -> tuple[float, float]:
        q, qb = cfg.q, qbar(cfg.q)
        aw = (q / (q - 1.0) * (1.0 - cfg.eta) * (cfg.f_train - cfg.f_test)**2
              + q / (q - 1.0) * cfg.eta
              * (1.0 - cfg.f_train - cfg.f_test / (q - 1.0))**2)
        am = (cfg.eta * (q - 1.0) / q * qb
              * (1.0 - q * cfg.f_test / (q - 1.0))**2)
        return am / q, aw / q

    def trace(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        cm, cw = self._trace_weights(cfg)
        return (cm * self.h(1.0, r, cfg.q, -mu, cfg.f_train)
                + cw * self.h(cfg.q * (1 - cfg.f_train), r, cfg.q, -mu, cfg.f_train))

    def dtrace(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        cm, cw = self._trace_weights(cfg)
        return (cm * self.dh_dmu(1.0, mu, r, cfg.q, cfg.f_train)
                + cw * self.dh_dmu(cfg.q * (1 - cfg.f_train), mu, r, cfg.q, cfg.f_train))

    def tracedelta(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        cm, cw = self._tracedelta_weights(cfg)
        return (cm * self.h(1.0, r, cfg.q, -mu, cfg.f_train)
                + cw * self.h(cfg.q * (1 - cfg.f_train), r, cfg.q, -mu, cfg.f_train))

    def dtracedelta(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        cm, cw = self._tracedelta_weights(cfg)
        return (cm * self.dh_dmu(1.0, mu, r, cfg.q, cfg.f_train)
                + cw * self.dh_dmu(cfg.q * (1 - cfg.f_train), mu, r, cfg.q, cfg.f_train))

    def trace_response(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        cm, cw = self._trace_weights(cfg)
        return cm * self.h_response(1.0, mu, r, cfg) + cw * self.h_response(cfg.q * (1 - cfg.f_train), mu, r, cfg)

    def tracedelta_response(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        cm, cw = self._tracedelta_weights(cfg)
        return cm * self.h_response(1.0, mu, r, cfg) + cw * self.h_response(cfg.q * (1 - cfg.f_train), mu, r, cfg)

    def r_var(self, mu: float, r: float, cfg: DMSConfig) -> float:
        term_cov = (
            cfg.f_test
            / cfg.f_train
            * cfg.sigma**2
            * r
            * self.trace_response(mu, r, cfg)
        )
        term_delta = (
            (cfg.q - 1.0)
            / cfg.f_train
            * cfg.sigma**2
            * r
            * self.tracedelta_response(mu, r, cfg)
        )
        return self._as_real(term_cov + term_delta)

    # ---------- bias coefficients ----------
    def x(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        if self.stieltjes.resolvent_state is not None:
            return self.stieltjes.resolvent_state(mu, r, cfg.q, cfg.f_train).x
        m = self.stieltjes(-mu, r, cfg.q, cfg.f_train)
        return 1.0 - r * (1.0 - mu * m)

    def a1(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        q = cfg.q
        xv = self.x(mu, r, cfg)
        hm = self.h(1.0, r, q, -mu, cfg.f_train)
        hw = self.h(q * (1.0 - cfg.f_train), r, q, -mu, cfg.f_train)
        return r * xv**2 * (
            (q - 2.0) / q * hm**2
            + q * (1.0 - cfg.f_train) ** 2 * hw**2
        )

    def a2(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        q = cfg.q
        qb = qbar(q)
        xv = self.x(mu, r, cfg)
        hm = self.h(1.0, r, q, -mu, cfg.f_train)
        hw = self.h(q * (1.0 - cfg.f_train), r, q, -mu, cfg.f_train)

        cm = (
            q
            - 2.0
            - cfg.eta * qb
            + q * (1.0 - cfg.f_test) * cfg.eta * qb
        ) / q
        cw = (
            q * (1.0 - cfg.f_test) * (1.0 - cfg.eta * qb)
            + cfg.eta * qb
        ) * (1.0 - cfg.f_train)

        return r * xv**2 * (cm * hm**2 + cw * hw**2)

    def a3(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        q = cfg.q
        qb = qbar(q)
        xv = self.x(mu, r, cfg)
        hm = self.h(1.0, r, q, -mu, cfg.f_train)
        hw = self.h(q * (1.0 - cfg.f_train), r, q, -mu, cfg.f_train)

        tm = (
            cfg.eta
            * (q - 1.0)
            / q
            * qb
            * (1.0 - q * cfg.f_test / (q - 1.0)) ** 2
            * hm**2
        )
        tw = (
            q
            * (1.0 - cfg.f_train)
            * q
            / (q - 1.0)
            * (
                (1.0 - cfg.eta) * (cfg.f_train - cfg.f_test) ** 2
                + cfg.eta
                * (1.0 - cfg.f_train - cfg.f_test / (q - 1.0)) ** 2
            )
            * hw**2
        )

        return r * xv**2 * (q - 1.0) / cfg.f_test * (tm + tw) / q

    def a23(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        return self.a2(mu, r, cfg) + self.a3(mu, r, cfg)

    # ---------- signal-moment contractions ----------
    @staticmethod
    def g000(cfg: DMSConfig) -> float:
        return cfg.signal_norm**2 * (cfg.F - qbar(cfg.q) * cfg.F_neq)

    def g001(self, cfg: DMSConfig) -> float:
        return cfg.signal_norm**2 * cfg.F - self.g000(cfg)

    @staticmethod
    def g00(cfg: DMSConfig) -> float:
        return cfg.signal_norm**2 * (
            cfg.F - cfg.F_neq - cfg.C_neq / (cfg.q - 1.0)
        )

    def g010(self, cfg: DMSConfig) -> float:
        return self.g00(cfg) - self.g000(cfg)

    def g011(self, cfg: DMSConfig) -> float:
        return -self.g010(cfg)

    def g110(self, cfg: DMSConfig) -> float:
        return (
            cfg.signal_norm**2 * cfg.F_test
            + self.g000(cfg)
            - 2.0 * self.g00(cfg)
        )

    def g111(self, cfg: DMSConfig) -> float:
        return cfg.signal_norm**2 * (1.0 - cfg.F) - self.g110(cfg)

    @staticmethod
    def g11_delta(cfg: DMSConfig) -> float:
        q = cfg.q
        return (
            -(1.0 - q * cfg.f_test / (q - 1.0))
            * (cfg.M_neq_test + cfg.M_neq / (q - 1.0))
            * cfg.signal_norm
        )

    @staticmethod
    def g00_delta(cfg: DMSConfig) -> float:
        q = cfg.q
        m_eq = cfg.M_tot - cfg.M_neq
        return (
            cfg.signal_norm
            * q
            / (q - 1.0)
            * (
                -(cfg.f_train - cfg.f_test) * m_eq
                + cfg.M_neq
                * (
                    (cfg.f_train - cfg.f_test) / (q - 1.0)
                    + (1.0 - q * cfg.f_train / (q - 1.0))
                )
            )
        )

    def b1(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        q = cfg.q
        hw = self.h(q * (1.0 - cfg.f_train), r, q, -mu, cfg.f_train)
        hm = self.h(1.0, r, q, -mu, cfg.f_train)
        return cfg.signal_norm**2 * (
            cfg.F * q * (1.0 - cfg.f_train) * hw**2
            + (1.0 - cfg.F) * hm**2
        )

    def b2(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        q = cfg.q
        hw = self.h(q * (1.0 - cfg.f_train), r, q, -mu, cfg.f_train)
        hm = self.h(1.0, r, q, -mu, cfg.f_train)

        return (
            (q * (1.0 - cfg.f_test) * self.g000(cfg) + self.g001(cfg))
            * hw**2
            + (self.g111(cfg) + q * (1.0 - cfg.f_test) * self.g110(cfg))
            * hm**2
            + 2.0
            * (self.g011(cfg) + q * (1.0 - cfg.f_test) * self.g010(cfg))
            * hw
            * hm
        )

    def b3(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        q = cfg.q
        hw = self.h(q * (1.0 - cfg.f_train), r, q, -mu, cfg.f_train)
        hm = self.h(1.0, r, q, -mu, cfg.f_train)
        dw = self.g00_delta(cfg)
        dm = self.g11_delta(cfg)

        # Keep the square form explicitly; this prevents losing the factor 2.
        return (q - 1.0) / cfg.f_test * (dw * hw + dm * hm) ** 2

    def b23(self, mu: float, r: float, cfg: DMSConfig) -> complex:
        return self.b2(mu, r, cfg) + self.b3(mu, r, cfg)

    def r_bias(self, mu: float, r: float, cfg: DMSConfig) -> float:
        b23 = self.b23(mu, r, cfg)
        a1 = self.a1(mu, r, cfg)
        numerator = b23 + self.b1(mu, r, cfg) * self.a23(mu, r, cfg) - b23 * a1
        value = mu**2 * cfg.f_test / (cfg.q - 1.0) * numerator / (1.0 - a1)
        return self._as_real(value)

    # ---------- public theory interface ----------
    def risk(self, ridge_lambda: float, r: float, cfg: DMSConfig) -> float:
        """Signal-prediction MSE for the physical ridge value ``ridge_lambda``."""
        cfg.validate_theory()
        if not np.isfinite(ridge_lambda) or not np.isfinite(r) or ridge_lambda <= 0 or r <= 0:
            raise ValueError("ridge_lambda and r must be finite and positive")
        mu = ridge_lambda * (cfg.q - 1.0) / cfg.f_train
        return self.r_bias(mu, r, cfg) + self.r_var(mu, r, cfg)

    def curves(
        self, ridge_lambdas: np.ndarray, r_grid: np.ndarray, cfg: DMSConfig,
        *, normalize_by_sigma2: bool = True,
    ) -> np.ndarray:
        ridge_lambdas = np.asarray(ridge_lambdas, dtype=float)
        r_grid = np.asarray(r_grid, dtype=float)
        out = np.empty((ridge_lambdas.size, r_grid.size), dtype=float)
        scale = cfg.sigma**2 if normalize_by_sigma2 else 1.0
        for i, lam in enumerate(ridge_lambdas):
            for j, r in enumerate(r_grid):
                out[i, j] = self.risk(float(lam), float(r), cfg) / scale
        return out

    # ---------- r -> infinity diagnostics ----------
    @staticmethod
    def population_mean_shift(cfg: DMSConfig) -> float:
        """train population mean minus test population mean, divided by G."""
        q = cfg.q
        return (
            q / (q - 1.0) * (cfg.f_test - cfg.f_train) * cfg.M_tot
            + (1.0 - q * cfg.f_test / (q - 1.0))
            * (cfg.M_neq - cfg.M_neq_test)
        )

    @staticmethod
    def test_signal_variance(cfg: DMSConfig) -> float:
        q = cfg.q
        return (
            cfg.signal_norm**2
            * cfg.f_test
            / (q - 1.0)
            * (1.0 + (q * (1.0 - cfg.f_test) - 1.0) * cfg.F_test)
        )

    @classmethod
    def infinity_risk(cls, cfg: DMSConfig) -> float:
        shift = cls.population_mean_shift(cfg)
        return cls.test_signal_variance(cfg) + cfg.signal_norm**2 * shift**2

    def _as_real(self, z: complex | float) -> float:
        z = complex(z)
        if not np.isfinite(z):
            raise TheoryEvaluationError(f"Theory produced a non-finite value: {z}")
        scale = max(1.0, abs(z.real))
        if abs(z.imag) > self.imag_tol * scale:
            raise TheoryEvaluationError(
                f"Theory produced a non-negligible imaginary part: {z}"
            )
        return float(z.real)
