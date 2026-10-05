from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping


@dataclass(frozen=True)
class DMSConfig:
    """Model/theory parameters used by the generalized out-of-sample code.

    Names are intentionally close to the Mathematica notebook, while using
    snake_case for train/test and mismatch moment suffixes.
    """

    q: int
    sigma: float
    f_train: float
    f_test: float
    eta: float
    signal_norm: float
    F: float
    F_test: float
    F_neq: float
    C_neq: float
    M_tot: float
    M_neq: float
    M_neq_test: float

    def validate_basic(self) -> None:
        if self.q < 3:
            raise ValueError("q must be at least 3")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.signal_norm <= 0:
            raise ValueError("signal_norm must be positive")
        for name, value in (
            ("f_train", self.f_train),
            ("f_test", self.f_test),
            ("eta", self.eta),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1]")

    def validate_theory(self) -> None:
        self.validate_basic()
        if self.f_train <= 0:
            raise ValueError("f_train must be > 0 for the current ridge scaling")
        if self.f_test <= 0:
            raise ValueError(
                "f_test must be > 0 for the current a3/b3 formulas; "
                "handle the f_test -> 0 limit separately if needed"
            )

    @classmethod
    def from_mathematica_dict(cls, p: Mapping[str, Any]) -> "DMSConfig":
        """Build from the Association keys used in clean.nb."""
        return cls(
            q=int(p["q"]),
            sigma=float(p["Sigma"]),
            f_train=float(p["fTrain"]),
            f_test=float(p["fTest"]),
            eta=float(p["Eta"]),
            signal_norm=float(p["G"]),
            F=float(p["F"]),
            F_test=float(p["FTest"]),
            F_neq=float(p["Fneq"]),
            C_neq=float(p["Cneq"]),
            M_tot=float(p["Mtot"]),
            M_neq=float(p["Mneq"]),
            M_neq_test=float(p["MneqTest"]),
        )

    def to_mathematica_dict(self) -> dict[str, float | int]:
        return {
            "q": self.q,
            "Sigma": self.sigma,
            "fTrain": self.f_train,
            "fTest": self.f_test,
            "Eta": self.eta,
            "G": self.signal_norm,
            "F": self.F,
            "FTest": self.F_test,
            "Fneq": self.F_neq,
            "Cneq": self.C_neq,
            "Mtot": self.M_tot,
            "Mneq": self.M_neq,
            "MneqTest": self.M_neq_test,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
