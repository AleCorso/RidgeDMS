"""Cached numerical inputs for the notebook-editable appendix diagnostics.

These reuse the existing boundary and infinity diagnostics. Figure construction
stays in the notebooks; a cache-only request never evaluates the theory.
"""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import pickle

import numpy as np

from .config import DMSConfig
from .infinity_diagnostics import infinity_gap_certificate, stable_risk_gap
from .ridge_diagnostics import (
    effective_risk, kappa_from_lambda, zero_ridge_kappa,
    optimize_effective_ridge_over_r,
)
from .theory import DMSTTheory


_ROOT = Path(__file__).resolve().parent
_SOURCES = [_ROOT / (name + '.py') for name in (
    'config', 'stieltjes', 'theory', 'ridge_diagnostics',
    'infinity_diagnostics', 'appendix_evidence',
)]
_UNCACHED_RESULTS = {}


def cached_stage(cache, kind, inputs, producer, *, sources=(), compute=False, force=False):
    """Retain current-kernel results when disk caching is explicitly disabled."""
    token = sha256(pickle.dumps((inputs, {
        str(path): sha256(Path(path).read_bytes()).hexdigest() for path in sources
    }))).hexdigest()
    slot = (str(cache.directory), kind)
    if not cache.enabled and not compute:
        previous = _UNCACHED_RESULTS.get(slot)
        if force or previous is None or previous[0] != token:
            raise FileNotFoundError(f'{kind}: run the matching compute cell first (disk cache disabled)')
        return previous[1]
    result = cache.get(kind, inputs, producer, sources=sources, compute=compute, force=force)
    if not cache.enabled:
        _UNCACHED_RESULTS[slot] = (token, result)
    return result


def _zero_evidence(config, r_check, near_zero_lambdas, r_values, alignments, search):
    cfg = DMSConfig(**config)
    cfg.validate_theory()
    if (cfg.eta != 0 or cfg.f_train != cfg.f_test or cfg.F != cfg.F_test
            or any(getattr(cfg, key) != 0 for key in ('F_neq', 'C_neq', 'M_neq', 'M_neq_test'))):
        raise ValueError('The zero-boundary alignment path requires the matched baseline.')
    endpoint = effective_risk(zero_ridge_kappa(r_check, cfg), r_check, cfg)
    points = [effective_risk(kappa_from_lambda(float(lam), r_check, cfg), r_check, cfg)
              for lam in near_zero_lambdas]
    fields = ('bias', 'variance', 'risk')
    delta = np.array([[getattr(p, name) - getattr(endpoint, name) for name in fields]
                      for p in points])
    slopes = []
    for F in alignments:
        case = replace(cfg, F=float(F), F_test=float(F))
        p = effective_risk(zero_ridge_kappa(r_check, case), r_check, case)
        conversion = (case.q - 1) / case.f_train / p.denominator
        slopes.append([conversion * p.d_bias_d_kappa,
                       conversion * p.d_variance_d_kappa, p.d_risk_d_lambda])
    optimum = optimize_effective_ridge_over_r(r_values, cfg, **search)
    return dict(config=config, r_check=float(r_check),
                lambdas=np.asarray(near_zero_lambdas).copy(), delta=delta,
                alignments=np.asarray(alignments).copy(), slopes=np.asarray(slopes),
                optimum=optimum, R_infinity=DMSTTheory.infinity_risk(cfg))


def cached_zero_evidence(cache, inputs, *, compute=False, force=False):
    return cached_stage(cache, 'appendix_zero_evidence', inputs,
                     lambda: _zero_evidence(**inputs), sources=_SOURCES,
                     compute=compute, force=force)


def _infinity_evidence(config, r_slices, lambda_grid, r_transition, caps,
                       n_grid, bracket_steps, witness_rtol):
    cfg = DMSConfig(**config)
    cfg.validate_theory()
    if not np.isfinite(witness_rtol) or witness_rtol <= 0:
        raise ValueError('witness_rtol must be finite and positive.')
    if not len(caps) or any(not np.isfinite(cap) or cap <= 0 for cap in caps):
        raise ValueError('Use at least one positive finite search cap.')
    if len(set(caps)) != len(caps):
        raise ValueError('Search caps must be distinct.')
    rinf = DMSTTheory.infinity_risk(cfg)
    certificates = {float(r): infinity_gap_certificate(cfg, float(r), bracket_steps=bracket_steps)
                    for r in np.unique(np.r_[r_slices, r_transition])}
    slices = {}
    for r in r_slices:
        r = float(r)
        k = np.array([kappa_from_lambda(float(lam), r, cfg) for lam in lambda_grid])
        cert = certificates[r]
        slices[r] = dict(gap=stable_risk_gap(cert, k),
                         tail_coefficient=float(cert['tail_coefficient']), certificate=cert)
    cap_results = {float(cap): optimize_effective_ridge_over_r(
        r_transition, cfg, lambda_max=float(cap), n_grid=n_grid) for cap in caps}
    transition = []
    for i, r in enumerate(r_transition):
        cert = certificates[float(r)]
        candidate = min(cap_results.values(), key=lambda result: result.risks[i])
        gap = float(stable_risk_gap(cert, candidate.kappas[i]))
        certified = cert['status'] == 'certified_nonnegative'
        witness = gap < -witness_rtol * rinf
        if certified and witness:
            raise ArithmeticError('Certificate/search contradiction; audit the inputs.')
        status = 'infinity certified' if certified else 'finite witness' if witness else 'unresolved'
        transition.append(dict(r=float(r), status=status, candidate_gap=gap,
                               candidate_lambda=float(candidate.ridge_lambdas[i]), certificate=cert))
    return dict(config=config, R_infinity=rinf, lambdas=np.asarray(lambda_grid).copy(),
                slices=slices, caps=cap_results, transition=transition)


def cached_infinity_evidence(cache, inputs, *, compute=False, force=False):
    return cached_stage(cache, 'appendix_infinity_evidence', inputs,
                     lambda: _infinity_evidence(**inputs), sources=_SOURCES,
                     compute=compute, force=force)
