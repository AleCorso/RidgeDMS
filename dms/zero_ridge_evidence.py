"""Total-risk, multi-alignment appendix data; figures remain notebook-local.

Keep this adapter separate so zero-figure edits do not invalidate the existing
infinite-ridge numerical caches in appendix_evidence.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np

from .appendix_evidence import cached_stage, cached_zero_evidence as cached_single_zero_evidence
from .config import DMSConfig
from .ridge_diagnostics import effective_risk, zero_ridge_kappa, optimize_effective_ridge_over_r


_ROOT = Path(__file__).resolve().parent
_SOURCES = [_ROOT / (name + '.py') for name in (
    'config', 'stieltjes', 'theory', 'ridge_diagnostics',
    'infinity_diagnostics', 'appendix_evidence', 'zero_ridge_evidence',
)]


def _zero_comparison(cache, inputs, *, force=False):
    selected = np.asarray(inputs['F_values'], dtype=float)
    if (selected.ndim != 1 or not selected.size or np.any(~np.isfinite(selected))
            or np.any((selected < 0) | (selected > 1))
            or len(np.unique(selected)) != len(selected)):
        raise ValueError('Select distinct finite F values in [0, 1].')
    base = DMSConfig(**inputs['config'])
    common = {key: value for key, value in inputs.items() if key != 'F_values'}
    curves, identities = {}, {}
    total_slopes = None
    for F in selected:
        F = float(F)
        case = replace(base, F=F, F_test=F)
        case_inputs = dict(common, config=case.as_dict())
        # Reuse the existing F=.4 evidence when its full inputs still match.
        data, identity = cached_single_zero_evidence(cache, case_inputs, compute=True, force=force)
        identities[F] = identity
        # Optimize at the exact selected r, independently of the sampled r grid.
        cut_optimum = optimize_effective_ridge_over_r([inputs['r_check']], case, **inputs['search'])
        zero = effective_risk(zero_ridge_kappa(inputs['r_check'], case), inputs['r_check'], case)
        minimum_lambda = float(cut_optimum.ridge_lambdas[0])
        minimum_delta = float(cut_optimum.risks[0] - zero.risk)
        lambdas = np.asarray(data['lambdas']).copy()
        delta = np.asarray(data['delta'])[:, 2].copy()  # total risk only
        # Include the resolved minimum so the curve passes through its marker.
        same = lambdas == minimum_lambda
        if np.any(same):
            delta[same] = minimum_delta
        else:
            order = np.argsort(np.r_[lambdas, minimum_lambda])
            lambdas = np.r_[lambdas, minimum_lambda][order]
            delta = np.r_[delta, minimum_delta][order]
        curves[F] = dict(config=case.as_dict(), optimum=data['optimum'],
                         lambdas=lambdas, delta=delta, minimum_lambda=minimum_lambda,
                         minimum_delta=minimum_delta, minimum_status=str(cut_optimum.status[0]))
        if total_slopes is None:
            total_slopes = np.asarray(data['slopes'])[:, 2].copy()
    return dict(config=inputs['config'], r_check=float(inputs['r_check']), curves=curves,
                alignments=np.asarray(inputs['alignments']).copy(), slopes=total_slopes,
                source_ids=identities)


def cached_zero_evidence(cache, inputs, *, compute=False, force=False):
    """Compute explicitly, or restore exact matching comparison data only."""
    return cached_stage(cache, 'appendix_zero_comparison', inputs,
                        lambda: _zero_comparison(cache, inputs, force=force),
                        sources=_SOURCES, compute=compute, force=force)
