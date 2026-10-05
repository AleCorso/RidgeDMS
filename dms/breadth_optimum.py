"""Adaptive training-breadth minimization for the four-curve exploration.

A few probes locate candidate basins; golden-section searches refine each one.
This reduces ridge optimizations but does not certify a global breadth minimum.
Figure construction remains in the notebook, and cache-only reads run no theory.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from .optimization import _golden_minimize
from .parameter_planes import SOURCES, cached_parameter_map
from .result_cache import ResultCache
from .infinity_diagnostics import infinity_gap_certificate, _add, _scale, _shift


OPTIMUM_SOURCES = (*SOURCES, Path(__file__).with_name('optimization.py'),
                   Path(__file__).with_name('infinity_diagnostics.py'),
                   Path(__file__).resolve())


def _tail_exclusion_certificate(cfg, r, threshold, ridge_cap, steps):
    """Sufficient exact-decimal proof that R(lambda)>=threshold above the cap.

    Reuse the existing rational risk polynomial. Bracket the kappa corresponding
    to the finite ridge cap with Fraction arithmetic, then test nonnegativity of
    all coefficients of R-threshold after shifting to the lower bracket endpoint.
    Failure of this sufficient test is inconclusive, never evidence of dominance.
    """
    certificate = infinity_gap_certificate(cfg, r, bracket_steps=steps)
    Q = lambda value: Fraction(str(value))
    q, t, rr = Q(cfg.q), Q(cfg.q)*(1-Q(cfg.f_train)), Q(r)
    target_mu = Q(ridge_cap)*(q-1)/Q(cfg.f_train)

    def mu(k):
        return k*(1-rr/q*((q-2)/(1+k)+t/(t+k)))

    lo, physical_hi = certificate['kappa0_bracket']
    hi = max(Fraction(1), physical_hi)
    while mu(hi) < target_mu:
        hi *= 2
    for _ in range(steps):
        mid = (lo+hi)/2
        if mu(mid) <= target_mu:
            lo = mid
        else:
            hi = mid
    numerator = _add(certificate['numerator'], _scale(
        certificate['denominator'], certificate['R_infinity_exact']-Q(threshold)))
    shifted = _shift(numerator, lo)
    return dict(excluded=all(coefficient >= 0 for coefficient in shifted),
                threshold=float(threshold), ridge_cap=float(ridge_cap),
                kappa_bracket=(lo, hi), shifted_coefficients=shifted)


def _validated_inputs(settings):
    inputs = deepcopy(settings)
    if 'f_grid' in inputs:
        raise ValueError('Replace f_grid with f_search; reuse_f_grid is cache-only')
    grid = np.asarray(inputs['ftilde_grid'], dtype=float)
    if (grid.ndim != 1 or not grid.size or np.any(~np.isfinite(grid))
            or np.any(np.diff(grid) <= 0) or np.any((grid <= 0) | (grid >= 1))):
        raise ValueError('ftilde_grid must be increasing and strictly inside (0, 1)')
    inputs['ftilde_grid'] = grid
    values = np.asarray(inputs['Fneq_values'], dtype=float)
    if (values.ndim != 1 or not values.size or np.any(~np.isfinite(values))
            or np.unique(values).size != values.size):
        raise ValueError('Fneq_values must be distinct finite values')
    inputs['Fneq_values'] = values.tolist()
    if not np.isfinite(inputs['r']) or inputs['r'] <= 0:
        raise ValueError('r must be positive and finite')
    options = inputs['f_search']
    if set(options) != {'bounds', 'n_probe', 'tolerance', 'max_iter', 'extra_probes'}:
        raise ValueError('f_search requires bounds, n_probe, tolerance, max_iter, extra_probes')
    bounds = np.asarray(options['bounds'], dtype=float)
    if bounds.shape != (2,) or not 0 < bounds[0] < bounds[1] < 1:
        raise ValueError('f_search bounds must satisfy 0 < low < high < 1')
    for key, minimum in [('n_probe', 3), ('max_iter', 1)]:
        value = options[key]
        if not np.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f'f_search {key} must be an integer >= {minimum}')
        options[key] = int(value)
    if not np.isfinite(options['tolerance']) or not 0 < options['tolerance'] < np.diff(bounds)[0]:
        raise ValueError('f_search tolerance must be positive and smaller than the interval')
    extra = np.asarray(options['extra_probes'], dtype=float)
    if (extra.ndim != 1 or np.any(~np.isfinite(extra))
            or np.any((extra < bounds[0]) | (extra > bounds[1]))):
        raise ValueError('extra_probes must lie inside the search bounds')
    options['extra_probes'] = extra
    resolution = inputs['cap_resolution']
    if set(resolution) != {'growth', 'max_extensions', 'bracket_steps', 'risk_margin'}:
        raise ValueError('cap_resolution requires growth, max_extensions, bracket_steps, risk_margin')
    if not np.isfinite(resolution['growth']) or resolution['growth'] <= 1:
        raise ValueError('cap_resolution growth must be finite and > 1')
    for key, minimum in [('max_extensions', 0), ('bracket_steps', 1)]:
        value = resolution[key]
        if not np.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f'cap_resolution {key} must be an integer >= {minimum}')
        resolution[key] = int(value)
    if not np.isfinite(resolution['risk_margin']) or resolution['risk_margin'] < 0:
        raise ValueError('cap_resolution risk_margin must be finite and nonnegative')
    reuse = inputs.get('reuse_f_grid')
    if reuse is not None:
        reuse = np.asarray(reuse, dtype=float)
        if (reuse.ndim != 1 or not reuse.size or np.any(~np.isfinite(reuse))
                or np.any(np.diff(reuse) <= 0) or np.any((reuse <= 0) | (reuse >= 1))):
            raise ValueError('reuse_f_grid must be increasing and strictly inside (0, 1)')
        inputs['reuse_f_grid'] = reuse
    return inputs


def _cached_profile(inputs, value, ftilde, *, cache, compute, force):
    args = dict(
        base=replace(inputs['base'], Fneq=float(value), ftilde=float(ftilde)),
        x_key='f', y_key='ftilde', y_values=[float(ftilde)], r=float(inputs['r']),
        mean_convention=inputs['mean_convention'], reference_L=inputs['reference_L'],
        search=deepcopy(inputs['search']),
    )
    options = inputs['f_search']
    resolution = inputs['cap_resolution']
    profile_inputs = dict(**args, f_search=options, cap_resolution=resolution,
                          reuse_f_grid=inputs.get('reuse_f_grid'))

    def produce():
        lo, hi = options['bounds']
        points, identities, tail_checks = {}, [], []
        new_evaluations = 0
        point_cache = ResultCache(cache.directory, enabled=cache.enabled, verbose=False)

        def remember(data, identity, ridge_cap=None):
            identities.append(identity)
            for j, f in enumerate(data['x_values']):
                if lo <= f <= hi:
                    points[float(f)] = dict(
                        risk=float(data['risk'][0, j]), ridge=float(data['ridge'][0, j]),
                        status=str(data['status'][0, j]), reason=str(data['reason'][0, j]),
                        cap=float(args['search']['lambda_max'] if ridge_cap is None else ridge_cap),
                        extensions=0)

        # Reuse finished cuts from the former dense scan, but never compute one.
        reuse = inputs.get('reuse_f_grid')
        if reuse is not None and not force:
            try:
                data, identity = cached_parameter_map(
                    **args, x_values=reuse, cache=point_cache, compute=False)
            except FileNotFoundError:
                pass
            else:
                remember(data, identity)

        def exclude_tail(f):
            point = points[f]
            candidates = [p['risk'] for p in points.values() if np.isfinite(p['risk'])
                          and p['status'] in ('zero', 'interior', 'upper', 'upper_dominated')]
            if not candidates:
                return False
            best = min(candidates)
            threshold = best + resolution['risk_margin']*max(1., abs(best))
            if point['risk'] <= threshold:
                return False
            cfg = replace(args['base'], f=f).to_config(
                mean_convention=args['mean_convention'], reference_L=args['reference_L'])
            check_inputs = dict(cfg=cfg, r=args['r'], threshold=threshold,
                                ridge_cap=point['cap'], steps=resolution['bracket_steps'])
            check, identity = point_cache.get('breadth_ridge_tail', check_inputs,
                lambda: _tail_exclusion_certificate(**check_inputs),
                sources=OPTIMUM_SOURCES, force=force)
            identities.append(identity)
            tail_checks.append(dict(f=f, excluded=check['excluded'],
                                    ridge_cap=point['cap'], threshold=threshold, identity=identity))
            if check['excluded']:
                # The cap value itself is worse; this excludes any improvement
                # beyond it relative to an already evaluated attainable risk.
                point['status'] = 'upper_dominated'
                return True
            return False

        def resolve_cap(f):
            nonlocal new_evaluations
            while points[f]['status'] == 'upper':
                if exclude_tail(f):
                    return
                point = points[f]
                if point['extensions'] >= resolution['max_extensions']:
                    return
                cap = point['cap']*resolution['growth']
                if not np.isfinite(cap):
                    return
                extensions = point['extensions']+1
                expanded = dict(args, search=dict(args['search'], lambda_max=cap))
                try:
                    if force:
                        raise FileNotFoundError
                    data, identity = cached_parameter_map(
                        **expanded, x_values=[f], cache=point_cache, compute=False)
                except FileNotFoundError:
                    data, identity = cached_parameter_map(
                        **expanded, x_values=[f], cache=point_cache, compute=True, force=force)
                    new_evaluations += 1
                remember(data, identity, ridge_cap=cap)
                points[f]['extensions'] = extensions
                if (np.isfinite(point['risk']) and np.isfinite(points[f]['risk'])
                        and points[f]['risk'] > point['risk']
                        + resolution['risk_margin']*max(1., abs(point['risk']))):
                    points[f]['status'] = 'failed'
                    points[f]['reason'] = 'Extended ridge search missed a previously evaluated lower risk'

        def evaluate(f):
            nonlocal new_evaluations
            f = float(f)
            if f not in points:
                try:
                    if force:
                        raise FileNotFoundError
                    data, identity = cached_parameter_map(
                        **args, x_values=[f], cache=point_cache, compute=False)
                except FileNotFoundError:
                    data, identity = cached_parameter_map(
                        **args, x_values=[f], cache=point_cache, compute=True, force=force)
                    new_evaluations += 1
                remember(data, identity)
            if points[f]['status'] == 'upper':
                resolve_cap(f)
            point = points[f]
            return (point['risk'] if np.isfinite(point['risk']) and
                    point['status'] in ('zero', 'interior', 'upper', 'upper_dominated') else np.inf)

        # Cosine spacing retains probes near narrow-library and broad-library ends.
        probes = lo + (hi - lo) * (1. - np.cos(np.linspace(0., np.pi, options['n_probe']))) / 2.
        probes[0], probes[-1] = lo, hi
        probes = np.unique(np.r_[probes, options['extra_probes']])
        for f in probes:
            evaluate(f)
        initial = sorted(points)
        risks = [evaluate(f) for f in initial]
        brackets = []
        for i, risk in enumerate(risks):
            if not np.isfinite(risk):
                continue
            left, right = max(0, i - 1), min(len(initial) - 1, i + 1)
            neighbours = [risks[j] for j in (left, right) if j != i]
            if (all(np.isfinite(neighbours)) and all(risk <= y for y in neighbours)
                    and any(risk < y for y in neighbours)):
                brackets.append((initial[left], initial[right]))
        # A flat probe set still gets one local refinement; no uniqueness claim.
        if not brackets and np.any(np.isfinite(risks)):
            i = int(np.argmin(risks))
            brackets.append((initial[max(0, i - 1)], initial[min(len(initial) - 1, i + 1)]))
        refinements = []
        for left, right in dict.fromkeys(brackets):
            try:
                f, risk = _golden_minimize(evaluate, left, right,
                                           options['tolerance'], options['max_iter'])
            except RuntimeError as exc:
                refinements.append(dict(bounds=(left, right), converged=False, reason=str(exc)))
            else:
                refinements.append(dict(bounds=(left, right), converged=bool(np.isfinite(risk)),
                                        f=float(f), risk=float(risk), reason=''))
        # A better incumbent found during refinement can exclude an earlier tail
        # without any further risk evaluations or changes to its finite search.
        for f in list(points):
            if points[f]['status'] == 'upper':
                exclude_tail(f)
        candidates = [(f, p) for f, p in points.items() if np.isfinite(p['risk'])
                      and p['status'] in ('zero', 'interior', 'upper')]
        statuses = [p['status'] for p in points.values()]
        all_resolved = all(np.isfinite(p['risk']) and p['status'] in ('zero', 'interior', 'upper_dominated')
                           for p in points.values())
        reasons = sorted({p['reason'] for p in points.values() if p['reason']})
        if 'upper' in statuses:
            reasons.append('A ridge tail remains unresolved after the exclusion check and allowed extensions')
        if any(not result['converged'] for result in refinements):
            reasons.append('Breadth refinement did not converge; inspect refinements')
        row = dict(
            ftilde=float(ftilde), f_opt=np.nan, risk_opt=np.nan, ridge_opt=np.nan,
            ridge_status='', at_boundary=False,
            complete=bool(all_resolved and refinements and all(x['converged'] for x in refinements)),
            status_counts={str(k): int(v) for k, v in zip(*np.unique(statuses, return_counts=True))},
            reasons=reasons, n_evaluated_f=len(points), n_new_f=new_evaluations,
            n_refinements=len(refinements), refinements=refinements,
            evaluated_f=np.array(sorted(points)),
            evaluated_risk=np.array([points[f]['risk'] for f in sorted(points)]),
            evaluated_status=[points[f]['status'] for f in sorted(points)],
            tail_checks=tail_checks,
            cap_extensions={f: p['extensions'] for f, p in points.items() if p['extensions']},
            result_ids=identities,
        )
        if candidates:
            f, point = min(candidates, key=lambda item: (item[1]['risk'], item[0]))
            row.update(f_opt=float(f), risk_opt=point['risk'], ridge_opt=point['ridge'],
                       ridge_status=point['status'],
                       at_boundary=bool(min(f - lo, hi - f) <= options['tolerance']))
        return row

    return cache.get('breadth_optimum_profile', profile_inputs, produce,
                     sources=OPTIMUM_SOURCES, compute=compute, force=force)


def cached_breadth_optimum_study(settings, *, cache, compute=False, force=False):
    """Adaptively refine detected basins; reuse individual points and profiles.

    'complete' means evaluated points and local refinements resolved numerically,
    not that a global optimum is certified. A narrow basin between initial probes
    can still be missed; increase n_probe or add extra_probes to investigate it.
    Noncompetitive cap tails are excluded with a sufficient exact-decimal
    polynomial check; other caps trigger bounded, explicit extensions. Remaining
    unresolved tails, failed/infeasible points and nonconvergence stay flagged.
    """
    inputs = _validated_inputs(settings)

    def produce():
        profiles, identities = [], []
        for value in inputs['Fneq_values']:
            rows = []
            for ftilde in inputs['ftilde_grid']:
                row, identity = _cached_profile(inputs, value, ftilde, cache=cache,
                                                compute=True, force=force)
                rows.append(row)
                identities.append(identity)
            profiles.append((value, rows))
        q, eta = inputs['base'].q, inputs['base'].eta
        a = ((1. - eta) * (1. - inputs['ftilde_grid'])
             + eta * inputs['ftilde_grid'] / (q - 1.))
        benchmark = np.full(a.shape, np.nan)
        valid = (a > 0) & (a < 1)
        numerator = np.sqrt((q - 1.) * (1. - a[valid]))
        benchmark[valid] = numerator / (np.sqrt(a[valid]) + numerator)
        return dict(metadata=inputs, profiles=profiles,
                    small_r_f_opt=benchmark, result_ids=identities)

    return cache.get('breadth_optimum_study', inputs, produce,
                     sources=OPTIMUM_SOURCES, compute=compute, force=force)
