"""Cached breadth cuts and their underparameterized zero-ridge benchmark.

This module contains numerical preparation only. The editable figure renderer
lives in the library-breadth notebook. Cache-only reads never evaluate risk,
check moments, or construct analytical risk arrays.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from .parameter_planes import SOURCES, cached_parameter_map, cached_parameter_plane


BENCHMARK_SOURCES = (*SOURCES, Path(__file__).resolve())


def benchmark_arguments(settings, r, value):
    """Capture one exact cut, retaining every supplied scientific input."""
    return dict(
        base=replace(settings['base'], Fneq=float(value),
                     ftilde=float(settings['fixed_ftilde'])),
        x_key='f', x_values=np.asarray(settings['f_grid'], dtype=float),
        y_key='ftilde', y_values=[float(settings['fixed_ftilde'])], r=float(r),
        mean_convention=settings['mean_convention'],
        reference_L=settings['reference_L'], search=deepcopy(settings['search']),
    )


def _requested_values(settings):
    if not np.isfinite(settings['base'].sigma) or settings['base'].sigma <= 0:
        raise ValueError('R/sigma^2 requires strictly positive sigma')
    values = np.asarray(settings['Fneq_values'], dtype=float)
    if (values.ndim != 1 or not values.size or not np.all(np.isfinite(values))
            or np.unique(values).size != values.size):
        raise ValueError('Fneq_values must be a nonempty list of distinct finite values')
    grid = np.asarray(settings['f_grid'], dtype=float)
    if (grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid))
            or np.any(np.diff(grid) <= 0) or np.any((grid <= 0) | (grid >= 1))):
        raise ValueError('f_grid must be increasing, with at least three points strictly in (0, 1)')
    return values.tolist()


def preflight_breadth_benchmark(settings, *, cache, force=False):
    """Moment feasibility only; invalid requested values remain explicit."""
    reports = []
    for value in _requested_values(settings):
        args = benchmark_arguments(settings, 1., value)
        args = {key: val for key, val in args.items() if key not in ('r', 'search')}
        plane, identity = cached_parameter_plane(**args, cache=cache, force=force)
        reports.append(dict(
            Fneq=value, feasible=int(np.sum(plane['feasible'])),
            requested=int(plane['feasible'].size), identity=identity,
            reasons=np.unique(plane['reason'][~plane['feasible']]).tolist(),
        ))
    return reports


def _zero_ridge_benchmark(q, eta, ftilde, r, f_grid):
    """Exact asymptotic zero-ridge expression, only below interpolation."""
    if q <= 1 or not 0 <= eta <= 1 or not 0 < ftilde < 1:
        raise ValueError('The benchmark requires q > 1, eta in [0,1], and 0 < ftilde < 1')
    a = (1. - eta) * (1. - ftilde) + eta * ftilde / (q - 1.)
    if not 0 < a < 1:
        raise ValueError('An interior analytical minimum requires 0 < a < 1')
    f_star = np.sqrt((q - 1.) * (1. - a)) / (
        np.sqrt(a) + np.sqrt((q - 1.) * (1. - a)))
    valid = bool(0 < r < q / (q - 1.))
    risk = None
    if valid:
        risk = r / (q - (q - 1.) * r) * (
            a / (1. - f_grid) + (q - 1.) * (1. - a) / f_grid - 1.)
    return dict(a=float(a), f_star=float(f_star), valid=valid,
                interpolation_r=float(q / (q - 1.)), risk_sigma2=risk)


def cached_breadth_benchmark_case(settings, r, *, cache, compute=False, force=False):
    """Independent fixed-r stage including captured diagnostics and benchmark.

    The optimized risks use the original finite-cap search. Failed/infeasible
    points remain gaps; finite cap hits retain their status and are not infinity
    certificates. Minima in f are sampled minima, without added interpolation.
    """
    values = _requested_values(settings)
    if not np.isfinite(r) or r <= 0:
        raise ValueError('r must be positive and finite')
    inputs = dict(
        base=settings['base'], Fneq_values=values,
        f_grid=np.asarray(settings['f_grid'], dtype=float), r=float(r),
        fixed_ftilde=float(settings['fixed_ftilde']),
        mean_convention=settings['mean_convention'], reference_L=settings['reference_L'],
        search=deepcopy(settings['search']),
    )

    def produce():
        benchmark = _zero_ridge_benchmark(
            inputs['base'].q, inputs['base'].eta, inputs['fixed_ftilde'],
            inputs['r'], inputs['f_grid'])
        curves, identities, diagnostics = [], [], []
        for value in values:
            data, identity = cached_parameter_map(
                **benchmark_arguments(settings, r, value), cache=cache,
                compute=True, force=force)
            curves.append((value, data))
            identities.append(identity)
            y = data['risk'][0] / data['sigma2'][0]
            valid = np.isfinite(y)
            row = dict(Fneq=value, finite_points=int(valid.sum()),
                       requested_points=int(y.size),
                       status_counts={str(k): int(v) for k, v in
                                      zip(*np.unique(data['status'], return_counts=True))},
                       reasons=np.unique(data['reason'][~np.isin(
                           data['status'], ['zero', 'interior', 'upper'])]).tolist())
            if np.any(valid):
                j = int(np.nanargmin(np.where(valid, y, np.nan)))
                row.update(sampled_min_f=float(data['x_values'][j]),
                           sampled_min_risk_sigma2=float(y[j]),
                           sampled_min_ridge=float(data['ridge'][0, j]),
                           sampled_min_status=str(data['status'][0, j]),
                           sampled_min_at_grid_edge=bool(j in (0, len(y)-1)),
                           delta_f_from_small_r_reference=float(data['x_values'][j]-benchmark['f_star']))
                if benchmark['valid']:
                    prediction = benchmark['risk_sigma2'][valid]
                    difference = y[valid] - prediction
                    row.update(max_abs_gap_to_zero_ridge=float(np.max(np.abs(difference))),
                               max_relative_gap_to_zero_ridge=float(
                                   np.max(np.abs(difference) / np.maximum(prediction, 1e-15))),
                               relative_gap_at_sampled_min=float(
                                   (y[j] - benchmark['risk_sigma2'][j])
                                   / benchmark['risk_sigma2'][j]),
                               max_opt_minus_zero_ridge=float(np.max(difference)))
            diagnostics.append(row)
        return dict(metadata=deepcopy(inputs), curves=curves, result_ids=identities,
                    benchmark=benchmark, diagnostics=diagnostics)

    return cache.get('breadth_benchmark_case', inputs, produce,
                     sources=BENCHMARK_SOURCES, compute=compute, force=force)
