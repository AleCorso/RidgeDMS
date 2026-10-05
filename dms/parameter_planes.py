"""Cached fixed-r parameter planes using the existing Figure 2 optimizer.

The x and y coordinates are two distinct floating-point ``SweepParameters``
fields. Arrays have shape (len(y_values), len(x_values)); singleton axes give
exact one-dimensional cuts without interpolation. Parameters outside the two
axes remain at their supplied values, including every mean and cross-moment.
Figure construction stays in the notebook. Cache-only reads never run theory.
"""
from __future__ import annotations

from dataclasses import asdict, fields, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .exploration import SweepParameters
from .ridge_diagnostics import optimize_effective_ridge_over_r
from .simulation import moment_feasibility
from .theory import DMSTTheory


_SOURCE_DIR = Path(__file__).resolve().parent
_PLANE_SOURCES = tuple(_SOURCE_DIR / name for name in (
    'parameter_planes.py', 'exploration.py', 'config.py', 'simulation.py',
    'result_cache.py',
))
_MAP_SOURCES = _PLANE_SOURCES + tuple(_SOURCE_DIR / name for name in (
    'ridge_diagnostics.py', 'theory.py', 'stieltjes.py',
))
# Notebook caches can include these paths in their figure identities.
SOURCES = _MAP_SOURCES


def _plane_inputs(base, x_key, x_values, y_key, y_values,
                  mean_convention, reference_L):
    if not isinstance(base, SweepParameters):
        raise TypeError('base must be SweepParameters')
    names = {field.name for field in fields(SweepParameters)} - {'q'}
    if x_key not in names or y_key not in names:
        raise ValueError('x_key and y_key must name floating-point SweepParameters fields')
    if x_key == y_key:
        raise ValueError('x_key and y_key must be distinct')
    axes = []
    for name, values in (('x_values', x_values), ('y_values', y_values)):
        values = np.asarray(values, dtype=float)
        if (values.ndim != 1 or not values.size
                or np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0)):
            raise ValueError(f'{name} must contain finite increasing values (a singleton is allowed)')
        axes.append(values.tolist())
    if mean_convention not in ('normalized', 'raw'):
        raise ValueError("mean_convention must be 'normalized' or 'raw'")
    if not np.isfinite(reference_L) or reference_L <= 0 or int(reference_L) != reference_L:
        raise ValueError('reference_L must be a positive integer')
    base_values = {key: value.item() if isinstance(value, np.generic) else value
                   for key, value in asdict(base).items()}
    if any(value is not None and not np.isfinite(value) for value in base_values.values()):
        raise ValueError('all supplied baseline parameters must be finite')
    return dict(base=base_values, x_key=x_key, x_values=axes[0],
                y_key=y_key, y_values=axes[1], mean_convention=mean_convention,
                reference_L=int(reference_L))


def prepare_parameter_plane(base, x_key, x_values, y_key, y_values, *,
                            mean_convention, reference_L):
    """Collect every point's feasibility without adjusting other coordinates.

    ``feasibility_reports`` retains complete reports indexed by (iy, ix).
    Conversion validation errors are infeasible; unexpected conversion or
    feasibility errors are failed. All messages are retained in ``reason``.
    ``sigma2`` and ``ridge_factor=(q-1)/f`` are pointwise arrays, including when
    sigma or f is swept. A baseline marker coordinate is None when that optional
    baseline field is unset. Choosing Cneq or rhoneq does not clear its alternative:
    the caller must explicitly supply the desired cross-moment convention.
    """
    inputs = _plane_inputs(base, x_key, x_values, y_key, y_values,
                           mean_convention, reference_L)
    xs, ys = np.asarray(inputs['x_values']), np.asarray(inputs['y_values'])
    shape = (len(ys), len(xs))
    feasible = np.zeros(shape, dtype=bool)
    status = np.full(shape, 'infeasible', dtype='<U12')
    reason = np.full(shape, '', dtype=object)
    sigma2, ridge_factor = np.full(shape, np.nan), np.full(shape, np.nan)
    configs, reports = {}, {}
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            index = (iy, ix)
            params = replace(base, **{x_key: float(x), y_key: float(y)})
            try:
                cfg = params.to_config(mean_convention=mean_convention,
                                       reference_L=reference_L)
            except ValueError as exc:
                reason[index] = f'{type(exc).__name__}: {exc}'
                continue
            except Exception as exc:
                status[index] = 'failed'
                reason[index] = f'{type(exc).__name__}: {exc}'
                continue
            sigma2[index] = cfg.sigma**2
            ridge_factor[index] = (cfg.q - 1) / cfg.f_train
            try:
                report = moment_feasibility(int(reference_L), cfg)
            except Exception as exc:
                status[index] = 'failed'
                reason[index] = f'{type(exc).__name__}: {exc}'
                continue
            reports[index] = report
            feasible[index] = report.feasible
            reason[index] = '; '.join(report.messages)
            if report.feasible:
                status[index] = 'feasible'
                configs[index] = cfg
    signature = sha256(json.dumps(inputs, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(x_values=xs, y_values=ys, x_key=x_key, y_key=y_key,
                feasible=feasible, reason=reason, preparation_status=status,
                configs=configs, feasibility_reports=reports,
                baseline_x=inputs['base'][x_key], baseline_y=inputs['base'][y_key],
                sigma2=sigma2, ridge_factor=ridge_factor,
                metadata=inputs, signature=signature)


def cached_parameter_plane(base, x_key, x_values, y_key, y_values, *,
                           mean_convention, reference_L, cache,
                           force=False, compute=True):
    """Return (plane, identity), reusing feasibility at different fixed r.

    With compute=False, only cache identity validation and lookup take place;
    parameter conversion and feasibility evaluation remain inside the producer.
    """
    inputs = _plane_inputs(base, x_key, x_values, y_key, y_values,
                           mean_convention, reference_L)
    return cache.get(
        'parameter_plane', inputs,
        lambda: prepare_parameter_plane(
            base, x_key, inputs['x_values'], y_key, inputs['y_values'],
            mean_convention=mean_convention, reference_L=reference_L,
        ),
        sources=_PLANE_SOURCES, force=force, compute=compute,
    )


def _search_options(search):
    options = dict(search)
    if options.pop('method', 'nonnegative') != 'nonnegative':
        raise ValueError("parameter maps require search method='nonnegative'")
    if set(options) != {'lambda_max', 'n_grid'}:
        raise ValueError('search must provide lambda_max and n_grid (and optional method)')
    cap, count = options['lambda_max'], options['n_grid']
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError('lambda_max must be finite and positive')
    if not np.isfinite(count) or int(count) != count or count < 3:
        raise ValueError('n_grid must be an integer >= 3')
    return dict(lambda_max=float(cap), n_grid=int(count))


def cached_parameter_map(base, x_key, x_values, y_key, y_values, r, *,
                         mean_convention, reference_L, search, cache,
                         force=False, compute=True):
    """Return (data, identity) for one fixed r plane or exact singleton-axis cut.

    Arrays contain ridge, risk, r_infinity, kappa, zero_slope, ridge_factor and
    sigma2. Status is zero/interior/upper/infeasible/failed. ``upper`` denotes a
    finite search-cap hit, not certified infinity. Feasible points with numerical
    failures remain feasible, with failed status and a diagnostic reason; every
    feasible point is attempted. The existing optimizer includes zero and searches
    the supplied finite range; its finite grid is not a proof of global optimality.

    ``force`` rebuilds this r map but reuses independent feasibility results.
    ``compute=False`` reads the completed map directly and never invokes parameter
    conversion, feasibility checks, risk evaluation or optimization. Cache inputs
    include every supplied baseline coordinate, both axes, r, reference length,
    mean convention and search control, plus relevant implementation fingerprints.
    """
    inputs = _plane_inputs(base, x_key, x_values, y_key, y_values,
                           mean_convention, reference_L)
    if not np.isfinite(r) or r <= 0:
        raise ValueError('r must be finite and positive')
    options = _search_options(search)
    map_inputs = dict(**inputs, r=float(r), search=dict(method='nonnegative', **options))

    def produce():
        plane, plane_identity = cached_parameter_plane(
            base, x_key, inputs['x_values'], y_key, inputs['y_values'],
            mean_convention=mean_convention, reference_L=reference_L, cache=cache,
        )
        shape = plane['feasible'].shape
        data = dict(plane, r=float(r), plane_identity=plane_identity,
                    metadata=map_inputs, reason=plane['reason'].copy(),
                    status=plane['preparation_status'].copy())
        for name in ('ridge', 'risk', 'r_infinity', 'kappa', 'zero_slope'):
            data[name] = np.full(shape, np.nan)
        for index, cfg in plane['configs'].items():
            try:
                optimum = optimize_effective_ridge_over_r([float(r)], cfg, **options)
                r_infinity = float(DMSTTheory.infinity_risk(cfg))
                if not np.isfinite(r_infinity):
                    raise ValueError('nonfinite infinite-ridge reference risk')
            except Exception as exc:
                data['status'][index] = 'failed'
                data['reason'][index] = f'{type(exc).__name__}: {exc}'
                continue
            data['ridge'][index] = optimum.ridge_lambdas[0]
            data['risk'][index] = optimum.risks[0]
            data['r_infinity'][index] = r_infinity
            data['kappa'][index] = optimum.kappas[0]
            data['zero_slope'][index] = optimum.zero_slopes[0]
            data['status'][index] = optimum.status[0]
        return data

    return cache.get('parameter_map', map_inputs, produce, sources=_MAP_SOURCES,
                     force=force, compute=compute)
