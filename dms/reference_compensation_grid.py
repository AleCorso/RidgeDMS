"""Cached mean/aspect-ratio grids for the equal-reference compensation study.

Numerical work delegates to reference_compensation; this module only assembles
independent map and exact-cut stages. Rendering remains in the active notebook.
Cache-only reads do not evaluate feasibility, risk, or ridge optima.
"""
from pathlib import Path

import numpy as np

from .reference_compensation import (
    SOURCES as CURVE_SOURCES,
    _path_inputs,
    _search_options,
    cached_compensation_curve,
)


SOURCES = (*CURVE_SOURCES, Path(__file__).resolve())


def cached_compensation_grid(base, mean_values, r_values, *, stage,
                             mean_convention, reference_L, search, cache,
                             curve_cache=None, force=False, compute=True):
    """Return a full grid and its identity, with arrays indexed (r, mean).

    stage distinguishes the map, mean_cuts, and r_cuts cache records. Every
    requested grid point is evaluated directly, never interpolated. Completed
    individual r curves are checkpoints and may be reused from an existing
    compensation cache. A forced grid rebuild also forces its curve searches,
    while retaining the independent moment-feasibility paths.

    Invalid moments and numerical failures stay masked with their reasons and
    distinct status values. An upper status is a finite search-cap hit, not a
    certificate that infinite ridge is optimal.
    """
    if stage not in ('map', 'mean_cuts', 'r_cuts'):
        raise ValueError('stage must be map, mean_cuts, or r_cuts')
    inputs = _path_inputs(base, mean_values, mean_convention, reference_L)
    rs = np.asarray(r_values, dtype=float)
    if (rs.ndim != 1 or not rs.size or np.any(~np.isfinite(rs))
            or np.any(rs <= 0) or np.any(np.diff(rs) <= 0)):
        raise ValueError('r_values must be finite positive increasing values')
    inputs.update(stage=stage, r_values=rs.tolist(),
                  search=dict(method='nonnegative', **_search_options(search)))

    def produce():
        rows, row_ids = [], []
        for r in rs:
            data, identity = cached_compensation_curve(
                base, inputs['mean_values'], float(r),
                mean_convention=mean_convention, reference_L=reference_L,
                search=inputs['search'], cache=curve_cache if curve_cache is not None else cache,
                force=force, compute=True)
            rows.append(data)
            row_ids.append(identity)
        names = ('feasible', 'preparation_status', 'reason', 'status',
                 'ridge', 'risk', 'kappa', 'zero_slope', 'r_infinity', 'sigma2',
                 'raw_s', 'phi_train', 'phi_test', 'population_mean_train',
                 'population_mean_test', 'population_mean_shift', 'test_variance')
        arrays = {name: np.stack([row[name] for row in rows]) for name in names}
        return dict(**arrays, mean_values=np.asarray(inputs['mean_values']),
                    r_values=rs, metadata=inputs, row_identities=row_ids,
                    mean_bound=rows[0]['mean_bound'])

    return cache.get('reference_compensation_grid', inputs, produce,
                     sources=SOURCES, force=force, compute=compute)
