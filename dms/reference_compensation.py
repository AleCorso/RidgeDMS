"""Cached equal-phenotype reference compensation paths for Figure 2.

This adapter changes only the coupled mean coordinates
``m=0, mneq=mneqtilde=value`` at equal train/test breadth and alignment.
It uses the existing moment feasibility checker, risk and ridge optimizer.
Figure construction belongs in the notebook. Cache-only reads do no theory.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .exploration import SweepParameters
from .ridge_diagnostics import kappa_from_lambda, optimize_effective_ridge_over_r
from .simulation import moment_feasibility
from .stieltjes import project_stieltjes_evaluator
from .theory import DMSTTheory


_SOURCE_DIR = Path(__file__).resolve().parent
_PATH_SOURCES = tuple(_SOURCE_DIR / name for name in (
    'reference_compensation.py', 'exploration.py', 'config.py', 'simulation.py',
    'theory.py', 'result_cache.py',
))
SOURCES = _PATH_SOURCES + tuple(_SOURCE_DIR / name for name in (
    'ridge_diagnostics.py', 'stieltjes.py',
))


def _path_inputs(base, mean_values, mean_convention, reference_L):
    """Validate the declared intervention, without converting or checking moments."""
    if not isinstance(base, SweepParameters):
        raise TypeError('base must be SweepParameters')
    values = np.asarray(mean_values, dtype=float)
    if (values.ndim != 1 or not values.size
            or np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0)):
        raise ValueError('mean_values must be finite increasing values (a singleton is allowed)')
    if mean_convention not in ('normalized', 'raw'):
        raise ValueError("mean_convention must be 'normalized' or 'raw'")
    if (not np.isfinite(reference_L) or reference_L <= 0
            or int(reference_L) != reference_L):
        raise ValueError('reference_L must be a positive integer')
    supplied = {key: value.item() if isinstance(value, np.generic) else value
                for key, value in asdict(base).items()}
    if any(value is not None and not np.isfinite(value) for value in supplied.values()):
        raise ValueError('all supplied baseline parameters must be finite')
    errors = []
    if base.q < 3 or int(base.q) != base.q:
        errors.append('q must be an integer >= 3')
    if base.sigma <= 0 or base.G2 <= 0:
        errors.append('sigma and G2 must be positive')
    if base.f != base.ftilde or not 0 < base.f <= 1:
        errors.append('this path requires equal positive breadths f=ftilde <= 1')
    if base.F != base.FTest:
        errors.append('this symmetric path requires FTest=F')
    if not 0 < base.Fneq <= base.F <= 1:
        errors.append('this path requires 0 < Fneq <= F <= 1')
    if not 0 < base.eta <= 1:
        errors.append('this path requires 0 < eta <= 1')
    if any(value != 0 for value in (base.m, base.mneq, base.mneqtilde)):
        errors.append('BASE must start at m=mneq=mneqtilde=0; the path supplies both mismatch means')
    if (base.Cneq is None) == (base.rhoneq is None):
        errors.append('specify exactly one of Cneq or rhoneq')
    elif (base.Cneq if base.rhoneq is None else base.rhoneq) != 0:
        errors.append('this path requires zero mismatch overlap: Cneq=0 or rhoneq=0')
    if errors:
        raise ValueError('; '.join(errors))
    return dict(base=supplied, mean_values=values.tolist(),
                mean_convention=mean_convention, reference_L=int(reference_L))


def _mean_bound(base, mean_convention, reference_L):
    """Necessary PSD bound, using the feasibility checker's rounded site counts.

    The exact finite-L checker remains authoritative: a PSD bound alone does
    not enforce finite-rank constraints, residual signal norm or empty classes.
    Normalized means use eta*L in their definition even when eta*L is not integer;
    the raw bound uses k=round-half-up(eta*L), never an unrounded site count.
    """
    count = base.eta * reference_L
    changed = int(np.floor(count + 0.5))
    equal = reference_L - changed
    gamma2 = (base.q - 1.0) / base.q
    raw_bound = np.sqrt(gamma2 * min(equal * (base.F - base.Fneq),
                                    changed * base.Fneq / 2.0))
    conversion = np.sqrt(gamma2 * count * base.Fneq)
    normalized_bound = raw_bound / conversion
    return dict(
        raw_s_psd=float(raw_bound), normalized_mu_psd=float(normalized_bound),
        grid_coordinate_psd=float(normalized_bound if mean_convention == 'normalized'
                                  else raw_bound),
        changed_sites=changed, equal_sites=equal,
        exact_integer_etaL=bool(np.isclose(count, changed, rtol=0.0, atol=1e-12)),
        note=('Necessary centered-Gram PSD bound using rounded site counts; '
              'moment_feasibility also checks finite ranks and signal norm.'),
    )


def prepare_compensation_path(base, mean_values, *, mean_convention, reference_L):
    """Collect all path points, moment checks and biological invariants.

    With raw mismatch mean s, the changed sites contribute G*s to each
    reference and the equal sites contribute -G*s. Both reference phenotypes
    and both population means are zero; test variance and R_infinity are fixed.
    All quantities are relative to the model's zero-sum phenotype baseline.

    Arrays have shape (len(mean_values),). ``params`` contains every supplied
    point; ``configs`` contains feasible points only, indexed by integer.
    ``feasibility_reports`` retains all completed reports. No requested point
    is clipped or silently adjusted. Conversion errors and all constraint
    failures are retained in ``reason``. This checks moment-level realizability,
    not the existence of a connected sequence path on a particular fixed signal.
    """
    inputs = _path_inputs(base, mean_values, mean_convention, reference_L)
    values = np.asarray(inputs['mean_values'], dtype=float)
    count = values.size
    data = dict(
        mean_values=values, feasible=np.zeros(count, dtype=bool),
        preparation_status=np.full(count, 'infeasible', dtype='<U12'),
        reason=np.full(count, '', dtype=object), params={}, configs={},
        feasibility_reports={}, mean_bound=_mean_bound(base, mean_convention, reference_L),
        metadata=inputs,
        signature=sha256(json.dumps(inputs, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    )
    for name in ('raw_s', 'contribution', 'equal_contribution', 'phi_train', 'phi_test',
                 'population_mean_train', 'population_mean_test', 'population_mean_shift',
                 'test_variance', 'r_infinity', 'sigma2', 'ridge_factor'):
        data[name] = np.full(count, np.nan)
    for index, value in enumerate(values):
        params = replace(base, m=0.0, mneq=float(value), mneqtilde=float(value))
        data['params'][index] = params
        try:
            cfg = params.to_config(mean_convention=mean_convention, reference_L=reference_L)
        except ValueError as exc:
            data['reason'][index] = f'{type(exc).__name__}: {exc}'
            continue
        except Exception as exc:
            data['preparation_status'][index] = 'failed'
            data['reason'][index] = f'{type(exc).__name__}: {exc}'
            continue
        try:
            data['raw_s'][index] = cfg.M_neq
            data['contribution'][index] = cfg.signal_norm * cfg.M_neq
            data['equal_contribution'][index] = cfg.signal_norm * (cfg.M_tot - cfg.M_neq)
            data['phi_train'][index] = cfg.signal_norm * cfg.M_tot
            data['phi_test'][index] = cfg.signal_norm * (cfg.M_tot - cfg.M_neq + cfg.M_neq_test)
            data['population_mean_train'][index] = (
                1.0 - cfg.q * cfg.f_train / (cfg.q - 1.0)) * data['phi_train'][index]
            data['population_mean_test'][index] = (
                1.0 - cfg.q * cfg.f_test / (cfg.q - 1.0)) * data['phi_test'][index]
            data['population_mean_shift'][index] = cfg.signal_norm * DMSTTheory.population_mean_shift(cfg)
            data['test_variance'][index] = DMSTTheory.test_signal_variance(cfg)
            data['r_infinity'][index] = DMSTTheory.infinity_risk(cfg)
            data['sigma2'][index] = cfg.sigma**2
            data['ridge_factor'][index] = (cfg.q - 1.0) / cfg.f_train
            report = moment_feasibility(int(reference_L), cfg)
        except Exception as exc:
            data['preparation_status'][index] = 'failed'
            data['reason'][index] = f'{type(exc).__name__}: {exc}'
            continue
        data['feasibility_reports'][index] = report
        data['feasible'][index] = report.feasible
        data['reason'][index] = '; '.join(report.messages)
        if report.feasible:
            data['preparation_status'][index] = 'feasible'
            data['configs'][index] = cfg
    return data


def cached_compensation_path(base, mean_values, *, mean_convention, reference_L,
                             cache, force=False, compute=True):
    """Return (path, identity); feasibility and invariants are shared across r."""
    inputs = _path_inputs(base, mean_values, mean_convention, reference_L)
    return cache.get(
        'compensation_path', inputs,
        lambda: prepare_compensation_path(
            base, inputs['mean_values'], mean_convention=mean_convention,
            reference_L=reference_L),
        sources=_PATH_SOURCES, force=force, compute=compute,
    )


def _search_options(search):
    options = dict(search)
    if options.pop('method', 'nonnegative') != 'nonnegative':
        raise ValueError("compensation curves require search method='nonnegative'")
    if set(options) != {'lambda_max', 'n_grid'}:
        raise ValueError('search must provide lambda_max and n_grid (and optional method)')
    cap, count = options['lambda_max'], options['n_grid']
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError('lambda_max must be finite and positive')
    if not np.isfinite(count) or int(count) != count or count < 3:
        raise ValueError('n_grid must be an integer >= 3')
    return dict(lambda_max=float(cap), n_grid=int(count))


def cached_compensation_curve(base, mean_values, r, *, mean_convention, reference_L,
                              search, cache, force=False, compute=True):
    """Return (curve, identity), using the existing nonnegative ridge optimizer.

    Results retain the complete prepared path and add ridge, risk, kappa,
    zero_slope and status arrays. Status is zero/interior/upper/infeasible/failed;
    upper is a finite search-cap hit, not a certified infinite optimum. Every
    feasible point is attempted, and numerical failures are reported separately.

    force=True rebuilds this curve while reusing the independent path cache.
    compute=False only looks up the curve: it never invokes conversion,
    feasibility, risk evaluation, optimization or a path-cache producer.
    """
    inputs = _path_inputs(base, mean_values, mean_convention, reference_L)
    if not np.isfinite(r) or r <= 0:
        raise ValueError('r must be finite and positive')
    options = _search_options(search)
    curve_inputs = dict(**inputs, r=float(r), search=dict(method='nonnegative', **options))

    def produce():
        path, path_identity = cached_compensation_path(
            base, inputs['mean_values'], mean_convention=mean_convention,
            reference_L=reference_L, cache=cache)
        data = dict(path, r=float(r), path_identity=path_identity, metadata=curve_inputs,
                    reason=path['reason'].copy(), status=path['preparation_status'].copy())
        for name in ('ridge', 'risk', 'kappa', 'zero_slope'):
            data[name] = np.full(path['mean_values'].shape, np.nan)
        for index, cfg in path['configs'].items():
            try:
                optimum = optimize_effective_ridge_over_r([float(r)], cfg, **options)
                if not np.isfinite(data['r_infinity'][index]):
                    raise ValueError('nonfinite infinite-ridge reference risk')
            except Exception as exc:
                data['status'][index] = 'failed'
                data['reason'][index] = f'{type(exc).__name__}: {exc}'
                continue
            data['ridge'][index] = optimum.ridge_lambdas[0]
            data['risk'][index] = optimum.risks[0]
            data['kappa'][index] = optimum.kappas[0]
            data['zero_slope'][index] = optimum.zero_slopes[0]
            data['status'][index] = optimum.status[0]
        return data

    return cache.get('compensation_curve', curve_inputs, produce, sources=SOURCES,
                     force=force, compute=compute)


def cached_fixed_ridge_check(base, mean_values, r_values, ridge_values, *,
                             mean_convention, reference_L, cache,
                             force=False, compute=True):
    """Check the compensation mechanism with the original positive-ridge theory.

    At equal breadth b and raw mismatch means +/-s, the difference from s=0 is
    ``[G*a*q/(q-1)*kappa*(1/(t+kappa)-1/(1+kappa))]**2 * s**2``,
    where a=1-q*b/(q-1), t=q*(1-b). This small, explicit diagnostic compares
    that expression and sign symmetry to DMSTTheory.risk. It performs no ridge
    optimization. Arrays are indexed (r, ridge, mean); all requested points and
    their negatives are checked for feasibility first. Absolute tolerances are
    1e-9 times max(1, |R(0)|, |R(s)|, |R(-s)|).

    The targeted check has its own cache. compute=False does not evaluate any
    theory or feasibility. Invalid points are reported, not silently removed.
    """
    inputs = _path_inputs(base, mean_values, mean_convention, reference_L)
    axes = []
    for name, values in (('r_values', r_values), ('ridge_values', ridge_values)):
        values = np.asarray(values, dtype=float)
        if (values.ndim != 1 or not values.size or np.any(~np.isfinite(values))
                or np.any(values <= 0)):
            raise ValueError(f'{name} must be a nonempty positive finite vector')
        axes.append(values.tolist())
    check_inputs = dict(**inputs, r_values=axes[0], ridge_values=axes[1])

    def produce():
        means = np.asarray(inputs['mean_values'], dtype=float)
        rs, ridges = (np.asarray(values, dtype=float) for values in axes)
        signed_values = np.unique(np.concatenate(([0.0], means, -means)))
        path, path_identity = cached_compensation_path(
            base, signed_values, mean_convention=mean_convention,
            reference_L=reference_L, cache=cache)
        indices = {float(value): index for index, value in enumerate(signed_values)}
        shape = (rs.size, ridges.size, means.size)
        data = dict(
            metadata=check_inputs, mean_values=means, r_values=rs, ridge_values=ridges,
            path_identity=path_identity, path=path,
            status=np.full(shape, 'unchecked', dtype='<U12'),
            reason=np.full(shape, '', dtype=object),
            identity_ok=np.zeros(shape, dtype=bool),
            sign_symmetry_ok=np.zeros(shape, dtype=bool),
        )
        for name in ('risk_zero', 'risk_plus', 'risk_minus', 'computed_delta',
                     'predicted_delta', 'identity_abs_error', 'sign_symmetry_abs_error',
                     'tolerance', 'raw_s', 'kappa'):
            data[name] = np.full(shape, np.nan)
        theory = DMSTTheory(project_stieltjes_evaluator())
        a = 1.0 - base.q * base.f / (base.q - 1.0)
        t = base.q * (1.0 - base.f)
        zero_index = indices[0.0]
        for ir, r in enumerate(rs):
            for il, ridge in enumerate(ridges):
                zero_cfg = path['configs'].get(zero_index)
                zero_risk, baseline_error = np.nan, ''
                if zero_cfg is not None:
                    try:
                        zero_risk = float(theory.risk(float(ridge), float(r), zero_cfg))
                    except Exception as exc:
                        baseline_error = f'{type(exc).__name__}: {exc}'
                for im, value in enumerate(means):
                    index = (ir, il, im)
                    plus_index, minus_index = indices[float(value)], indices[float(-value)]
                    needed = (zero_index, plus_index, minus_index)
                    missing = [i for i in needed if i not in path['configs']]
                    if missing:
                        data['status'][index] = (
                            'failed' if any(path['preparation_status'][i] == 'failed' for i in missing)
                            else 'infeasible')
                        data['reason'][index] = '; '.join(
                            f'mean={signed_values[i]:g}: {path["reason"][i]}'
                            for i in dict.fromkeys(missing))
                        continue
                    if baseline_error:
                        data['status'][index] = 'failed'
                        data['reason'][index] = baseline_error
                        continue
                    plus_cfg, minus_cfg = path['configs'][plus_index], path['configs'][minus_index]
                    try:
                        plus_risk = float(theory.risk(float(ridge), float(r), plus_cfg))
                        minus_risk = float(theory.risk(float(ridge), float(r), minus_cfg))
                        kappa = float(kappa_from_lambda(float(ridge), float(r), zero_cfg))
                        coefficient = (zero_cfg.signal_norm * a * base.q / (base.q - 1.0)
                                       * kappa * (1.0 / (t + kappa) - 1.0 / (1.0 + kappa)))
                        predicted = coefficient**2 * plus_cfg.M_neq**2
                        if not np.all(np.isfinite([zero_risk, plus_risk, minus_risk, predicted])):
                            raise ValueError('nonfinite fixed-ridge diagnostic')
                    except Exception as exc:
                        data['status'][index] = 'failed'
                        data['reason'][index] = f'{type(exc).__name__}: {exc}'
                        continue
                    actual = plus_risk - zero_risk
                    identity_error = abs(actual - predicted)
                    symmetry_error = abs(plus_risk - minus_risk)
                    tolerance = 1e-9 * max(1.0, abs(zero_risk), abs(plus_risk), abs(minus_risk))
                    values = dict(risk_zero=zero_risk, risk_plus=plus_risk, risk_minus=minus_risk,
                                  computed_delta=actual, predicted_delta=predicted,
                                  identity_abs_error=identity_error,
                                  sign_symmetry_abs_error=symmetry_error,
                                  tolerance=tolerance, raw_s=plus_cfg.M_neq, kappa=kappa)
                    for name, number in values.items():
                        data[name][index] = number
                    data['identity_ok'][index] = identity_error <= tolerance
                    data['sign_symmetry_ok'][index] = symmetry_error <= tolerance
                    data['status'][index] = 'checked'
        checked = data['status'] == 'checked'
        data['max_identity_error'] = (float(np.max(data['identity_abs_error'][checked]))
                                      if np.any(checked) else np.nan)
        data['max_sign_symmetry_error'] = (float(np.max(data['sign_symmetry_abs_error'][checked]))
                                           if np.any(checked) else np.nan)
        data['identity_passed'] = bool(np.all(data['identity_ok']))
        data['sign_symmetry_passed'] = bool(np.all(data['sign_symmetry_ok']))
        data['all_passed'] = data['identity_passed'] and data['sign_symmetry_passed']
        return data

    return cache.get('compensation_check', check_inputs, produce, sources=SOURCES,
                     force=force, compute=compute)
