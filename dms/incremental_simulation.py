"""Persistent trial prefixes for the Figures 1/3 simulation averages.

The legacy gr_job and its seeded realizations are unchanged. This cache uses
independent per-r streams derived from the same user seed: changing the trial
count cannot change a model or an earlier trial at another r. Legacy aggregate
caches remain readable, but lack the raw trials/model/RNG needed to extend them.
"""
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np

from .jobs import GRJobResult, gr_job_sizes
from .simulation import gr_model, gr_trial


STREAM_SCHEME = 'per-r-seedsequence-v1'
_DIRECTORY = Path(__file__).resolve().parent
LEGACY_SOURCES = tuple(_DIRECTORY / f'{name}.py' for name in ('config', 'simulation', 'jobs'))
SOURCES = LEGACY_SOURCES + (Path(__file__).resolve(), _DIRECTORY / 'result_cache.py')


def _validate(r_values, ridge_lambdas, cfg, budget, trials, seed):
    if isinstance(trials, (bool, np.bool_)) or not isinstance(trials, (int, np.integer)) or trials < 2:
        raise ValueError('trials must be an integer of at least 2 to estimate a standard error')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rs = np.asarray(r_values, dtype=float)
    lambdas = np.asarray(ridge_lambdas, dtype=float)
    if lambdas.ndim != 1 or not lambdas.size or np.any(~np.isfinite(lambdas)) or np.any(lambdas <= 0):
        raise ValueError('ridge_lambdas must be a nonempty positive finite vector')
    sizes = gr_job_sizes(rs, cfg, budget)
    return rs, lambdas, sizes


def _start_point(index, length, lambdas, cfg, seed):
    started = perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(index,)))
    model = gr_model(int(length), cfg, rng)
    return dict(model=model, rng_state=deepcopy(rng.bit_generator.state),
                risks=np.empty((0, len(lambdas))), trial_seconds=np.empty(0),
                model_seconds=perf_counter() - started)


def _validate_point(state, length, n_lambdas):
    risks = state['risks']
    if (state['model'].L != length or risks.ndim != 2 or risks.shape[1] != n_lambdas
            or not np.all(np.isfinite(risks)) or state['trial_seconds'].shape != (len(risks),)):
        raise ValueError('Invalid incremental checkpoint; use force=True to rebuild this simulation')


def cached_gr_job(r_values, ridge_lambdas, cfg, budget, trials, *, seed=17,
                  cache, force=False, compute=True, progress=None):
    """Return (GRJobResult, result_identity, reuse_report).

    A checkpoint identity includes the complete simulation plan except trials.
    Each completed trial is saved atomically; only the missing suffix is run.
    Means/standard errors always use the first requested number of raw trials,
    not an average of batch means/errors. Smaller requests leave longer stored
    prefixes intact. Final aggregate cache identities still include trials.

    compute=False reads exact aggregate caches only, including compatible legacy
    caches, and never draws a model, samples trials, or computes a new summary.
    force=True deliberately rebuilds the requested trials and their checkpoints.
    With cache.enabled=False, no disk state is read or written.
    """
    rs, lambdas, sizes = _validate(r_values, ridge_lambdas, cfg, budget, trials, seed)
    inputs = dict(config=cfg.as_dict(), budget=budget, trials=int(trials),
                  r_values=rs, ridge_lambdas=lambdas, seed=int(seed))
    reused = np.zeros(len(rs), dtype=int)
    computed = np.zeros(len(rs), dtype=int)

    def report(legacy=False):
        return dict(reused_trials_per_r=reused.copy(), new_trials_per_r=computed.copy(),
                    requested_trials=int(trials), legacy_aggregate=legacy,
                    stream_scheme='legacy-gr-job' if legacy else STREAM_SCHEME)

    if not force:
        try:
            job, identity = cache.get('simulation_incremental', inputs, sources=SOURCES, compute=False)
        except FileNotFoundError:
            pass
        else:
            reused[:] = trials
            if cache.verbose:
                print(f'simulation trials: reused {len(rs)*trials}; computed 0 ({trials} per r)')
            return job, identity, report()

    if not compute:
        # Keep existing figures restorable without rerunning scientific work.
        job, identity = cache.get('simulation', inputs, sources=LEGACY_SOURCES,
                                  force=force, compute=False)
        reused[:] = trials
        if cache.verbose:
            print('Legacy averages restored; raw trial checkpoints are not available.')
        return job, identity, report(legacy=True)

    base = {key: value for key, value in inputs.items() if key != 'trials'}
    base['stream_scheme'] = STREAM_SCHEME
    data = np.empty((len(lambdas), len(rs), 3))
    measured = []
    seconds = 0.0
    # Use the same cache class/options; silence per-trial file messages in favour
    # of one readable reuse/extension report per r and a final total.
    point_cache = type(cache)(cache.directory, enabled=cache.enabled, verbose=False)
    for j, (n, length) in enumerate(sizes):
        point_inputs = dict(base, r_index=j, n=int(n), L=int(length))
        state, _ = point_cache.get(
            'simulation_trials', point_inputs,
            lambda: _start_point(j, length, lambdas, cfg, seed),
            sources=SOURCES, force=force,
        )
        _validate_point(state, length, len(lambdas))
        completed = len(state['risks'])
        reused[j] = min(completed, trials)
        if cache.verbose:
            print(f'r={rs[j]:g}: reusing {reused[j]}/{trials} trials; '
                  f'computing {max(0, trials-completed)} more')
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state['rng_state'])
        for t in range(completed, trials):
            started = perf_counter()
            risk = np.asarray(gr_trial(int(n), lambdas, state['model'], cfg, rng), dtype=float)
            if risk.shape != (len(lambdas),) or not np.all(np.isfinite(risk)):
                raise ValueError(f'Invalid simulated risk at r={rs[j]:g}, trial={t+1}; checkpoint retained')
            state['risks'] = np.vstack((state['risks'], risk))
            state['trial_seconds'] = np.append(state['trial_seconds'], perf_counter() - started)
            state['rng_state'] = deepcopy(rng.bit_generator.state)
            # Commit before reporting completion, so an interrupted job resumes
            # after the last saved trial rather than repeating a completed batch.
            point_cache.get('simulation_trials', point_inputs, lambda: state,
                            sources=SOURCES, force=True)
            computed[j] += 1
            if progress is not None:
                progress(dict(done=int(reused[:j+1].sum()+computed.sum()),
                              total=len(rs)*trials, requested_r=float(rs[j]),
                              realized_r=float(cfg.q*length/n), n=int(n), L=int(length), trial=t+1))
        risks = state['risks'][:trials]
        data[:, j, 0] = cfg.q * length / n
        data[:, j, 1] = risks.mean(axis=0)
        data[:, j, 2] = risks.std(axis=0, ddof=1) / np.sqrt(trials)
        measured.append(state['model'].measured_parameters)
        seconds += state['model_seconds'] + float(state['trial_seconds'][:trials].sum())

    job = GRJobResult(data=data, ridge_lambdas=lambdas.copy(), config=cfg,
                      measured_parameters=tuple(measured), sizes=sizes,
                      seconds=seconds, requested_r=rs.copy())
    job, identity = cache.get('simulation_incremental', inputs, lambda: job,
                              sources=SOURCES, force=True)
    if cache.verbose:
        print(f'simulation trials: reused {reused.sum()}; computed {computed.sum()} '
              f'({trials} per r)')
    return job, identity, report()
