"""Independent training/test differing-site signal coordinates for sweeps."""
from dataclasses import replace

import numpy as np

from .exploration import parameter_sweep_cases


def differing_signal_sweep_cases(baseline, parameter, values, *, fixed_other,
                                mean_convention, reference_L):
    """Hold F and the opposite differing-site fraction fixed; derive FTest.

    Normalized means and rhoneq retain the caller's baseline values. The usual
    full moment-feasibility check is applied to every requested configuration.
    """
    if parameter not in ('Fneq', 'Fneqtilde'):
        raise ValueError('parameter must be Fneq or Fneqtilde')
    values = np.asarray(values, float)
    if values.ndim != 1 or not values.size or np.any(~np.isfinite(values)):
        raise ValueError('values must be a nonempty finite vector')
    if not np.isfinite(fixed_other) or fixed_other < 0:
        raise ValueError('fixed_other must be finite and nonnegative')
    cases, errors = [], []
    for value in values:
        train, test = ((value, fixed_other) if parameter == 'Fneq'
                       else (fixed_other, value))
        try:
            if min(train, test) < 0:
                raise ValueError('differing-site fractions must be nonnegative')
            case, = parameter_sweep_cases(
                baseline, 'FTest', [baseline.F - train + test],
                fixed=dict(Fneq=float(train)), mean_convention=mean_convention,
                reference_L=reference_L,
            )
        except ValueError as exc:
            errors.append(f'{parameter}={value:g}: {exc}')
        else:
            cases.append(replace(case, parameter=parameter, value=float(value)))
    if errors:
        raise ValueError('\n'.join(errors))
    return tuple(cases)
