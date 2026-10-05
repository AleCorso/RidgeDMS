"""Theory-only total-risk slices and single-axis presentation helpers."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from .theory import DMSTTheory
from .stieltjes import project_stieltjes_evaluator


def compute_risk_slices(cfg, *, fixed_lambdas, fixed_rs, r_grid, lambda_grid):
    """Evaluate actual requested slice coordinates, without grid interpolation."""
    arrays = {}
    for name, values in dict(fixed_lambdas=fixed_lambdas, fixed_rs=fixed_rs,
                             r_grid=r_grid, lambda_grid=lambda_grid).items():
        a = np.asarray(values, dtype=float)
        if a.ndim != 1 or a.size == 0 or np.any(~np.isfinite(a)) or np.any(a <= 0):
            raise ValueError(f'{name} must be a nonempty vector of finite positive values')
        arrays[name] = np.unique(a)
    theory = DMSTTheory(project_stieltjes_evaluator())
    return dict(
        config=cfg, **arrays, R_infinity=theory.infinity_risk(cfg),
        fixed_lambda_risk=np.array([[theory.risk(float(lam), float(r), cfg)
                                    for r in arrays['r_grid']] for lam in arrays['fixed_lambdas']]),
        fixed_r_risk=np.array([[theory.risk(float(lam), float(r), cfg)
                               for lam in arrays['lambda_grid']] for r in arrays['fixed_rs']]),
    )


def plot_risk_slices(data, options, *, title_prefix=''):
    """Return two separate figures; each legend identifies the fixed coordinate."""
    norm = options['normalization']
    scale = {'R_infinity': data['R_infinity'], 'sigma2': data['config'].sigma**2,
             'none': 1.0}[norm]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Normalization must be finite and positive')
    ylabel = options['y_label'] or {'R_infinity': r'$R/R_\infty$',
                                    'sigma2': r'$R/\sigma^2$', 'none': r'$R$'}[norm]
    label_size = options['label_size'] if options['label_size'] is not None else plt.rcParams['axes.labelsize']
    tick_size = options['tick_size'] if options['tick_size'] is not None else plt.rcParams['xtick.labelsize']
    figures = {}
    specs = (
        ('fixed_lambda', data['fixed_lambdas'], data['r_grid'], data['fixed_lambda_risk'],
         'lambda_cmap', r'$r=qL/n$', r'Fixed ridge $\lambda$', 'Risk versus r'),
        ('fixed_r', data['fixed_rs'], data['lambda_grid'], data['fixed_r_risk'],
         'r_cmap', r'Ridge $\lambda$', r'Fixed $r$', 'Risk versus ridge'),
    )
    for key, values, x, risks, cmap_key, xlabel, legend_title, title in specs:
        lo, hi = options['color_span']
        if not 0 <= lo <= hi <= 1:
            raise ValueError('color_span must lie in [0,1]')
        colors = plt.get_cmap(options[cmap_key])(
            np.linspace(lo, hi, len(values)) if len(values) > 1 else [(lo+hi)/2])
        fig, ax = plt.subplots(figsize=(options['figure_width'], options['figure_height']),
                               dpi=options['dpi'], layout='constrained')
        for value, risk, color in zip(values, risks, colors):
            if options['yscale'] == 'log' and np.any(risk <= 0):
                raise ValueError('Log risk axis requires positive values')
            ax.plot(x, risk/scale, color=color, lw=options['linewidth'], ls='-',
                    label=f'{value:g}')
        ax.set_xscale(options[key+'_xscale'])
        ax.set_yscale(options['yscale'])
        ax.set_xlabel(options[key+'_xlabel'] or xlabel, fontsize=label_size)
        ax.set_ylabel(ylabel, fontsize=label_size)
        ax.tick_params(axis='both', which='both', labelsize=tick_size)
        if options[key+'_xlim'] is not None:
            ax.set_xlim(options[key+'_xlim'])
        if options[key+'_ylim'] is not None:
            ax.set_ylim(options[key+'_ylim'])
        if options['show_rinf']:
            ax.axhline(data['R_infinity']/scale, color='0.65', ls=':', lw=1)
        if options['show_legend']:
            ax.legend(title=legend_title, loc=options['legend_loc'],
                      fontsize=options['legend_size'], title_fontsize=options['legend_size'])
        if options['grid']:
            ax.grid(alpha=0.15)
        ax.set_title(f'{title_prefix}: {title}' if title_prefix else title,
                     fontsize=options['title_size'])
        figures[key] = fig
    return figures
