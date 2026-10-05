"""Shared figure assembly for the provisional paper notebooks.

No calculations run on import. Plot functions consume saved arrays only.
"""
import numpy as np
import matplotlib.pyplot as plt
from .ridge_diagnostics import effective_risk, kappa_from_lambda, optimize_effective_ridge_over_r
from .landscape_slices import compute_risk_slices
from .jobs import gr_job, gr_job_sizes
from .simulation import moment_feasibility


def compute_panels(cfg, *, r_grid, lambda_grid, fixed_lambdas, fixed_rs, search):
    rs, ls = np.asarray(r_grid,float), np.asarray(lambda_grid,float)
    for a in (rs, ls):
        if a.ndim!=1 or len(a)<2 or np.any(~np.isfinite(a)) or np.any(a<=0) or np.any(np.diff(a)<=0):
            raise ValueError('Map grids must be positive finite strictly increasing vectors')
    slices=compute_risk_slices(cfg,fixed_lambdas=fixed_lambdas,fixed_rs=fixed_rs,
                              r_grid=rs,lambda_grid=ls)
    risk=np.array([[effective_risk(kappa_from_lambda(float(lam),float(r),cfg),float(r),cfg).risk
                    for r in rs] for lam in ls])
    return dict(config=cfg, slices=slices, r=rs, ridge=ls, risk=risk,
                optimum=optimize_effective_ridge_over_r(rs,cfg,**search), search=dict(search))


def run_panel_simulations(cfg, *, r_values, fixed_rs, fixed_lambdas, slice_lambdas,
                          budget, trials, seed, fixed_r_points=False):
    """Retain the project's budget sizing; audit every planned size first."""
    groups=[('fixed_lambda',r_values,fixed_lambdas)]
    if fixed_r_points: groups.append(('fixed_r',fixed_rs,slice_lambdas))
    errors=[]
    for kind,rs,_ in groups:
        for n,L in gr_job_sizes(rs,cfg,budget):
            report=moment_feasibility(int(L),cfg)
            if not report.feasible: errors.append(dict(kind=kind,n=int(n),L=int(L),messages=report.messages))
    if errors: raise ValueError(errors)
    return {kind:gr_job(rs,ls,cfg,budget,trials,seed=seed+i)
            for i,(kind,rs,ls) in enumerate(groups)}


def ridge_path(ax, r, y, *, color, width, style='dashed', ratio=5., label=None, jump_positions=()):
    """Positive path; only explicit transitions may be dashed (ratio is ignored)."""
    if style not in ('solid','dashed','gap'):
        raise ValueError('Require jump style solid/dashed/gap')
    r,y=np.asarray(r),np.asarray(y)
    ax.plot([],[],color=color,lw=width,label=label)
    for i in range(len(r)-1):
        a,b=y[i:i+2]
        if not np.isfinite(a+b) or min(a,b)<=0: continue
        marked=any(r[i]<position<=r[i+1] for position in jump_positions)
        if not marked or style=='solid':
            ax.plot(r[i:i+2],[a,b],color=color,lw=width)
        elif style=='dashed':
            mid=np.sqrt(r[i]*r[i+1])
            ax.plot([r[i],mid],[a,a],color=color,lw=width)
            ax.plot([mid,r[i+1]],[b,b],color=color,lw=width)
            ax.plot([mid,mid],[a,b],color=color,lw=width,ls='--')


def _format(ax, o, xlabel, ylabel, xlim=None, ylim=None):
    ax.set_xlabel(xlabel,fontsize=o['label_size'])
    ax.set_ylabel(ylabel,fontsize=o['label_size'])
    ax.tick_params(labelsize=o['tick_size'])
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)




def plot_sweep_columns(results, specs, options):
    """Figure 2: one column per study, ridge above risk with exact-zero strip."""
    o=options;keys=list(results)
    if not keys:raise ValueError('No computed studies selected')
    fig=plt.figure(figsize=(o['column_width']*len(keys),o['height']),dpi=o['dpi'],layout='constrained')
    gs=fig.add_gridspec(3,len(keys),height_ratios=[1,.18,1])
    for col,key in enumerate(keys):
        curves=results[key];spec=specs[key];p=o|spec.get('plot',{})
        top=fig.add_subplot(gs[0,col]);zero=fig.add_subplot(gs[1,col],sharex=top);bottom=fig.add_subplot(gs[2,col],sharex=top)
        colors=plt.get_cmap(p['cmap'])(np.linspace(.05,.9,len(curves)))
        positive=[]
        for j,(curve,color) in enumerate(zip(curves,colors)):
            opt=curve.optimum;r=opt.r_values
            coordinate='lambda' if p['coordinate']=='lambda' else 'mu'
            values=curve.regularization_values(coordinate)
            risk=curve.risk_values(p['normalization'])
            label=fr"${spec['symbol']}={curve.case.value:g}$"
            positive.extend(values[(values>0)&(opt.status!='upper')])
            yy=values.copy();yy[opt.status=='upper']=np.nan
            ridge_path(top,r,yy,color=color,width=p['linewidth'],style=p['jump_style'],ratio=p['jump_ratio'])
            zero.plot(r,np.where(opt.status=='zero',j,np.nan),color=color,lw=p['linewidth'])
            ridge_path(bottom,r,risk,color=color,width=p['linewidth'],style=p['risk_jump_style'],ratio=p['jump_ratio'],label=label)
        if p['ridge_ylim'] is not None:top.set_ylim(p['ridge_ylim'])
        elif positive:top.set_ylim(min(positive)/1.4,max(positive)*1.4)
        else:top.set_ylim(1e-6,1)
        for curve,color in zip(curves,colors):
            opt=curve.optimum;values=curve.regularization_values('lambda' if p['coordinate']=='lambda' else 'mu')
            upper=opt.status=='upper'
            if np.any(upper):top.plot(opt.r_values[upper],np.full(upper.sum(),top.get_ylim()[1]),'^',mfc='none',color=color,clip_on=False)
            # Continue positive-to-zero segments to the log panel's bottom edge.
            for j in range(len(values)-1):
                if (values[j]==0)!=(values[j+1]==0) and 'upper' not in opt.status[j:j+2]:
                    top.plot(opt.r_values[j:j+2],np.maximum(values[j:j+2],top.get_ylim()[0]),color=color,lw=p['linewidth'])
        top.set_yscale('log');top.set_xscale('log');bottom.set_xscale('log')
        zero.set(yticks=[],ylim=(-.5,len(curves)-.5));zero.tick_params(axis='x',labelbottom=False)
        zero.set_ylabel(r'$\lambda_{opt}=0$',rotation=0,ha='right',fontsize=p['label_size'])
        top.tick_params(axis='x',labelbottom=False)
        ylabel=r'Ridge $\lambda_{opt}$' if p['coordinate']=='lambda' else r'Effective $\lambda_{opt}$'
        _format(top,p,'',ylabel,p['r_xlim'],p['ridge_ylim'])
        risklabel={'R_infinity':r'$R_{opt}/R_\infty$','none':r'$R_{opt}$','sigma2':r'$R_{opt}/\sigma^2$'}[p['normalization']]
        _format(bottom,p,p['r_label'],risklabel,p['r_xlim'],p['risk_ylim'])
        if p['normalization']=='R_infinity':bottom.axhline(1,color='0.6',ls=':',lw=1)
        bottom.legend(fontsize=p['legend_size'],loc=p['legend_loc'])
        top.set_title(spec['title'],fontsize=p['title_size'])
    fig.suptitle(o['title'],fontsize=o['title_size'])
    return fig
