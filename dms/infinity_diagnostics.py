"""Exact rational risk-gap certificates for a fixed, decimalized DMS config.

The coefficient test is sufficient, not necessary. It does not validate the
statistical derivation or certify a continuous interval of aspect ratios.
"""
from fractions import Fraction as Q
from math import comb
import numpy as np


def _add(*ps):
    out = [Q(0)] * max(map(len, ps))
    for p in ps:
        for i, v in enumerate(p):
            out[i] += v
    return out


def _scale(p, a):
    return [a*v for v in p]


def _mul(a, b):
    out = [Q(0)] * (len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def _shift(p, a):
    return [sum((p[j]*comb(j, i)*a**(j-i) for j in range(i, len(p))), Q(0))
            for i in range(len(p))]


def infinity_gap_certificate(cfg, r, *, bracket_steps=100):
    """Build R-Rinf=N(k)/D(k) with exact Fraction arithmetic.

    Floats are interpreted as their printed decimal values, including the raw
    means already converted by exploration.py. No sign tolerances are used.
    Positive coefficients after shifting by a rigorous lower bound on kappa0
    certify the whole physical half-line. Failure is only 'inconclusive'.
    """
    cfg.validate_theory()
    if not np.isfinite(r) or r <= 0 or not 0 < cfg.f_train < 1:
        raise ValueError('Require r>0 and 0<f_train<1')
    if bracket_steps < 1:
        raise ValueError('bracket_steps must be positive')
    v = {k: Q(str(x)) for k, x in cfg.as_dict().items()}
    q, f, ft, eta = (v[k] for k in ('q','f_train','f_test','eta'))
    g, F, Ft, Fn, C = (v[k] for k in ('signal_norm','F','F_test','F_neq','C_neq'))
    Mt, Mn, Mnt = (v[k] for k in ('M_tot','M_neq','M_neq_test'))
    rr, qb, t = Q(str(r)), 1-1/(q-1)**2, q*(1-f)
    qr = q/(q-1)
    cm = (q-2-eta*qb+q*(1-ft)*eta*qb)/q
    cw = (q*(1-ft)*(1-eta*qb)+eta*qb)/q
    dm = eta*(q-1)/q*qb*(1-q*ft/(q-1))**2/q
    dw = (qr*(1-eta)*(f-ft)**2+qr*eta*(1-f-ft/(q-1))**2)/q
    cm += (q-1)/ft*dm
    cw += (q-1)/ft*dw
    g000 = g*g*(F-qb*Fn)
    g001 = g*g*F-g000
    g00 = g*g*(F-Fn-C/(q-1))
    g010, g011 = g00-g000, -(g00-g000)
    g110 = g*g*Ft+g000-2*g00
    g111 = g*g*(1-F)-g110
    bw, bm = q*(1-ft)*g000+g001, g111+q*(1-ft)*g110
    cross = g011+q*(1-ft)*g010
    sw = g*qr*(-(f-ft)*(Mt-Mn)+Mn*((f-ft)/(q-1)+1-q*f/(q-1)))
    sm = -(1-q*ft/(q-1))*(Mnt+Mn/(q-1))*g
    # U=(k+1), W=(k+t); all polynomial arrays are ascending powers.
    U, W = [Q(1),Q(1)], [t,Q(1)]
    U2, W2 = _mul(U,U), _mul(W,W)
    H = _mul(U2,W2)
    A = _scale(_add(_scale(W2,q-2),_scale(U2,t*t)),rr/q)
    J = _add(H,_scale(A,-1))  # H*(1-a1)
    T = _scale(_add(_scale(W2,cm),_scale(U2,t*cw)),rr)
    B = _scale(_add(_scale(W2,1-F),_scale(U2,F*t)),g*g)
    shift = _add(_scale(U,sw),_scale(W,sm))
    E = _add(_scale(U2,bw),_scale(W2,bm),_scale(_mul(U,W),2*cross),
             _scale(_mul(shift,shift),(q-1)/ft))
    c, noise = ft/(q-1), v['sigma']**2*ft/f
    D = _mul(H,J)
    risk_num = _add(_scale(_mul([Q(0),Q(0),Q(1)],
                              _add(_mul(E,J),_mul(B,T))),c),
                    _scale(_mul(T,H),noise))
    mean_shift = qr*(ft-f)*Mt+(1-q*ft/(q-1))*(Mn-Mnt)
    rinf = g*g*ft/(q-1)*(1+(q*(1-ft)-1)*Ft)+g*g*mean_shift**2
    N = _add(risk_num,_scale(D,-rinf))
    if N[-1] != 0:
        raise ArithmeticError('Exact leading cancellation failed: audit R_infinity')
    b, cc = 1+t-rr*(q-2+t)/q, t*(1-rr*(q-1)/q)
    lo = hi = Q(0)
    if cc < 0:
        hi = Q(1)
        while hi*hi+b*hi+cc < 0:
            hi *= 2
        for _ in range(bracket_steps):
            mid = (lo+hi)/2
            if mid*mid+b*mid+cc <= 0:
                lo = mid
            else:
                hi = mid
    shifted = _shift(N,lo)
    proven = all(a >= 0 for a in shifted)
    # D>0 for k>kappa0; at interpolation kappa0=0 is a singular limit.
    # a1 is decreasing, and at x=0 a1 < r E[t/(t+k)]=1 for k>0.
    strict = proven and shifted[0] > 0
    tail = N[-2]/D[-1]*f/(q-1)  # R-Rinf = tail/lambda + O(lambda^-2)
    return dict(status='certified_nonnegative' if proven else 'inconclusive',
                strictly_positive= strict, r=float(rr), numerator=N, denominator=D,
                shifted_coefficients=shifted, kappa0_bracket=(lo,hi),
                R_infinity_exact=rinf, tail_coefficient=tail,
                singular_zero=(rr*(q-1)==q))


def stable_risk_gap(certificate, kappas):
    """Evaluate the already-cancelled numerator, avoiding subtraction of risks.

    Polynomial construction is exact; this evaluation is ordinary float64.
    """
    k = np.asarray(kappas, dtype=float)
    n = np.array([float(x) for x in certificate['numerator']])
    d = np.array([float(x) for x in certificate['denominator']])
    return np.polynomial.polynomial.polyval(k,n)/np.polynomial.polynomial.polyval(k,d)
