#!/usr/bin/env python3
"""Temporal trend analysis: EAPC + joinpoint (BIC-selected segmented log-linear) regression.

Outputs (results/):
  trends_global_eapc.csv      global EAPC by measure/sex/age, full & recent period
  trends_global_joinpoint.csv global joinpoint segments + APC/AAPC
  trends_country_eapc.csv     country-level EAPC (ASR incidence/deaths/DALYs, both sexes)
Methods:
  EAPC: ln(rate)=a+b*year ; EAPC=(exp(b)-1)*100 ; 95% CI via b±1.96*SE -> (exp(.)-1)*100
  Joinpoint: continuous piecewise ln-linear, hinges (t-tau)+; grid-search tau; BIC selects k (0..3).
             APC_segment=(exp(slope)-1)*100 ; AAPC=(exp(sum w_i*slope_i)-1)*100, w_i=segment years/T.
"""
import os, json, itertools, math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)

Z = 1.959963985

def load_panel(name):
    return pd.read_pickle(os.path.join(PANELS, name + '.pkl'))

def eapc(years, rates, start=None, end=None):
    """Return dict EAPC + CI for ln-linear fit over [start,end]."""
    y = np.asarray(years, float); r = np.asarray(rates, float)
    m = np.isfinite(r) & (r > 0)
    if start is not None: m &= (y >= start)
    if end is not None: m &= (y <= end)
    y, r = y[m], r[m]
    if len(y) < 3: return None
    ly = np.log(r)
    X = sm.add_constant(y - y.mean())
    fit = sm.OLS(ly, X).fit()
    b, se = fit.params[1], fit.bse[1]
    return {
        'n_years': int(len(y)), 'year_start': int(y.min()), 'year_end': int(y.max()),
        'eapc': (math.exp(b) - 1) * 100,
        'ci_low': (math.exp(b - Z * se) - 1) * 100,
        'ci_high': (math.exp(b + Z * se) - 1) * 100,
        'pvalue': float(fit.pvalues[1]),
    }

def _seg_design(t, taus):
    cols = [np.ones_like(t), t]
    for tau in taus:
        cols.append(np.maximum(t - tau, 0.0))
    return np.column_stack(cols)

def joinpoint(years, rates, max_k=3, min_seg=3):
    """BIC-selected continuous segmented log-linear regression. Returns best model dict."""
    y = np.asarray(years, float); r = np.asarray(rates, float)
    m = np.isfinite(r) & (r > 0)
    y, r = y[m], r[m]
    if len(y) < 6: return None
    t = y - y.min(); T = len(t); ly = np.log(r)
    # candidate interior joinpoint indices (ensure min_seg spacing)
    cand = list(range(min_seg, T - min_seg + 1))
    best = None
    for k in range(0, max_k + 1):
        if k == 0:
            combos = [()]
        else:
            combos = [c for c in itertools.combinations(cand, k)
                      if all(c[i+1] - c[i] >= min_seg for i in range(k-1))]
        for taus_idx in combos:
            taus = [t[i] for i in taus_idx]
            X = _seg_design(t, taus)
            try:
                fit = sm.OLS(ly, X).fit()
            except Exception:
                continue
            rss = float(np.sum(fit.resid ** 2))
            npar = X.shape[1]
            bic = T * math.log(rss / T) + npar * math.log(T)
            if best is None or bic < best['bic']:
                best = {'bic': bic, 'k': k, 'taus_idx': taus_idx, 'fit': fit, 't': t, 'T': T, 'y': y}
    if best is None: return None
    fit = best['fit']; t = best['t']; T = best['T']; y = best['y']
    taus_idx = best['taus_idx']
    # segment boundaries in index space
    bounds = [0] + list(taus_idx) + [T]
    # slope per segment = base slope + sum of hinge slopes active in that segment
    base_slope = fit.params[1]
    hinge_slopes = list(fit.params[2:])
    segments = []
    slopes = []
    weights = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i+1]
        slope = base_slope + sum(hinge_slopes[:i])  # hinges before this segment active
        seg_years = int(hi - lo)
        # APC CI via delta method using covariance of (base + relevant hinges)
        idx = [1] + list(range(2, 2 + i))
        cov = np.asarray(fit.cov_params())[np.ix_(idx, idx)]
        var = float(np.sum(cov))
        se = math.sqrt(max(var, 0))
        segments.append({
            'seg': i + 1, 'year_start': int(y[lo]), 'year_end': int(y[hi-1]),
            'apc': (math.exp(slope) - 1) * 100,
            'ci_low': (math.exp(slope - Z * se) - 1) * 100,
            'ci_high': (math.exp(slope + Z * se) - 1) * 100,
        })
        slopes.append(slope); weights.append(seg_years / T)
    aapc_slope = float(np.dot(slopes, weights))
    # AAPC SE via weighted covariance
    idx_all = [1] + list(range(2, 2 + len(hinge_slopes)))
    # build weight vector over params for each segment slope, then AAPC linear combo
    wvec = np.zeros(len(fit.params))
    for i in range(len(bounds) - 1):
        w = weights[i]
        wvec[1] += w
        for j in range(i):
            wvec[2 + j] += w
    cov = np.asarray(fit.cov_params())
    aapc_var = float(wvec @ cov @ wvec)
    aapc_se = math.sqrt(max(aapc_var, 0))
    return {
        'n_joinpoints': best['k'], 'bic': best['bic'],
        'joinpoint_years': [int(y[i]) for i in taus_idx],
        'aapc': (math.exp(aapc_slope) - 1) * 100,
        'aapc_ci_low': (math.exp(aapc_slope - Z * aapc_se) - 1) * 100,
        'aapc_ci_high': (math.exp(aapc_slope + Z * aapc_se) - 1) * 100,
        'segments': segments,
    }

MEASURES = {1: 'Deaths', 2: 'DALYs', 6: 'Incidence'}

def run_global():
    df = load_panel('A_global_trends')
    df = df[(df.location == 1) & (df.cause == 1059) & (df.metric == 3)]
    eapc_rows, jp_rows = [], []
    for (measure, sex, age), g in df.groupby(['measure', 'sex', 'age']):
        g = g.sort_values('year')
        if age not in (24, 27, 22):  # 15-49, age-standardized, all ages
            continue
        for (start, end, label) in [(1990, 2023, '1990-2023'), (2013, 2023, '2013-2023'), (1990, 2009, '1990-2009')]:
            e = eapc(g.year.values, g.val.values, start, end)
            if e:
                eapc_rows.append({'measure': MEASURES.get(measure, measure), 'sex': g.sex_name.iloc[0],
                                  'age': g.age_name.iloc[0], 'period': label, **e})
        if age in (24, 27):
            jp = joinpoint(g.year.values, g.val.values)
            if jp:
                base = {'measure': MEASURES.get(measure, measure), 'sex': g.sex_name.iloc[0], 'age': g.age_name.iloc[0]}
                jp_rows.append({**base, 'n_joinpoints': jp['n_joinpoints'], 'joinpoint_years': ';'.join(map(str, jp['joinpoint_years'])),
                                'aapc': jp['aapc'], 'aapc_ci_low': jp['aapc_ci_low'], 'aapc_ci_high': jp['aapc_ci_high'], 'bic': jp['bic']})
                for s in jp['segments']:
                    jp_rows.append({**base, 'segment': s['seg'], 'year_start': s['year_start'], 'year_end': s['year_end'],
                                    'apc': s['apc'], 'apc_ci_low': s['ci_low'], 'apc_ci_high': s['ci_high']})
    pd.DataFrame(eapc_rows).to_csv(os.path.join(RES, 'trends_global_eapc.csv'), index=False)
    pd.DataFrame(jp_rows).to_csv(os.path.join(RES, 'trends_global_joinpoint.csv'), index=False)
    print(f"global EAPC rows: {len(eapc_rows)}; joinpoint rows: {len(jp_rows)}")

def run_country():
    frames = []
    for sex in ['both', 'male', 'female']:
        p = os.path.join(PANELS, f'BCD_country_eo_{sex}.pkl')
        if os.path.exists(p):
            d = pd.read_pickle(p); d['sex_query'] = sex; frames.append(d)
    if not frames:
        print('no country panels yet'); return
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.metric == 3) & (df.age == 24) & (df.location.isin(
        [int(k) for k in json.load(open(os.path.join(ROOT, 'data', 'id_map.json')))['countries']]))]
    rows = []
    for (loc, measure, sexq), g in df.groupby(['location', 'measure', 'sex_query']):
        g = g.sort_values('year')
        for (start, end, label) in [(1990, 2023, '1990-2023'), (2013, 2023, '2013-2023')]:
            e = eapc(g.year.values, g.val.values, start, end)
            if e:
                rows.append({'location_id': int(loc), 'location': g.location_name.iloc[0],
                             'measure': MEASURES.get(measure, measure), 'sex': sexq, 'period': label, **e})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'trends_country_eapc.csv'), index=False)
    print(f"country EAPC rows: {len(out)}")

if __name__ == '__main__':
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all', 'global'): run_global()
    if what in ('all', 'country'): run_country()