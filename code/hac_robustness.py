#!/usr/bin/env python3
"""HAC (Newey-West) robust uncertainty for trend estimates.

Refits every log-linear trend used in the manuscript with heteroskedasticity- and
autocorrelation-consistent (HAC) covariance and compares it with the conventional
OLS interval:

  * global directly standardised rates (incidence, deaths, DALYs), 1990-2023 and
    2013-2023;
  * sex-specific directly standardised rates;
  * site-specific GBD age-standardised rates (29 sites, incidence and deaths);
  * annual cross-national RII series (incidence, deaths, metabolic PAF).

The estimand is unchanged (slope of ln rate on year, or ln RII on year); only the
standard error changes.  Maximum lag follows the Newey-West rule of thumb
floor(4*(T/100)^(2/9)).

Outputs (results/):
  hac_comparison.csv   conventional vs HAC intervals for every series
  hac_summary.json     counts of changed conclusions and SE inflation
"""
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
PANELS = os.path.join(ROOT, 'data', 'panels')
Z = 1.959963985


def nw_lags(n):
    return max(1, int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def fit_trend(years, values, log_scale=True, start=None, end=None, maxlags=None):
    y = np.asarray(years, float)
    v = np.asarray(values, float)
    m = np.isfinite(v) & (v > 0) if log_scale else np.isfinite(v)
    if start is not None:
        m &= y >= start
    if end is not None:
        m &= y <= end
    y, v = y[m], v[m]
    if len(y) < 5:
        return None
    t = y - y.mean()
    X = sm.add_constant(t)
    dep = np.log(v) if log_scale else v
    ols = sm.OLS(dep, X).fit()
    lags = nw_lags(len(y)) if maxlags is None else maxlags
    hac = sm.OLS(dep, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    b0, b1 = float(ols.params[1]), float(hac.params[1])
    se0, se1 = float(ols.bse[1]), float(hac.bse[1])
    p0 = 2 * (1 - stats.norm.cdf(abs(b0 / se0))) if se0 > 0 else np.nan
    p1 = 2 * (1 - stats.norm.cdf(abs(b1 / se1))) if se1 > 0 else np.nan
    conv = (math.exp(b0) - 1) * 100 if log_scale else b0
    hac_e = (math.exp(b1) - 1) * 100 if log_scale else b1
    # transform the interval with the matching point estimate
    conv_lo = (math.exp(b0 - Z * se0) - 1) * 100 if log_scale else b0 - Z * se0
    conv_hi = (math.exp(b0 + Z * se0) - 1) * 100 if log_scale else b0 + Z * se0
    hac_lo = (math.exp(b1 - Z * se1) - 1) * 100 if log_scale else b1 - Z * se1
    hac_hi = (math.exp(b1 + Z * se1) - 1) * 100 if log_scale else b1 + Z * se1
    return {
        'n_years': int(len(y)), 'year_start': int(y.min()), 'year_end': int(y.max()),
        'maxlags': lags,
        'estimate_conv': conv, 'ci_low_conv': conv_lo, 'ci_high_conv': conv_hi,
        'p_conv': float(p0),
        'estimate_hac': hac_e, 'ci_low_hac': hac_lo, 'ci_high_hac': hac_hi,
        'p_hac': float(p1),
        'se_conv': se0, 'se_hac': se1,
        'se_ratio': se1 / se0 if se0 > 0 else np.nan,
        'sig_conv': bool(p0 < 0.05), 'sig_hac': bool(p1 < 0.05),
    }


def run():
    rows = []

    def add(analysis, series, measure, period, years, values, log_scale=True):
        for (start, end, tag) in period:
            r = fit_trend(years, values, log_scale=log_scale, start=start, end=end)
            if r is None:
                continue
            rows.append({'analysis': analysis, 'series': series, 'measure': measure,
                         'period': tag, **r})

    # 1. global directly standardised rates
    g = pd.read_csv(os.path.join(RES, 'global_asr_series.csv'))
    periods = [(1990, 2023, '1990-2023'), (2013, 2023, '2013-2023')]
    for meas in ('Incidence', 'Deaths', 'DALYs'):
        d = g[g.measure == meas].sort_values('year')
        add('Global ASR', 'Both sexes', meas, periods, d.year, d.asr_15_49)

    # 2. sex-specific ASR
    s = pd.read_csv(os.path.join(RES, 'sex_asr_series.csv'))
    for (sex, meas), d in s.groupby(['sex', 'measure']):
        d = d.sort_values('year')
        add('Sex-specific ASR', sex, meas, periods, d.year, d.asr_15_49)

    # 3. site-specific GBD ASR (incidence and deaths)
    for meas, panel in ((6, 'F_sites_m6_a24'), (1, 'F_sites_m1_a24')):
        p = pd.read_pickle(os.path.join(PANELS, panel + '.pkl'))
        p = p[(p.location == 1) & (p.age == 24) & (p.metric == 3) & (p.sex == 3)]
        for cause, d in p.groupby('cause_name'):
            d = d.sort_values('year')
            add('Site ASR', cause, 'Incidence' if meas == 6 else 'Deaths',
                [(1990, 2023, '1990-2023')], d.year, d.val)

    # 4. RII trends (log scale; estimate = annual % change in RII)
    it = pd.read_csv(os.path.join(RES, 'inequality_trends.csv'))
    for outcome, d in it.groupby('outcome'):
        d = d.sort_values('year')
        add('RII trend', outcome, 'RII', periods, d.year, d.rii)

    out = pd.DataFrame(rows)
    out['direction_agree'] = np.sign(out.estimate_conv) == np.sign(out.estimate_hac)
    out.to_csv(os.path.join(RES, 'hac_comparison.csv'), index=False)

    # lag-length sensitivity for the headline series
    lag_rows = []
    headline = [
        ('Global ASR', 'Both sexes', 'Incidence', g[g.measure == 'Incidence']),
        ('Global ASR', 'Both sexes', 'Deaths', g[g.measure == 'Deaths']),
        ('Global ASR', 'Both sexes', 'DALYs', g[g.measure == 'DALYs']),
    ]
    for outcome, d in it.groupby('outcome'):
        if outcome in ('Total Incidence (15-49 ASR)', 'Total Deaths (15-49 ASR)'):
            headline.append(('RII trend', outcome, 'RII', d))
    for analysis, series, measure, d in headline:
        d = d.sort_values('year')
        for start, end, tag in periods:
            for lag in (1, 2, 3, 4, 5):
                r = fit_trend(d.year, d.asr_15_49 if 'asr_15_49' in d else d.rii,
                              start=start, end=end, maxlags=lag)
                if r is None:
                    continue
                lag_rows.append({'analysis': analysis, 'series': series,
                                 'measure': measure, 'period': tag, 'maxlags': lag,
                                 'estimate': r['estimate_hac'],
                                 'ci_low': r['ci_low_hac'],
                                 'ci_high': r['ci_high_hac'], 'p': r['p_hac'],
                                 'sig': r['sig_hac']})
    pd.DataFrame(lag_rows).to_csv(os.path.join(RES, 'hac_lag_sensitivity.csv'),
                                  index=False)

    changed = out[out.sig_conv != out.sig_hac]
    summary = {
        'n_series': int(len(out)),
        'n_significance_changed': int(len(changed)),
        'changed_series': changed[['analysis', 'series', 'measure', 'period',
                                   'p_conv', 'p_hac']].to_dict(orient='records'),
        'se_ratio_median': float(out.se_ratio.median()),
        'se_ratio_p95': float(out.se_ratio.quantile(0.95)),
        'se_ratio_max': float(out.se_ratio.max()),
        'max_se_ratio_series': out.loc[out.se_ratio.idxmax(),
                                       ['analysis', 'series', 'measure', 'period']].to_dict(),
        'direction_disagreements': int((~out.direction_agree).sum()),
    }
    json.dump(summary, open(os.path.join(RES, 'hac_summary.json'), 'w'), indent=2)
    print(f'hac_comparison.csv: {len(out)} series; '
          f'{len(changed)} significance changes; median SE ratio '
          f'{summary["se_ratio_median"]:.3f}; max {summary["se_ratio_max"]:.3f}')
    if len(changed):
        print(changed[['analysis', 'series', 'measure', 'period', 'p_conv', 'p_hac']]
              .to_string(index=False))
    return out


if __name__ == '__main__':
    run()
