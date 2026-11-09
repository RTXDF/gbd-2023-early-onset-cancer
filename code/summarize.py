#!/usr/bin/env python3
"""Extract headline numbers from all results into results/headline.json + readable summary.
Units: PAF converted to PERCENT (API returns fraction); rates per 100,000; numbers are counts.
"""
import os, json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')

def rd(n):
    p = os.path.join(RES, n)
    if not os.path.exists(p) or os.path.getsize(p) == 0: return None
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None

def trip(r):
    return {'val': float(r.val), 'low': float(r.lower), 'high': float(r.upper)}

H = {}
# ---- global burden 1990 & 2023 (15-49), by sex ----
p = os.path.join(ROOT, 'data', 'panels', 'A_global_trends.pkl')
if os.path.exists(p):
    d = pd.read_pickle(p)
    d = d[(d.location == 1) & (d.cause == 1059) & (d.age == 24)]
    for sex, sl in [(3, 'both'), (1, 'male'), (2, 'female')]:
        for year in [1990, 2023]:
            dd = d[(d.year == year) & (d.sex == sex)]
            g = {}
            for measure, ml in [(6, 'incidence'), (1, 'deaths'), (2, 'dalys')]:
                for metric, mt in [(1, 'number'), (3, 'rate')]:
                    r = dd[(dd.measure == measure) & (dd.metric == metric)]
                    if len(r): g[f'{ml}_{mt}'] = trip(r.iloc[0])
            H[f'global_{year}_15_49_{sl}'] = g
# ---- global EAPC / AAPC ----
d = rd('trends_global_eapc.csv')
if d is not None:
    d = d[(d.age == '15-49') & (d.period == '1990-2023') & (d.sex == 'Both')]
    H['global_eapc_1990_2023'] = {r.measure: {'eapc': r.eapc, 'low': r.ci_low, 'high': r.ci_high} for r in d.itertuples()}
d = rd('trends_global_joinpoint.csv')
if d is not None:
    d = d[(d.age == '15-49') & (d.sex == 'Both') & d.n_joinpoints.notna()]
    H['global_aapc'] = {r.measure: {'aapc': r.aapc, 'low': r.aapc_ci_low, 'high': r.aapc_ci_high, 'jp': r.joinpoint_years} for r in d.itertuples()}
# ---- country EAPC extremes (15-49 ASR, both sexes) ----
d = rd('trends_country_eapc.csv')
if d is not None:
    d = d[(d.period == '1990-2023') & (d.sex == 'both')]
    ext = {}
    for m, g in d.groupby('measure'):
        g = g.dropna(subset=['eapc']).sort_values('eapc')
        if len(g) >= 5:
            ext[m] = {'top5_increase': g.tail(5)[['location', 'eapc', 'ci_low', 'ci_high']].to_dict('records'),
                      'top5_decrease': g.head(5)[['location', 'eapc', 'ci_low', 'ci_high']].to_dict('records'),
                      'n_countries': len(g)}
    H['country_eapc_extremes'] = ext
# ---- attributable global 1990 & 2023 (PAF -> percent) ----
d = rd('attributable_global.csv')
if d is not None:
    for year in [1990, 2023]:
        dd = d[(d.sex == 'both') & (d.age == '15-49') & (d.year == year)]
        a = {}
        for r in dd.itertuples():
            v = {'val': r.val, 'low': r.lower, 'high': r.upper}
            if r.metric == 'Percent':
                v = {k: x * 100 for k, x in v.items()}
            a[f'{r.risk}|{r.measure}|{r.metric}'] = v
        H[f'attributable_global_{year}'] = a
d = rd('attributable_trends_eapc.csv')
if d is not None:
    H['attributable_eapc'] = d.to_dict('records')
d = rd('attributable_country_2023.csv')
if d is not None:
    dd = d[(d.risk == 'Metabolic risks') & (d.measure == 'Deaths') & (d.metric == 'Percent')]
    if len(dd):
        dd = dd.dropna(subset=['val']).sort_values('val')
        H['paf_country_extremes_2023'] = {'highest': dd.tail(10).to_dict('records'), 'lowest': dd.head(5).to_dict('records')}
# ---- sites ----
d = rd('sites_global_2023.csv')
if d is not None:
    for m in d.measure.unique():
        dd = d[(d.measure == m) & (d.metric == 'Number')].sort_values('val', ascending=False)
        H[f'sites_top10_2023_{m}'] = dd.head(10).to_dict('records')
d = rd('sites_eapc.csv')
if d is not None:
    H['sites_eapc'] = d.to_dict('records')
d = rd('sites_attributable_2023.csv')
if d is not None and len(d):
    H['sites_attributable_2023'] = d.to_dict('records')
# ---- APC-IE ----
d = rd('apc_ie_effects.csv')
if d is not None and len(d):
    H['apc_ie'] = d.to_dict('records')
# ---- inequality ----
d = rd('inequality_indices.csv')
if d is not None:
    H['inequality'] = d.to_dict('records')
# ---- decomposition global ----
d = rd('decomposition_global.csv')
if d is not None:
    H['decomposition_global'] = d.to_dict('records')
# ---- projections 2050 (global + SDI quintiles) ----
d = rd('projections_rate.csv')
if d is not None:
    H['projection_2050_rate'] = d[d.year == 2050].to_dict('records')
d = rd('projections_number.csv')
if d is not None:
    H['projection_2050_number'] = d[d.year == 2050].to_dict('records')
d = rd('projections_attributable.csv')
if d is not None and len(d):
    H['projection_2050_attributable'] = d[d.year == 2050].to_dict('records')

json.dump(H, open(os.path.join(RES, 'headline.json'), 'w'), indent=2, default=float)
print(f"headline keys: {list(H.keys())}")
print(json.dumps({k: H[k] for k in list(H)[:3]}, indent=2, default=float)[:2500])