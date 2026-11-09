#!/usr/bin/env python3
"""Das Gupta symmetric decomposition of change in (attributable) early-onset cancer burden.

General exact decomposition of N = prod(factors) via average over all factor orderings.
Application: attributable deaths D = Pop(15-49) x Total_cancer_rate x PAF.
  Effects: population growth, baseline rate change, risk-exposure (PAF) change.
Also decomposes total EO cancer deaths D = Pop x Rate into population vs rate effects.
Outputs (results/): decomposition_global.csv, decomposition_country.csv
"""
import os, json, itertools
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results'); os.makedirs(RES, exist_ok=True)
idmap = json.load(open(os.path.join(ROOT, 'data', 'id_map.json')))
COUNTRIES = set(int(k) for k in idmap['countries'])

def dasgupta(f0: dict, f1: dict):
    """Exact symmetric decomposition of prod(f1)-prod(f0). Returns effect per factor."""
    keys = list(f0.keys())
    k = len(keys)
    effects = {key: 0.0 for key in keys}
    perms = list(itertools.permutations(keys))
    for order in perms:
        cur = dict(f0)
        for key in order:
            before = np.prod([cur[x] for x in keys])
            cur[key] = f1[key]
            after = np.prod([cur[x] for x in keys])
            effects[key] += (after - before)
    for key in keys:
        effects[key] /= len(perms)
    return effects

def _get(df, location, measure, metric, age, sex, year, col='val'):
    g = df[(df.location == location) & (df.measure == measure) & (df.metric == metric) &
           (df.age == age) & (df.sex == sex) & (df.year == year)]
    if len(g) == 0: return None
    return float(g[col].iloc[0])

def run():
    # population panel (15-49 age=24, measure 44, metric 1)
    popf = os.path.join(PANELS, 'L_pop_sex3.pkl')
    causef = os.path.join(PANELS, 'BCD_country_eo_both.pkl')
    riskf = os.path.join(PANELS, 'GHI_risk104_country_both.pkl')
    if not all(os.path.exists(p) for p in [popf, causef, riskf]):
        print('Required panels not ready (pop/cause/risk).'); return
    pop = pd.read_pickle(popf)
    cause = pd.read_pickle(causef)
    risk = pd.read_pickle(riskf)
    # population: age 24 (15-49). cause total deaths rate: age 24 metric 3 measure 1.
    # PAF: risk measure 1 (deaths) metric 2 (percent) age 24 -> API returns a FRACTION (0.039 = 3.9%)
    rows = []
    locs = [1] + sorted(COUNTRIES)
    for loc in locs:
        for (y0, y1) in [(1990, 2023), (2000, 2023), (2010, 2023)]:
            P0 = _get(pop, loc, 44, 1, 24, 3, y0); P1 = _get(pop, loc, 44, 1, 24, 3, y1)
            R0 = _get(cause, loc, 1, 3, 24, 3, y0); R1 = _get(cause, loc, 1, 3, 24, 3, y1)
            F0 = _get(risk, loc, 1, 2, 24, 3, y0); F1 = _get(risk, loc, 1, 2, 24, 3, y1)
            if None in (P0, P1, R0, R1): continue
            # total deaths decomposition (Pop x Rate); rate per 100k -> /1e5
            f0 = {'population': P0, 'rate': R0 / 1e5}
            f1 = {'population': P1, 'rate': R1 / 1e5}
            eff = dasgupta(f0, f1)
            name = idmap['locations'].get(str(loc), str(loc))
            rows.append({'location_id': loc, 'location': name, 'period': f'{y0}-{y1}', 'outcome': 'total_deaths',
                         'value_0': P0*R0/1e5, 'value_1': P1*R1/1e5,
                         'population_effect': eff['population'], 'rate_effect': eff['rate'],
                         'attributable': False})
            if None not in (F0, F1):
                a0 = {'population': P0, 'rate': R0/1e5, 'paf': F0}
                a1 = {'population': P1, 'rate': R1/1e5, 'paf': F1}
                ae = dasgupta(a0, a1)
                rows.append({'location_id': loc, 'location': name, 'period': f'{y0}-{y1}', 'outcome': 'attributable_deaths',
                             'value_0': P0*R0/1e5*F0, 'value_1': P1*R1/1e5*F1,
                             'population_effect': ae['population'], 'rate_effect': ae['rate'], 'paf_effect': ae['paf'],
                             'attributable': True})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'decomposition_country.csv'), index=False)
    g = out[out.location_id == 1]
    g.to_csv(os.path.join(RES, 'decomposition_global.csv'), index=False)
    print(f"decomposition rows: {len(out)} (global {len(g)})")
    print(g[g.period == '1990-2023'].to_string(index=False))

if __name__ == '__main__':
    run()