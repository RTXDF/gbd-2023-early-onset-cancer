#!/usr/bin/env python3
"""Directly age-standardised rates for countries, GBD regions and SDI quintiles.

Country-level EAPCs in trends.py use GBD crude rates (metric 3), which mix
age composition with risk.  The E panels contain age-specific numbers for all
204 countries (sex 3, ages 8-14), so the same 2023 global 15-49 standard
population used for the global series can be applied everywhere.

Outputs (results/):
  country_asr_series.csv          country x year x measure ASR + counts
  country_asr_trends.csv          country EAPC 1990-2023 / 2013-2023 + BH q
  regions_asr_2023.csv            region ASR, counts, PAF context
  regions_asr_trends.csv          region EAPC
  sdi_quintile_asr_series.csv     SDI quintile (2023 assignment) ASR series
  sdi_quintile_asr_trends.csv     SDI quintile EAPC
"""
import os, math
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trends import eapc, joinpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
AGES = [8, 9, 10, 11, 12, 13, 14]
MEASURES = {6: 'Incidence', 1: 'Deaths', 2: 'DALYs'}

REGION_IDS = [5, 9, 21, 32, 42, 56, 65, 70, 73, 96, 100, 104, 120, 124, 134,
              138, 159, 167, 174, 192, 199]

# Territories appended after the main region blocks in the GBD location list;
# the sequential rule would misassign them, so map them explicitly.
REGION_OVERRIDES = {
    298: 21, 305: 100, 320: 21, 349: 100, 351: 21, 367: 73, 369: 21,
    374: 21, 376: 21, 380: 21, 385: 100, 393: 104, 396: 73, 413: 21,
    416: 21, 422: 104, 435: 174, 522: 138,
}


def region_map():
    """Infer the country -> GBD region mapping from the contiguous location IDs
    in the BCD panel (each region row is followed by its member countries)."""
    b = pd.read_pickle(os.path.join(PANELS, 'BCD_country_eo_both.pkl'))
    loc = b[['location', 'location_name']].drop_duplicates().sort_values('location')
    loc = loc[loc.location < 44634]
    out, current = {}, None
    for lid, name in zip(loc.location, loc.location_name):
        if lid in REGION_IDS:
            current = (lid, name)
        elif lid != 1 and current is not None:
            out[lid] = current
    names = dict(zip(loc.location, loc.location_name))
    for lid, rid in REGION_OVERRIDES.items():
        if lid in names:
            out[lid] = (rid, names[rid])
    return out


def weights(pop):
    w = pop[(pop.location == 1) & (pop.year == 2023) & (pop.age.isin(AGES)) & (pop.metric == 1)]
    w = w.set_index('age').val
    return w / w.sum()


def series_for(num, pop, loc_ids, label_map, w):
    """ASR series for a set of locations (aggregating members where needed)."""
    frames = []
    for meas, mlabel in MEASURES.items():
        d = num[meas]
        d = d[(d.metric == 1) & (d.sex == 3) & (d.age.isin(AGES)) & (d.location.isin(loc_ids))]
        p = pop[(pop.metric == 1) & (pop.sex == 3) & (pop.age.isin(AGES)) & (pop.location.isin(loc_ids))]
        for lid, lname in label_map.items():
            dd = d[d.location == lid]
            pp = p[p.location == lid]
            for yr in sorted(dd.year.unique()):
                rates, ns, ps = {}, 0.0, 0.0
                for a in AGES:
                    n = dd[(dd.year == yr) & (dd.age == a)].val.sum()
                    q = pp[(pp.year == yr) & (pp.age == a)].val.sum()
                    rates[a] = n / q if q > 0 else np.nan
                    ns += n
                    ps += q
                frames.append({'location_id': lid, 'location': lname, 'measure': mlabel,
                               'year': int(yr), 'count': ns, 'population': ps,
                               'crude_rate': ns / ps * 1e5 if ps > 0 else np.nan,
                               'asr': sum(w[a] * rates[a] for a in AGES) * 1e5})
    return pd.DataFrame(frames)


def country_series(num, pop, w):
    import json
    with open(os.path.join(ROOT, 'data', 'id_map.json')) as fh:
        names = {int(k): v for k, v in json.load(fh)['countries'].items()}
    return series_for(num, pop, set(names), names, w)


def aggregate_series(cs, mapping, key_name):
    out = []
    for key, members in mapping.items():
        sub = cs[cs.location_id.isin(members)]
        for (meas, yr), g in sub.groupby(['measure', 'year']):
            n = g['count'].sum()
            p = g['population'].sum()
            out.append({'group': key, 'measure': meas, 'year': int(yr),
                        'count': n, 'population': p, 'crude_rate': n / p * 1e5,
                        'asr': float((g['asr'] * g['population']).sum() / p)})
    return pd.DataFrame(out).rename(columns={'group': key_name})


def trend_table(s, group_col):
    rows = []
    for (grp, meas), g in s.groupby([group_col, 'measure']):
        g = g.sort_values('year')
        rec = {group_col: grp, 'measure': meas,
               'asr_1990': g.asr.iloc[0], 'asr_2023': g.asr.iloc[-1],
               'asr_change_percent': (g.asr.iloc[-1] / g.asr.iloc[0] - 1) * 100}
        for start, end, tag in [(1990, 2023, 'eapc_1990_2023'), (2013, 2023, 'eapc_2013_2023')]:
            e = eapc(g.year.values, g.asr.values, start, end)
            if e:
                rec[tag] = e['eapc']; rec[tag + '_low'] = e['ci_low']
                rec[tag + '_high'] = e['ci_high']; rec[tag + '_p'] = e['pvalue']
        jp = joinpoint(g.year.values, g.asr.values, max_k=3)
        if jp:
            rec['joinpoints'] = ';'.join(map(str, jp['joinpoint_years']))
            rec['last_apc'] = jp['segments'][-1]['apc']
            rec['last_segment'] = f"{jp['segments'][-1]['year_start']}-{jp['segments'][-1]['year_end']}"
        rows.append(rec)
    out = pd.DataFrame(rows)
    if group_col == 'location':
        for meas in out.measure.unique():
            m = out.measure == meas
            for tag in ['eapc_1990_2023_p', 'eapc_2013_2023_p']:
                ok = m & out[tag].notna()
                out.loc[ok, tag.replace('_p', '_q')] = np.nan
                out.loc[ok, tag.replace('_p', '_q')] = multipletests(out.loc[ok, tag], method='fdr_bh')[1]
    return out


def main():
    num = {m: pd.read_pickle(os.path.join(PANELS, f'E_country_age_m{m}.pkl')) for m in MEASURES}
    pop = pd.read_pickle(os.path.join(PANELS, 'L_pop_sex3.pkl'))
    w = weights(pop)

    cs = country_series(num, pop, w)
    cs.to_csv(os.path.join(RES, 'country_asr_series.csv'), index=False)
    ct = trend_table(cs, 'location')
    ct.to_csv(os.path.join(RES, 'country_asr_trends.csv'), index=False)

    rm = region_map()
    reg = aggregate_series(cs, {v[1]: [k for k, vv in rm.items() if vv == v] for v in set(rm.values())}, 'region')
    reg.to_csv(os.path.join(RES, 'regions_asr_series.csv'), index=False)
    rt = trend_table(reg, 'region')
    rt.to_csv(os.path.join(RES, 'regions_asr_trends.csv'), index=False)
    r23 = reg[reg.year == 2023][['region', 'measure', 'count', 'population', 'crude_rate', 'asr']]
    r23.to_csv(os.path.join(RES, 'regions_asr_2023.csv'), index=False)

    ineq = pd.read_csv(os.path.join(RES, 'inequality_country_data.csv'))
    sdi23 = ineq[(ineq.year == 2023) & (ineq.outcome == 'Total Incidence (15-49 ASR)')][['location_id', 'sdi']]
    sdi23['quintile'] = pd.qcut(sdi23.sdi, 5, labels=['Q1 lowest', 'Q2', 'Q3', 'Q4', 'Q5 highest'])
    qmap = sdi23.groupby('quintile', observed=True).location_id.apply(list).to_dict()
    qs = aggregate_series(cs, qmap, 'sdi_quintile')
    qs.to_csv(os.path.join(RES, 'sdi_quintile_asr_series.csv'), index=False)
    qt = trend_table(qs, 'sdi_quintile')
    qt.to_csv(os.path.join(RES, 'sdi_quintile_asr_trends.csv'), index=False)

    print('countries:', ct.location.nunique(), 'regions:', rt.region.nunique(),
          'quintiles:', qt.sdi_quintile.nunique())
    print('\nIncidence 2023 ASR by region:')
    print(r23[r23.measure == 'Incidence'].sort_values('asr', ascending=False)
          [['region', 'asr', 'crude_rate']].to_string(index=False))
    print('\nIncidence EAPC extremes (1990-2023):')
    i = ct[ct.measure == 'Incidence'].sort_values('eapc_1990_2023', ascending=False)
    print(i[['location', 'eapc_1990_2023', 'eapc_1990_2023_low', 'eapc_1990_2023_high',
             'eapc_1990_2023_q']].head(8).to_string(index=False))
    print(i[['location', 'eapc_1990_2023', 'eapc_1990_2023_low', 'eapc_1990_2023_high',
             'eapc_1990_2023_q']].tail(8).to_string(index=False))


if __name__ == '__main__':
    main()
