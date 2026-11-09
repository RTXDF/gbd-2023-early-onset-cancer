#!/usr/bin/env python3
"""Age-standardised (within 15-49) trends, composition decomposition and scenarios.

The GBD 'age-standardized' rate uses the all-age standard population and is not
appropriate for the 15-49 band.  Here we directly standardise the seven 5-year
age groups (15-19 ... 45-49) to the global 2023 15-49 population, both sexes.

Outputs (results/):
  global_asr_series.csv              annual counts, crude rate, ASR and UI
  asr_trends.csv                     EAPC / joinpoint / ARIMA summary
  age_composition_decomposition.csv  Kitagawa rate vs composition effects
  projection_recent_trend.csv        2013-2023 log-linear scenario to 2050
  sex_eapc_wide.csv                  sex-specific EAPC + last-segment APC
"""
import os, json, math, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trends import eapc, joinpoint
from projection import fit_project

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
AGES = [8, 9, 10, 11, 12, 13, 14]
MEASURES = {6: 'Incidence', 1: 'Deaths', 2: 'DALYs'}
Z = 1.959963985


def load_inputs():
    num = {m: pd.read_pickle(os.path.join(PANELS, f'E_country_age_m{m}.pkl')) for m in MEASURES}
    pop = pd.read_pickle(os.path.join(PANELS, 'L_pop_sex3.pkl'))
    return num, pop


def weights(pop):
    w = pop[(pop.location == 1) & (pop.year == 2023) & (pop.age.isin(AGES)) & (pop.metric == 1)]
    w = w.set_index('age').val
    return w / w.sum()


def global_series(num, pop, w):
    rows = []
    for meas, label in MEASURES.items():
        d = num[meas]
        d = d[(d.location == 1) & (d.metric == 1) & (d.sex == 3) & (d.age.isin(AGES))]
        p = pop[(pop.location == 1) & (pop.metric == 1) & (pop.sex == 3) & (pop.age.isin(AGES))]
        for yr in sorted(d.year.unique()):
            rr, rl, rh = {}, {}, {}
            for a in AGES:
                pp = p[(p.year == yr) & (p.age == a)].val.sum()
                row = d[(d.year == yr) & (d.age == a)]
                rr[a] = row.val.sum() / pp * 1e5
                rl[a] = row.lower.sum() / pp * 1e5
                rh[a] = row.upper.sum() / pp * 1e5
            n = d[d.year == yr].val.sum()
            nlo = d[d.year == yr].lower.sum()
            nhi = d[d.year == yr].upper.sum()
            pp = p[p.year == yr].val.sum()
            rows.append({
                'year': int(yr), 'measure': label, 'count': n, 'count_low': nlo, 'count_high': nhi,
                'population': pp, 'crude_rate': n / pp * 1e5,
                'crude_low': nlo / pp * 1e5, 'crude_high': nhi / pp * 1e5,
                'asr_15_49': sum(w[a] * rr[a] for a in AGES),
                'asr_low': sum(w[a] * rl[a] for a in AGES),
                'asr_high': sum(w[a] * rh[a] for a in AGES),
            })
    return pd.DataFrame(rows)


def kitagawa(num, pop, meas, label):
    d = num[meas]
    d = d[(d.location == 1) & (d.metric == 1) & (d.sex == 3) & (d.age.isin(AGES))]
    p = pop[(pop.location == 1) & (pop.metric == 1) & (pop.sex == 3) & (pop.age.isin(AGES))]
    out = {}
    for yr in (1990, 2023):
        rr, ww = {}, {}
        for a in AGES:
            n = d[(d.year == yr) & (d.age == a)].val.sum()
            pp = p[(p.year == yr) & (p.age == a)].val.sum()
            rr[a] = n / pp * 1e5
            ww[a] = pp
        tot = sum(ww.values())
        out[yr] = ({a: rr[a] for a in AGES}, {a: ww[a] / tot for a in AGES})
    r0, w0 = out[1990]
    r1, w1 = out[2023]
    crude0 = sum(w0[a] * r0[a] for a in AGES)
    crude1 = sum(w1[a] * r1[a] for a in AGES)
    rate_eff = sum(w0[a] * (r1[a] - r0[a]) for a in AGES)
    comp_eff = sum((w1[a] - w0[a]) * r0[a] for a in AGES)
    inter = sum((w1[a] - w0[a]) * (r1[a] - r0[a]) for a in AGES)
    return {'measure': label, 'period': '1990-2023', 'crude_1990': crude0, 'crude_2023': crude1,
            'total_change': crude1 - crude0, 'rate_effect': rate_eff,
            'composition_effect': comp_eff, 'interaction': inter,
            'composition_share_percent': 100 * comp_eff / (crude1 - crude0) if crude1 != crude0 else np.nan}


def sex_eapc_table():
    p = os.path.join(PANELS, 'A_global_trends.pkl')
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_pickle(p)
    d = d[(d.location == 1) & (d.cause == 1059) & (d.age == 24) & (d.metric == 3)]
    rows = []
    for meas, label in MEASURES.items():
        for sex, sname in [(1, 'Male'), (2, 'Female'), (3, 'Both')]:
            g = d[(d.measure == meas) & (d.sex == sex)].sort_values('year')
            if len(g) < 10:
                continue
            rec = {'measure': label, 'sex': sname}
            for start, end, tag in [(1990, 2023, 'eapc_1990_2023'), (2013, 2023, 'eapc_2013_2023')]:
                e = eapc(g.year.values, g.val.values, start, end)
                if e:
                    rec[f'{tag}'] = e['eapc']
                    rec[f'{tag}_low'] = e['ci_low']
                    rec[f'{tag}_high'] = e['ci_high']
                    rec[f'{tag}_p'] = e['pvalue']
            jp = joinpoint(g.year.values, g.val.values, max_k=3)
            if jp:
                rec['n_joinpoints'] = jp['n_joinpoints']
                rec['joinpoint_years'] = ';'.join(map(str, jp['joinpoint_years']))
                s = jp['segments'][-1]
                rec['last_apc'] = s['apc']
                rec['last_apc_low'] = s['ci_low']
                rec['last_apc_high'] = s['ci_high']
                rec['last_segment'] = f"{s['year_start']}-{s['year_end']}"
            rows.append(rec)
    return pd.DataFrame(rows)


def recent_trend_scenarios(series):
    p = pd.read_csv(os.path.join(RES, 'projections_number.csv'))
    pop = p[(p.location == 'Global') & (p.measure == 'Incidence')].set_index('year').pop_scenario
    rows = []
    for meas, label in MEASURES.items():
        g = series[series.measure == label].sort_values('year')
        e = eapc(g.year.values, g.crude_rate.values, 2013, 2023)
        if not e:
            continue
        r0 = float(g[g.year == 2023].crude_rate.iloc[0])
        for yr in range(2024, 2051):
            h = yr - 2023
            rate = r0 * math.exp(e['eapc'] / 100 * h)
            rows.append({'location': 'Global', 'measure': label, 'scenario': 'recent_trend_loglinear',
                         'year': yr, 'rate': rate, 'number': rate / 1e5 * pop.get(yr, np.nan),
                         'eapc_2013_2023': e['eapc']})
    # attributable rate, recent-trend extrapolation
    ap = os.path.join(RES, 'attributable_trends_eapc.csv')
    ag = os.path.join(RES, 'attributable_global.csv')
    if os.path.exists(ap) and os.path.exists(ag):
        at = pd.read_csv(ap)
        a = pd.read_csv(ag)
        for meas, label in [(1, 'Deaths'), (2, 'DALYs')]:
            row = at[(at.measure == label) & (at.period == '2013-2023')]
            if len(row) == 0:
                continue
            e = float(row.eapc.iloc[0])
            g = a[(a.risk == 'Metabolic risks') & (a.sex == 'both') & (a.age == '15-49') &
                  (a.measure == label) & (a.metric == 'Rate')].sort_values('year')
            r0 = float(g[g.year == 2023].val.iloc[0])
            for yr in range(2024, 2051):
                h = yr - 2023
                rate = r0 * math.exp(e / 100 * h)
                rows.append({'location': 'Global', 'measure': label, 'scenario': 'recent_trend_loglinear',
                             'year': yr, 'rate': rate, 'number': rate / 1e5 * pop.get(yr, np.nan),
                             'eapc_2013_2023': e, 'outcome_type': 'metabolic_attributable'})
    return pd.DataFrame(rows)


def main():
    num, pop = load_inputs()
    w = weights(pop)
    series = global_series(num, pop, w)
    series.to_csv(os.path.join(RES, 'global_asr_series.csv'), index=False)

    summary, comp = [], []
    for meas, label in MEASURES.items():
        g = series[series.measure == label].sort_values('year')
        rec = {'measure': label, 'asr_1990': g.asr_15_49.iloc[0], 'asr_2023': g.asr_15_49.iloc[-1],
               'crude_1990': g.crude_rate.iloc[0], 'crude_2023': g.crude_rate.iloc[-1]}
        for start, end, tag in [(1990, 2023, 'eapc_1990_2023'), (2013, 2023, 'eapc_2013_2023')]:
            e = eapc(g.year.values, g.asr_15_49.values, start, end)
            rec[tag] = e['eapc']; rec[tag + '_low'] = e['ci_low']; rec[tag + '_high'] = e['ci_high']
            rec[tag + '_p'] = e['pvalue']
        jp = joinpoint(g.year.values, g.asr_15_49.values, max_k=3)
        rec['joinpoints'] = ';'.join(map(str, jp['joinpoint_years']))
        rec['aapc'] = jp['aapc']; rec['aapc_low'] = jp['aapc_ci_low']; rec['aapc_high'] = jp['aapc_ci_high']
        s = jp['segments'][-1]
        rec['last_apc'] = s['apc']; rec['last_apc_low'] = s['ci_low']; rec['last_apc_high'] = s['ci_high']
        rec['last_segment'] = f"{s['year_start']}-{s['year_end']}"
        fp = fit_project(g.year.values, g.asr_15_49.values, 27)
        rec['proj_2050'] = fp['mean'][-1]; rec['proj_2050_low'] = fp['low'][-1]; rec['proj_2050_high'] = fp['high'][-1]
        rec['arima_order'] = str(fp['order'])
        summary.append(rec)
        comp.append(kitagawa(num, pop, meas, label))
    pd.DataFrame(summary).to_csv(os.path.join(RES, 'asr_trends.csv'), index=False)
    pd.DataFrame(comp).to_csv(os.path.join(RES, 'age_composition_decomposition.csv'), index=False)

    sex_eapc_table().to_csv(os.path.join(RES, 'sex_eapc_wide.csv'), index=False)
    recent_trend_scenarios(series).to_csv(os.path.join(RES, 'projection_recent_trend.csv'), index=False)
    print('wrote global_asr_series, asr_trends, age_composition_decomposition, sex_eapc_wide, projection_recent_trend')


if __name__ == '__main__':
    main()
