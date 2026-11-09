#!/usr/bin/env python3
"""Site x sex decomposition of early-onset cancer death changes.

Two exact decompositions (Das Gupta symmetric averaging):

  total site deaths        N = population(15-49) x site death rate
  metabolic-attributable   A = population x site death rate x PAF

for 1990-2023 and for the post-2019 window (2019-2023) that carries the mortality
reversal.  Counts are summed over the seven 5-year age groups within 15-49; the
population is the matching sex-specific 15-49 population.  PAF is the GBD
metabolic-attributable fraction for that site and sex (rei 104).

Outputs (results/):
  site_sex_decomposition.csv          every site x sex x period x outcome
  site_sex_decomposition_summary.csv  top contributors and aggregate checks
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
AGES = list(range(8, 15))
PERIODS = [(1990, 2023), (2019, 2023)]
SEX = {'Male': 1, 'Female': 2}


def dasgupta(f0, f1):
    keys = list(f0)
    perms = list(itertools.permutations(keys))
    eff = {k: 0.0 for k in keys}
    for order in perms:
        cur = dict(f0)
        for key in order:
            before = np.prod([cur[x] for x in keys])
            cur[key] = f1[key]
            after = np.prod([cur[x] for x in keys])
            eff[key] += after - before
    return {k: v / len(perms) for k, v in eff.items()}


def load_deaths():
    frames = []
    for sex, sid in SEX.items():
        d = pd.read_pickle(os.path.join(PANELS, f'F_sites_age_dth_m{sid}.pkl'))
        d = d[(d.location == 1) & (d.measure == 1) & (d.metric == 1) &
              (d.age.isin(AGES))]
        d = d.groupby(['cause', 'cause_name', 'year']).val.sum().reset_index()
        d['sex'] = sex
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    # complete the site x sex x year grid with structural zeros
    grid = pd.MultiIndex.from_product(
        [sorted(d.cause.unique()), sorted(d.sex.unique()), sorted(d.year.unique())],
        names=['cause', 'sex', 'year'])
    d = d.set_index(['cause', 'sex', 'year'])[['val']].reindex(grid, fill_value=0).reset_index()
    names = pd.read_pickle(os.path.join(PANELS, 'F_sites_age_dth_m3.pkl'))
    names = names.drop_duplicates('cause').set_index('cause').cause_name
    d['cause_name'] = d.cause.map(names)
    return d


def load_pop():
    out = []
    for sex, sid in SEX.items():
        p = pd.read_pickle(os.path.join(PANELS, f'L_pop_sex{sid}.pkl'))
        p = p[(p.location == 1) & (p.measure == 44) & (p.metric == 1) & (p.age == 24)]
        out.append(p[['year', 'val']].rename(columns={'val': 'pop'}).assign(sex=sex))
    return pd.concat(out, ignore_index=True)


def load_attributable(deaths):
    frames = []
    for f in ('K_risk104_sites_sex_m1', 'K_risk104_sites_sex_2019'):
        d = pd.read_pickle(os.path.join(PANELS, f + '.pkl'))
        d = d[(d.location == 1) & (d.measure == 1) & (d.metric == 1) & (d.age == 24)]
        frames.append(d[['cause', 'year', 'sex', 'val']])
    d = pd.concat(frames, ignore_index=True)
    d['sex'] = d.sex.map({1: 'Male', 2: 'Female', 3: 'Both'})
    d = d[d.sex.isin(['Male', 'Female'])]

    # Pancreatic cancer (456): the sex-specific attributable slice could not be
    # retrieved (GBD API backend outage).  Split the both-sex total from the
    # complete rei-104 series by the sex-specific share of total pancreatic
    # deaths, so the two sexes sum exactly to the published both-sex total.
    pan = pd.read_pickle(os.path.join(PANELS, 'K_risk104_sites_m1_a24.pkl'))
    pan = pan[(pan.location == 1) & (pan.measure == 1) & (pan.metric == 1) &
              (pan.age == 24) & (pan.cause == 456) & (pan.rei == 104) &
              (pan.sex == 3)]
    pan = pan.groupby('year').val.sum()
    tot = deaths[deaths.cause == 456].pivot_table(
        index='year', columns='sex', values='val', aggfunc='sum')
    rows = []
    for sex in ('Male', 'Female'):
        share = tot[sex] / (tot['Male'] + tot['Female'])
        for yr, a in pan.items():
            if yr in share.index:
                rows.append({'cause': 456, 'year': int(yr), 'sex': sex,
                             'val': float(a * share.loc[yr])})
    return pd.concat([d, pd.DataFrame(rows)], ignore_index=True)


def run():
    deaths = load_deaths()
    pop = load_pop()
    attr = load_attributable(deaths)

    rows = []
    for sex in ('Male', 'Female'):
        p = pop[pop.sex == sex].set_index('year')['pop']
        for cause, site in deaths[deaths.sex == sex].drop_duplicates('cause').set_index('cause').cause_name.items():
            d = deaths[(deaths.sex == sex) & (deaths.cause == cause)].set_index('year').val
            for y0, y1 in PERIODS:
                N0, N1 = float(d.loc[y0]), float(d.loc[y1])
                P0, P1 = float(p.loc[y0]), float(p.loc[y1])
                R0, R1 = N0 / P0, N1 / P1
                if N0 <= 0 or N1 <= 0:
                    continue
                e = dasgupta({'population': P0, 'rate': R0},
                             {'population': P1, 'rate': R1})
                rows.append({
                    'sex': sex, 'cause_id': int(cause), 'site': site,
                    'period': f'{y0}-{y1}', 'outcome': 'total_deaths',
                    'value_0': N0, 'value_1': N1, 'change': N1 - N0,
                    'population_effect': e['population'], 'rate_effect': e['rate'],
                    'paf_effect': np.nan,
                })
                # attributable, where GBD quantifies the metabolic PAF
                a = attr[(attr.sex == sex) & (attr.cause == cause)]
                if len(a) and y0 in set(a.year) and y1 in set(a.year):
                    A0 = float(a[a.year == y0].val.sum())
                    A1 = float(a[a.year == y1].val.sum())
                    if A0 > 0 and A1 >= 0:
                        F0 = A0 / N0
                        F1 = A1 / N1
                        ae = dasgupta({'population': P0, 'rate': R0, 'paf': F0},
                                      {'population': P1, 'rate': R1, 'paf': F1})
                        rows.append({
                            'sex': sex, 'cause_id': int(cause), 'site': site,
                            'period': f'{y0}-{y1}', 'outcome': 'attributable_deaths',
                            'value_0': A0, 'value_1': A1, 'change': A1 - A0,
                            'population_effect': ae['population'],
                            'rate_effect': ae['rate'], 'paf_effect': ae['paf'],
                        })

    out = pd.DataFrame(rows)
    for c in ('population_effect', 'rate_effect', 'paf_effect'):
        out[c.replace('_effect', '_share')] = np.where(
            out.change.abs() > 1e-9, out[c] / out.change * 100, np.nan)
    out.to_csv(os.path.join(RES, 'site_sex_decomposition.csv'), index=False)

    # summary: top contributors and checks
    lines = []
    for sex in ('Male', 'Female'):
        for period in ('1990-2023', '2019-2023'):
            t = out[(out.sex == sex) & (out.period == period) &
                    (out.outcome == 'total_deaths')].copy()
            t = t.sort_values('change', ascending=False)
            lines.append(t.assign(sex=sex, period=period))
    summ = pd.concat(lines, ignore_index=True)
    summ.to_csv(os.path.join(RES, 'site_sex_decomposition_summary.csv'), index=False)

    print('Top 6 sites by change in deaths, female 2019-2023:')
    v = summ[(summ.sex == 'Female') & (summ.period == '2019-2023')].head(6)
    print(v[['site', 'value_0', 'value_1', 'change', 'population_effect',
             'rate_effect']].to_string(index=False))
    print('\nTop 6 sites by change in deaths, male 2019-2023:')
    v = summ[(summ.sex == 'Male') & (summ.period == '2019-2023')].head(6)
    print(v[['site', 'value_0', 'value_1', 'change', 'population_effect',
             'rate_effect']].to_string(index=False))

    # check: attributable deaths by sex sum to the published global totals
    totals = {}
    for sex in ('Male', 'Female'):
        a = out[(out.sex == sex) & (out.period == '1990-2023') &
                (out.outcome == 'attributable_deaths')]
        totals[sex] = [float(a.value_0.sum()), float(a.value_1.sum())]
        print(f'{sex}: attributable deaths 1990 {a.value_0.sum():,.0f} -> '
              f'2023 {a.value_1.sum():,.0f}')
    pub = json.load(open(os.path.join(RES, 'attributable_sex_notes.json')))
    published_2023 = float(pub['sex_additivity']['Deaths']['both_2023'])
    print(f'Both sexes 2023: {totals["Male"][1] + totals["Female"][1]:,.0f} '
          f'(published {published_2023:,.0f})')

    notes = {
        'pancreas_attributable': (
            'Sex-specific pancreatic cancer (cause 456) attributable deaths were '
            'obtained by splitting the both-sex metabolic-risk (rei 104) total '
            'from the complete annual series by the sex-specific share of total '
            'pancreatic deaths.  The sex-specific GBD attributable slice could '
            'not be retrieved because of a server-side API failure; the split '
            'values sum exactly to the published both-sex total.'),
        'attributable_deaths_2023': totals,
        'published_both_sexes_2023': published_2023,
    }
    with open(os.path.join(RES, 'site_sex_decomposition_notes.json'), 'w') as fh:
        json.dump(notes, fh, indent=2)
    return out


if __name__ == '__main__':
    run()
