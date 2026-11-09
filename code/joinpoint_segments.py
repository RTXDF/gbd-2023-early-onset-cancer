#!/usr/bin/env python3
"""Package joinpoint output as a clean segment table + maximum-joinpoint sensitivity.

Inputs
------
results/trends_global_joinpoint.csv
    Produced by scripts/trends.py.  For every measure x sex x age there is one
    summary row (n_joinpoints, joinpoint_years, AAPC, BIC) followed by one row per
    segment (segment, year_start, year_end, APC and CI).
data/panels/A_global_trends.pkl
    Global (location 1) sex-specific 15-49 rates (cause 1059, age 24, metric 3)
    used to refit the joinpoint model under different maximum numbers of
    joinpoints.

Outputs
-------
results/joinpoint_segments.csv
    One row per segment, restricted to age 15-49 (both sexes + male + female).
    Segment summary fields (n_joinpoints, joinpoint years, AAPC, BIC) are carried
    onto every segment row; segment_years = year_end - year_start + 1.
results/joinpoint_sensitivity.csv
    Refits of the Global both-sex and sex-specific 15-49 series with
    max_k = 1, 2, 3, 4, reporting the BIC-selected number of joinpoints, the
    joinpoint years, BIC, AAPC and the last-segment APC.  This tests whether the
    3-joinpoint selection used in the manuscript is stable.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
PANELS = os.path.join(ROOT, 'data', 'panels')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from trends import joinpoint  # noqa: E402  (reuse the published joinpoint model)

MEASURES = {1: 'Deaths', 2: 'DALYs', 6: 'Incidence'}
SEXES = {1: 'Male', 2: 'Female', 3: 'Both'}


def segment_table():
    src = pd.read_csv(os.path.join(RES, 'trends_global_joinpoint.csv'))
    src = src[src.age == '15-49']
    rows = []
    for (measure, sex, age), g in src.groupby(['measure', 'sex', 'age'], sort=False):
        summary = g[g.n_joinpoints.notna()]
        if summary.empty:
            continue
        s = summary.iloc[0]
        segs = g[g.segment.notna()].sort_values('segment')
        for r in segs.itertuples():
            rows.append({
                'measure': measure, 'sex': sex, 'age': age,
                'n_joinpoints': int(s.n_joinpoints),
                'joinpoint_years': s.joinpoint_years,
                'aapc': s.aapc, 'aapc_ci_low': s.aapc_ci_low,
                'aapc_ci_high': s.aapc_ci_high, 'bic': s.bic,
                'segment': int(r.segment), 'year_start': int(r.year_start),
                'year_end': int(r.year_end),
                'segment_years': int(r.year_end - r.year_start + 1),
                'apc': r.apc, 'apc_ci_low': r.apc_ci_low, 'apc_ci_high': r.apc_ci_high,
            })
    out = pd.DataFrame(rows)[['measure', 'sex', 'age', 'n_joinpoints',
                              'joinpoint_years', 'aapc', 'aapc_ci_low',
                              'aapc_ci_high', 'bic', 'segment', 'year_start',
                              'year_end', 'segment_years', 'apc', 'apc_ci_low',
                              'apc_ci_high']]
    out.to_csv(os.path.join(RES, 'joinpoint_segments.csv'), index=False)
    return out


def sensitivity_table():
    a = pd.read_pickle(os.path.join(PANELS, 'A_global_trends.pkl'))
    a = a[(a.location == 1) & (a.cause == 1059) & (a.age == 24) & (a.metric == 3) &
          (a.sex.isin([1, 2, 3]))]
    rows = []
    for measure in (1, 2, 6):
        for sex in (3, 1, 2):
            g = a[(a.measure == measure) & (a.sex == sex)].sort_values('year')
            if len(g) < 6:
                continue
            for max_k in (1, 2, 3, 4):
                jp = joinpoint(g.year.values, g.val.values, max_k=max_k)
                if jp is None:
                    continue
                last = jp['segments'][-1]
                rows.append({
                    'measure': MEASURES[measure], 'sex': SEXES[sex], 'age': '15-49',
                    'max_k': max_k,
                    'selected_n_joinpoints': int(jp['n_joinpoints']),
                    'joinpoint_years': ';'.join(map(str, jp['joinpoint_years'])),
                    'bic': jp['bic'], 'aapc': jp['aapc'],
                    'last_segment_apc': last['apc'],
                    'last_segment_year_start': last['year_start'],
                    'last_segment_year_end': last['year_end'],
                })
    out = pd.DataFrame(rows)[['measure', 'sex', 'age', 'max_k',
                              'selected_n_joinpoints', 'joinpoint_years', 'bic',
                              'aapc', 'last_segment_apc',
                              'last_segment_year_start', 'last_segment_year_end']]
    out.to_csv(os.path.join(RES, 'joinpoint_sensitivity.csv'), index=False)
    return out


def run():
    seg = segment_table()
    print(f'joinpoint_segments.csv: {len(seg)} rows; '
          f'{seg.groupby(["measure", "sex", "age"]).ngroups} series')
    print(seg.groupby(['measure', 'sex']).agg(
        n_joinpoints=('n_joinpoints', 'first'), segments=('segment', 'count'),
        last_apc=('apc', 'last')).to_string())

    sen = sensitivity_table()
    print(f'\njoinpoint_sensitivity.csv: {len(sen)} rows')
    print(sen[['measure', 'sex', 'max_k', 'selected_n_joinpoints',
               'joinpoint_years', 'last_segment_apc']].to_string(index=False))


if __name__ == '__main__':
    run()
