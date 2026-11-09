#!/usr/bin/env python3
"""Sex-specific directly age-standardised rates (15-49) and trends.

Direct standardisation of the seven 5-year age groups (15-19 ... 45-49)
within each sex to the 2023 global population of the same sex, using the
age-specific rates in A_global_trends.pkl (cause 1059, location 1).

Outputs (results/):
  sex_asr_series.csv   annual male/female ASR for incidence, deaths, DALYs
  sex_asr_trends.csv   EAPC (1990-2023, 2013-2023) and joinpoint summary
"""
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trends import eapc, joinpoint

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
AGES = [8, 9, 10, 11, 12, 13, 14]
MEASURES = {6: 'Incidence', 1: 'Deaths', 2: 'DALYs'}
SEXES = [(1, 'Male', 'L_pop_sex1.pkl'), (2, 'Female', 'L_pop_sex2.pkl')]


def main():
    d = pd.read_pickle(os.path.join(PANELS, 'A_global_trends.pkl'))
    d = d[(d.location == 1) & (d.cause == 1059) & (d.metric == 3) & (d.age.isin(AGES))]
    series_rows, trend_rows = [], []
    for meas, label in MEASURES.items():
        for sex, sname, pf in SEXES:
            pop = pd.read_pickle(os.path.join(PANELS, pf))
            w = pop[(pop.location == 1) & (pop.year == 2023) &
                    (pop.age.isin(AGES)) & (pop.metric == 1)].set_index('age').val
            w = w / w.sum()
            g = d[(d.measure == meas) & (d.sex == sex)]
            asr = g.groupby('year').apply(
                lambda x: sum(w[a] * x[x.age == a].val.iloc[0] for a in AGES),
                include_groups=False)
            for yr, v in asr.items():
                series_rows.append({'year': int(yr), 'measure': label, 'sex': sname,
                                    'asr_15_49': float(v)})
            rec = {'measure': label, 'sex': sname,
                   'asr_1990': float(asr.loc[1990]), 'asr_2023': float(asr.loc[2023]),
                   'change_percent': float((asr.loc[2023] / asr.loc[1990] - 1) * 100)}
            for start, end, tag in [(1990, 2023, 'eapc_1990_2023'), (2013, 2023, 'eapc_2013_2023')]:
                e = eapc(asr.index.values, asr.values, start, end)
                rec[tag] = e['eapc']
                rec[tag + '_low'] = e['ci_low']
                rec[tag + '_high'] = e['ci_high']
                rec[tag + '_p'] = e['pvalue']
            jp = joinpoint(asr.index.values, asr.values, max_k=3)
            rec['joinpoints'] = ';'.join(map(str, jp['joinpoint_years']))
            rec['aapc'] = jp['aapc']
            rec['aapc_low'] = jp['aapc_ci_low']
            rec['aapc_high'] = jp['aapc_ci_high']
            s = jp['segments'][-1]
            rec['last_apc'] = s['apc']
            rec['last_apc_low'] = s['ci_low']
            rec['last_apc_high'] = s['ci_high']
            rec['last_segment'] = f"{s['year_start']}-{s['year_end']}"
            trend_rows.append(rec)
    pd.DataFrame(series_rows).to_csv(os.path.join(RES, 'sex_asr_series.csv'), index=False)
    pd.DataFrame(trend_rows).to_csv(os.path.join(RES, 'sex_asr_trends.csv'), index=False)
    print('wrote sex_asr_series.csv, sex_asr_trends.csv')


if __name__ == '__main__':
    main()
