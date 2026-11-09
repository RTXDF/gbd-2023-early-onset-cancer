#!/usr/bin/env python3
"""Annual directly age-standardised ARIMA projections and 2050 scenarios.

Primary projections in the revision are fitted to the 15-49 directly
age-standardised series (not the crude GBD rate series).  This script stores
the annual trajectory and the scenario table used by Figure 9 and Table 7.
"""
import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from projection import fit_project

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
MEASURES = ['Incidence', 'Deaths', 'DALYs']

series = pd.read_csv(os.path.join(RES, 'global_asr_series.csv'))
num = pd.read_csv(os.path.join(RES, 'projections_number.csv'))
g = num[(num.location == 'Global') & (num.year == 2050)].set_index('measure')
pop_damped = float(g.loc['Incidence', 'pop_scenario'])
# projections_number.csv stores undamped absolute numbers; their ratio to the
# damped numbers equals the ratio of the two population scenarios.
pop_undamped = pop_damped * (float(g.loc['Incidence', 'undamped_pop_scenario']) /
                             float(g.loc['Incidence', 'projected_number']))

rows, scen = [], []
for meas in MEASURES:
    s = series[series.measure == meas].sort_values('year')
    fp = fit_project(s.year.values, s.asr_15_49.values, 27)
    for yr, r, lo, hi in zip(range(2024, 2051), fp['mean'], fp['low'], fp['high']):
        rows.append({'measure': meas, 'year': yr, 'asr': r, 'asr_low': lo, 'asr_high': hi,
                     'arima_order': str(fp['order'])})
    r0 = float(s[s.year == 2023].asr_15_49.iloc[0])
    r50 = float(fp['mean'][-1])
    lo50, hi50 = float(fp['low'][-1]), float(fp['high'][-1])
    scen.append({'measure': meas, 'scenario': 'arima_asr_damped_pop', 'asr_2050': r50,
                 'asr_2050_low': lo50, 'asr_2050_high': hi50,
                 'number_2050': r50 / 1e5 * pop_damped,
                 'number_2050_low': lo50 / 1e5 * pop_damped,
                 'number_2050_high': hi50 / 1e5 * pop_damped,
                 'rate_change_percent': (r50 / r0 - 1) * 100})
    scen.append({'measure': meas, 'scenario': 'arima_asr_undamped_pop', 'asr_2050': r50,
                 'asr_2050_low': lo50, 'asr_2050_high': hi50,
                 'number_2050': r50 / 1e5 * pop_undamped,
                 'number_2050_low': lo50 / 1e5 * pop_undamped,
                 'number_2050_high': hi50 / 1e5 * pop_undamped,
                 'rate_change_percent': (r50 / r0 - 1) * 100})
    scen.append({'measure': meas, 'scenario': 'constant_2023_asr_damped_pop', 'asr_2050': r0,
                 'asr_2050_low': np.nan, 'asr_2050_high': np.nan,
                 'number_2050': r0 / 1e5 * pop_damped, 'number_2050_low': np.nan,
                 'number_2050_high': np.nan, 'rate_change_percent': 0.0})

pd.DataFrame(rows).to_csv(os.path.join(RES, 'projection_asr_series.csv'), index=False)
out = pd.DataFrame(scen)
out.to_csv(os.path.join(RES, 'projections_asr_scenarios_2050.csv'), index=False)
print('damped population', pop_damped, 'undamped', pop_undamped)
print(out.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))
