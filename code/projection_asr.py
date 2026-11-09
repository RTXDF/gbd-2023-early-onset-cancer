#!/usr/bin/env python3
"""Absolute numbers implied by the directly age-standardised ARIMA projections.

The rate projections in asr_trends.csv are directly standardised within 15-49.
Absolute numbers apply those rates to the damped 2050 population projection used
in projections_number.csv (constant 2023 age structure scenario).
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')

t = pd.read_csv(os.path.join(RES, 'asr_trends.csv'))
p = pd.read_csv(os.path.join(RES, 'projections_number.csv'))
pop = float(p[(p.location == 'Global') & (p.measure == 'Incidence') &
              (p.year == 2050)].pop_scenario.iloc[0])
rows = []
for r in t.itertuples():
    rows.append({'measure': r.measure,
                 'asr_2023': r.asr_2023, 'asr_2050': r.proj_2050,
                 'asr_2050_low': r.proj_2050_low, 'asr_2050_high': r.proj_2050_high,
                 'population_2050': pop,
                 'number_2050': r.proj_2050 / 1e5 * pop,
                 'number_2050_low': r.proj_2050_low / 1e5 * pop,
                 'number_2050_high': r.proj_2050_high / 1e5 * pop,
                 'number_2023_constant': r.asr_2023 / 1e5 * pop,
                 'rate_change_percent': (r.proj_2050 / r.asr_2023 - 1) * 100})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, 'projection_asr_2050.csv'), index=False)
print(out.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))
