#!/usr/bin/env python3
"""Joinpoint segments for the directly age-standardised 15-49 series."""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trends import joinpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')


def run():
    g = pd.read_csv(os.path.join(RES, 'global_asr_series.csv'))
    s = pd.read_csv(os.path.join(RES, 'sex_asr_series.csv'))
    s = s.rename(columns={'asr_15_49': 'asr'})
    g = g.rename(columns={'asr_15_49': 'asr'})
    g['sex'] = 'Both'
    g = g[['year', 'measure', 'sex', 'asr']]
    s = s[['year', 'measure', 'sex', 'asr']]
    d = pd.concat([g, s], ignore_index=True)
    rows = []
    for (measure, sex), x in d.groupby(['measure', 'sex']):
        x = x.sort_values('year')
        jp = joinpoint(x.year.values, x.asr.values, max_k=3)
        if jp is None:
            continue
        for seg in jp['segments']:
            rows.append({'measure': measure, 'sex': sex,
                         'n_joinpoints': jp['n_joinpoints'],
                         'joinpoint_years': ';'.join(map(str, jp['joinpoint_years'])),
                         'aapc': jp['aapc'], 'aapc_ci_low': jp['aapc_ci_low'],
                         'aapc_ci_high': jp['aapc_ci_high'],
                         'segment': seg['seg'], 'year_start': seg['year_start'],
                         'year_end': seg['year_end'], 'apc': seg['apc'],
                         'apc_ci_low': seg['ci_low'], 'apc_ci_high': seg['ci_high']})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'asr_joinpoint_segments.csv'), index=False)
    print(out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    run()
