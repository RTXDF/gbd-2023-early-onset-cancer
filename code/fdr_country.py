#!/usr/bin/env python3
"""Benjamini-Hochberg FDR control for country-level EAPC significance.

Input : results/trends_country_eapc.csv (from scripts/trends.py)
        columns location_id, location, measure, sex, period, ..., eapc, pvalue
Output: results/trends_country_eapc_fdr.csv
        the input plus q_value (BH within each measure x sex x period) and
        significant_fdr (q_value < 0.05)
        results/fdr_summary.csv
        measure, sex, period, n, n_pos_p05, n_pos_q05, n_neg_p05, n_neg_q05

The country set should contain 204 countries in every measure x sex x period
group; the observed counts are printed and stored in the `n` column.
"""
import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')


def run():
    df = pd.read_csv(os.path.join(RES, 'trends_country_eapc.csv'))
    df['q_value'] = np.nan
    summaries = []
    for (measure, sex, period), idx in df.groupby(['measure', 'sex', 'period']).groups.items():
        sub = df.loc[idx]
        p = sub.pvalue.values.astype(float)
        ok = np.isfinite(p)
        q = np.full(len(sub), np.nan)
        if ok.sum():
            q[ok] = multipletests(p[ok], method='fdr_bh')[1]
        df.loc[idx, 'q_value'] = q
        e = sub.eapc.values
        pos = e > 0
        neg = e < 0
        p05 = p < 0.05
        q05 = q < 0.05
        summaries.append({
            'measure': measure, 'sex': sex, 'period': period, 'n': int(len(sub)),
            'n_pos_p05': int(np.sum(pos & p05)),
            'n_pos_q05': int(np.sum(pos & q05)),
            'n_neg_p05': int(np.sum(neg & p05)),
            'n_neg_q05': int(np.sum(neg & q05)),
        })

    df['significant_fdr'] = df.q_value < 0.05
    df.to_csv(os.path.join(RES, 'trends_country_eapc_fdr.csv'), index=False)
    summ = pd.DataFrame(summaries)[['measure', 'sex', 'period', 'n', 'n_pos_p05',
                                    'n_pos_q05', 'n_neg_p05', 'n_neg_q05']]
    summ.to_csv(os.path.join(RES, 'fdr_summary.csv'), index=False)

    bad = summ[summ.n != 204]
    print(f'trends_country_eapc_fdr.csv: {len(df)} rows')
    print(f'fdr_summary.csv: {len(summ)} groups; country counts: '
          f'{sorted(summ.n.unique())}')
    if len(bad):
        print('WARNING: groups not containing 204 countries:')
        print(bad.to_string(index=False))
    print(summ.to_string(index=False))


if __name__ == '__main__':
    run()
