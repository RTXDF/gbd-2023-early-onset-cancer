#!/usr/bin/env python3
"""Formal test of sex divergence in global early-onset cancer trends.

Model (per measure x period), on annual log rates for ages 15-49:
    log(rate_t) = b0 + b1 * (year - mean_year) + b2 * female + b3 * (year - mean_year) * female
with female = 1 for women.  b3 is the difference in log-scale annual change between
women and men, i.e. exp(b3) is the ratio of the female to male annual rate ratio.
An implied EAPC is computed for each sex:
    EAPC_male   = (exp(b1) - 1) * 100
    EAPC_female = (exp(b1 + b3) - 1) * 100
A significant b3 (and ratio_of_eapc != 1) is the formal evidence that the two sex
trends diverge.

Inputs : data/panels/A_global_trends.pkl (location 1, cause 1059, age 24,
         metric 3 rate, sex 1 male / 2 female).  If the global panel lacked
         sex-specific age 24 rates, the sex-specific BCD country panels would be
         used instead; this is checked explicitly.
Output : results/sex_interaction.csv
         measure, period, beta_interaction, se, pvalue, ratio_of_eapc,
         female_eapc, male_eapc (+ n_years, ci_low, ci_high, ratio_of_rate_ratio)
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
MEASURES = {1: 'Deaths', 2: 'DALYs', 6: 'Incidence'}
PERIODS = [('1990-2023', 1990, 2023), ('2013-2023', 2013, 2023)]
Z = 1.959963985


def load_global_sex_rates():
    a = pd.read_pickle(os.path.join(PANELS, 'A_global_trends.pkl'))
    a = a[(a.location == 1) & (a.cause == 1059) & (a.age == 24) & (a.metric == 3) &
          (a.sex.isin([1, 2]))]
    if a.empty:
        # fallback: sex-specific country panels, location 1 (Global)
        frames = []
        for sex, code in (('male', 1), ('female', 2)):
            d = pd.read_pickle(os.path.join(PANELS, f'BCD_country_eo_{sex}.pkl'))
            d = d[(d.location == 1) & (d.cause == 1059) & (d.age == 24) &
                  (d.metric == 3)].copy()
            d['sex'] = code
            frames.append(d)
        a = pd.concat(frames, ignore_index=True)
        if a.empty:
            raise RuntimeError('No global sex-specific 15-49 rates available')
        print('  note: fell back to sex-specific BCD panels for Global rates')
    return a


def run():
    a = load_global_sex_rates()
    rows = []
    for measure in (1, 2, 6):
        for label, y0, y1 in PERIODS:
            d = a[(a.measure == measure) & (a.year >= y0) & (a.year <= y1)]
            wide = d.pivot_table(index='year', columns='sex', values='val', aggfunc='first')
            wide = wide.dropna(subset=[1, 2])
            if len(wide) < 5:
                continue
            y = np.log(wide.values.astype(float))
            year = wide.index.values.astype(float)
            yc = year - year.mean()
            # rows: male then female
            n = len(year)
            yr = np.concatenate([yc, yc])
            female = np.concatenate([np.zeros(n), np.ones(n)])
            X = np.column_stack([np.ones(2 * n), yr, female, yr * female])
            yy = np.concatenate([y[:, 0], y[:, 1]])
            fit = sm.OLS(yy, X).fit()
            b1 = float(fit.params[1])
            b3 = float(fit.params[3])
            se3 = float(fit.bse[3])
            p3 = float(fit.pvalues[3])
            rows.append({
                'measure': MEASURES[measure], 'period': label,
                'beta_interaction': b3, 'se': se3, 'pvalue': p3,
                'ratio_of_eapc': float(np.exp(b3)),
                'female_eapc': float((np.exp(b1 + b3) - 1) * 100),
                'male_eapc': float((np.exp(b1) - 1) * 100),
                'n_years': int(n),
                'ci_low': float(np.exp(b3 - Z * se3)),
                'ci_high': float(np.exp(b3 + Z * se3)),
            })
    out = pd.DataFrame(rows)[['measure', 'period', 'beta_interaction', 'se', 'pvalue',
                              'ratio_of_eapc', 'female_eapc', 'male_eapc',
                              'n_years', 'ci_low', 'ci_high']]
    out.to_csv(os.path.join(RES, 'sex_interaction.csv'), index=False)
    print(f'sex_interaction.csv: {len(out)} rows')
    print(out[['measure', 'period', 'beta_interaction', 'se', 'pvalue',
               'ratio_of_eapc', 'female_eapc', 'male_eapc']].to_string(index=False))


if __name__ == '__main__':
    run()
