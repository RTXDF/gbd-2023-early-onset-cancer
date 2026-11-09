#!/usr/bin/env python3
"""Age-period-cohort analysis via the Intrinsic Estimator (IE; Yang et al. 2004).

5-year age groups (15-19..45-49) x 5-year periods (1990-94..2020-23).
Period rates reconstructed as sum(numbers)/sum(population)*1e5.
IE = minimum-norm least-squares solution b = pinv(X) y on log(rate),
   X = [intercept, full age dummies, full period dummies, full cohort dummies].
Outputs estimable age/period/cohort effect patterns (rate ratios = exp(effect)).

Uncertainty (revision)
----------------------
The two Global series (incidence and deaths) receive country-bootstrap confidence
intervals: 500 replicates resample the 204 countries with replacement, sum the
age-specific numbers (E_country_age_m*) and populations (L_pop_sex3) into the
7 age x 7 period cells, recompute rates, refit the identical IE, and record the
age/period/cohort effects.  The 2.5/97.5 percentiles are reported as ci_low/ci_high.
Point estimates remain the published Global (location 1) series, not the bootstrap
mean.  `n_boot` is 500 for the Global rows and 0 for any point-only series.
SDI-quintile incidence series are retained if the source panels provide quintile
locations; the age-specific panels used here contain 204 countries plus Global only,
so those series are skipped (the message is printed at run time).

Outputs (results/): apc_ie_effects.csv, apc_ie_fitted.csv
"""
import os
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)

with open(os.path.join(ROOT, 'data', 'id_map.json')) as fh:
    IDMAP = json.load(fh)
COUNTRY_IDS = sorted(int(k) for k in IDMAP['countries'])

AGES = {8: 17, 9: 22, 10: 27, 11: 32, 12: 37, 13: 42, 14: 47}      # age id -> midpoint
AGE_INDEX = {aid: i for i, aid in enumerate(AGES)}                   # 0..6
COHORT_BASE = 1975                                                   # birth year of cohort index 0
PERIODS = [(1990, 1994), (1995, 1999), (2000, 2004), (2005, 2009),
           (2010, 2014), (2015, 2019), (2020, 2023)]
PERIOD_OF = {y: i for i, (a, b) in enumerate(PERIODS) for y in range(a, b + 1)}
N_BOOT = 500
SEED = 20261108


def ie_fit(df):
    """Intrinsic estimator on log(rate) with full age/period/cohort dummies."""
    ages = sorted(df.age_id.unique())
    periods = sorted(df.period.unique())
    cohorts = sorted(df.cohort.unique())
    n = len(df)
    # design: intercept + full age + full period + full cohort dummies
    X = np.ones((n, 1))
    for a in ages:
        X = np.column_stack([X, (df.age_id.values == a).astype(float)])
    for p in periods:
        X = np.column_stack([X, (df.period.values == p).astype(float)])
    for c in cohorts:
        X = np.column_stack([X, (df.cohort.values == c).astype(float)])
    y = df.log_rate.values
    b = np.linalg.pinv(X) @ y  # intrinsic estimator (minimum-norm)
    fitted = X @ b
    idx = 1
    age_eff = {a: b[idx + i] for i, a in enumerate(ages)}
    idx += len(ages)
    per_eff = {p: b[idx + i] for i, p in enumerate(periods)}
    idx += len(periods)
    coh_eff = {c: b[idx + i] for i, c in enumerate(cohorts)}
    return age_eff, per_eff, coh_eff, fitted, ages, periods, cohorts


def build_matrix(loc, measure):
    """Age x period rates for one location (published GBD series)."""
    numf = os.path.join(PANELS, f'E_country_age_m{measure}.pkl')
    popf = os.path.join(PANELS, 'L_pop_sex3.pkl')
    if not (os.path.exists(numf) and os.path.exists(popf)):
        return None
    num = pd.read_pickle(numf)
    pop = pd.read_pickle(popf)
    num = num[(num.location == loc) & (num.metric == 1) & (num.age.isin(AGES)) &
              (num.sex == 3)]
    pop = pop[(pop.location == loc) & (pop.age.isin(AGES)) & (pop.sex == 3) &
              (pop.measure == 44)]
    if len(num) == 0 or len(pop) == 0:
        return None
    rows = []
    for aid, amid in AGES.items():
        for pi, (a, b) in enumerate(PERIODS):
            n = num[(num.age == aid) & (num.year >= a) & (num.year <= b)].val.sum()
            p = pop[(pop.age == aid) & (pop.year >= a) & (pop.year <= b)].val.sum()
            if p <= 0:
                continue
            rate = n / p * 1e5
            if rate <= 0:
                continue
            rows.append({'age_id': aid, 'age_mid': amid, 'period': pi,
                         'period_label': f'{a}-{b}',
                         'cohort': pi - AGE_INDEX[aid],
                         'rate': rate, 'log_rate': np.log(rate)})
    return pd.DataFrame(rows)


def cell_metadata():
    """Ordered cell table matching the columns built by build_country_cells."""
    rows = []
    for aid, amid in AGES.items():
        for pi, (a, b) in enumerate(PERIODS):
            rows.append({'age_id': aid, 'age_mid': amid, 'period': pi,
                         'period_label': f'{a}-{b}',
                         'cohort': pi - AGE_INDEX[aid],
                         'cohort_year': COHORT_BASE + 5 * (pi - AGE_INDEX[aid])})
    return pd.DataFrame(rows)


def build_country_cells(measure):
    """Country x cell matrices of numbers and population (rows = countries).

    Columns follow cell_metadata(): age id outer, period inner (49 cells).
    """
    num = pd.read_pickle(os.path.join(PANELS, f'E_country_age_m{measure}.pkl'))
    num = num[(num.metric == 1) & (num.sex == 3) & (num.age.isin(AGES)) &
              (num.location.isin(COUNTRY_IDS))]
    num = num.assign(period=num.year.map(PERIOD_OF))
    num = num.dropna(subset=['period'])
    num = num.groupby(['location', 'age', 'period']).val.sum()

    pop = pd.read_pickle(os.path.join(PANELS, 'L_pop_sex3.pkl'))
    pop = pop[(pop.measure == 44) & (pop.metric == 1) & (pop.sex == 3) &
              (pop.age.isin(AGES)) & (pop.location.isin(COUNTRY_IDS))]
    pop = pop.assign(period=pop.year.map(PERIOD_OF))
    pop = pop.dropna(subset=['period'])
    pop = pop.groupby(['location', 'age', 'period']).val.sum()

    idx = pd.MultiIndex.from_product([COUNTRY_IDS, list(AGES.keys()), range(len(PERIODS))],
                                     names=['location', 'age', 'period'])
    num = num.reindex(idx, fill_value=0.0).values.reshape(len(COUNTRY_IDS), -1)
    pop = pop.reindex(idx).values.reshape(len(COUNTRY_IDS), -1)
    return num, pop


def bootstrap_effects(measure, n_boot=N_BOOT, seed=SEED):
    """Country-bootstrap percentile CIs for the Global IE effects."""
    num_c, pop_c = build_country_cells(measure)
    meta = cell_metadata()
    n = len(COUNTRY_IDS)
    rng = np.random.default_rng(seed)
    age_draws, per_draws, coh_draws = {}, {}, {}
    ok = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        cnt = np.bincount(idx, minlength=n).astype(float)
        num_b = cnt @ num_c
        pop_b = cnt @ pop_c
        rate = np.divide(num_b, pop_b, out=np.zeros_like(num_b), where=pop_b > 0) * 1e5
        valid = np.isfinite(rate) & (rate > 0)
        if valid.sum() < 20:
            continue
        df = meta[valid].copy()
        df['rate'] = rate[valid]
        df['log_rate'] = np.log(df.rate.values)
        try:
            age_eff, per_eff, coh_eff, *_ = ie_fit(df)
        except Exception:
            continue
        ok += 1
        for a, v in age_eff.items():
            age_draws.setdefault(a, []).append(float(v))
        for p, v in per_eff.items():
            per_draws.setdefault(p, []).append(float(v))
        for c, v in coh_eff.items():
            coh_draws.setdefault(c, []).append(float(v))
    return age_draws, per_draws, coh_draws, ok


def _ci(draws, key):
    arr = np.asarray(draws.get(key, []), float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return np.nan, np.nan
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def run():
    eff_rows, fit_rows = [], []
    targets = [(1, 'Global', 'Deaths', True), (6, 'Global', 'Incidence', True)]
    # SDI quintiles (kept when the source panels contain the quintile locations)
    for qid, qname in IDMAP['sdi_quintiles'].items():
        targets.append((6, qname, 'Incidence', False))

    for measure, locname, mlabel, want_boot in targets:
        loc = next((int(k) for k, v in IDMAP['locations'].items() if v == locname), None)
        if loc is None:
            continue
        df = build_matrix(loc, measure)
        if df is None or len(df) < 20:
            print(f'  note: no age-specific panel for {locname} / {mlabel}; '
                  f'series skipped (only 204 countries + Global are available)')
            continue
        age_eff, per_eff, coh_eff, fitted, ages, periods, cohorts = ie_fit(df)

        if want_boot:
            age_d, per_d, coh_d, ok = bootstrap_effects(measure, n_boot=N_BOOT,
                                                        seed=SEED)
            print(f'  {locname} {mlabel}: {ok}/{N_BOOT} bootstrap replicates converged')
            nboot = ok
        else:
            age_d = per_d = coh_d = {}
            nboot = 0

        for a, e in age_eff.items():
            lo, hi = _ci(age_d, a)
            eff_rows.append({'location': locname, 'measure': mlabel, 'effect_type': 'age',
                             'level': IDMAP['ages'][str(a)], 'estimate': e,
                             'rate_ratio': np.exp(e), 'ci_low': lo, 'ci_high': hi,
                             'n_boot': nboot})
        for p, e in per_eff.items():
            lo, hi = _ci(per_d, p)
            eff_rows.append({'location': locname, 'measure': mlabel, 'effect_type': 'period',
                             'level': PERIODS[p][0], 'estimate': e,
                             'rate_ratio': np.exp(e), 'ci_low': lo, 'ci_high': hi,
                             'n_boot': nboot})
        for c, e in coh_eff.items():
            lo, hi = _ci(coh_d, c)
            eff_rows.append({'location': locname, 'measure': mlabel, 'effect_type': 'cohort',
                             'level': COHORT_BASE + 5 * c, 'estimate': e,
                             'rate_ratio': np.exp(e), 'ci_low': lo, 'ci_high': hi,
                             'n_boot': nboot})

        df2 = df.copy()
        df2['fitted_log_rate'] = fitted
        df2['cohort_year'] = COHORT_BASE + 5 * df2['cohort']
        df2['location'] = locname
        df2['measure'] = mlabel
        fit_rows.append(df2)

    effects = pd.DataFrame(eff_rows)[['location', 'measure', 'effect_type', 'level',
                                      'estimate', 'rate_ratio', 'ci_low', 'ci_high',
                                      'n_boot']]
    effects.to_csv(os.path.join(RES, 'apc_ie_effects.csv'), index=False)
    if fit_rows:
        pd.concat(fit_rows, ignore_index=True).to_csv(
            os.path.join(RES, 'apc_ie_fitted.csv'), index=False)
    print(f'APC-IE effect rows: {len(effects)}; fitted panels: {len(fit_rows)}')
    boot = effects[effects.n_boot > 0]
    print(f'  effects with bootstrap CIs: {len(boot)} rows '
          f'({boot.location.nunique()} series)')


if __name__ == '__main__':
    run()
