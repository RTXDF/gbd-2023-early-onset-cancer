#!/usr/bin/env python3
"""Cross-national socioeconomic inequality in early-onset cancer burden (ages 15-49).

Revision of the peer-reviewed inequality analysis.  Countries (n=204) are ranked by
the Socio-demographic Index (SDI); all indices are population weighted (population
aged 15-49).

Indices
-------
Concentration index (CI):
    C = 2 * sum_i w_i (y_i - mu) (r_i - rbar) / sum_i w_i / mu
where r_i is the population-weighted fractional SDI rank
    r_i = (cumsum(w)_i - 0.5 w_i) / sum(w).
Slope inequality index (SII): population-weighted least-squares slope of y on r.
Relative inequality index (RII): exp(b) from a Gaussian GLM with a log link and
    population var_weights.  The 95% CI is the corrected ratio-scale interval
        exp(b +/- 1.96 SE)
    (the previous release used exp(b) +/- 1.96 exp(b) SE, which is wrong on the
    ratio scale and could cross zero).
Uncertainty
-----------
CI and SII 95% CIs come from a country bootstrap (countries resampled with
replacement, original population weights retained, indices recomputed and the
2.5/97.5 percentiles taken).  The primary 2023 analysis uses 1000 replicates;
the annual trend analysis uses 500 replicates per year.  For RII the primary
table reports the analytic ratio-scale CI exp(b +/- 1.96 SE) (as required); the
bootstrap percentile interval is appended as rii_ci_low_boot/rii_ci_high_boot.
For the trend table the RII interval reported in rii_ci_low/rii_ci_high is the
bootstrap percentile interval, falling back to the analytic interval if too few
replicates converge.

Rates
-----
Primary 2023 both-sex outcomes are *directly age-standardised within 15-49*:
    ASR = sum_a w_a (N_a / P_a) / sum_a w_a,  a = 15-19 ... 45-49,
with N_a from the age-specific number panels (E_country_age_m*) and P_a from the
population panel (L_pop_sex3); w_a is the Global 2023 population in age group a.
This removes the confounding effect of age structure *within* 15-49 that affects
the crude 15-49 rate.
Age-specific numbers by sex are not available, so male/female outcomes use the
crude 15-49 rate (rate_type='crude').  The trend analysis also uses the directly
age-standardised rates (fixed Global 2023 age weights) for total incidence and
deaths, and the metabolic PAF for deaths (GHI risk 104, age 24, metric 2 x 100;
PAF is expressed against cause 410, identical attributable counts to cause 1059).

Multiple testing
----------------
BH-FDR is applied across the primary 2023 outcome set on the RII p-values and the
adjusted value is reported as q_value.

Outputs (results/)
------------------
inequality_indices.csv        primary 2023 indices (13 outcomes)
inequality_country_data.csv   country-level 2023 input data used for plotting
inequality_trends.csv         annual 1990-2023 indices for incidence, deaths, PAF
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
AUX = os.path.join(ROOT, 'data', 'aux')
RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)

with open(os.path.join(ROOT, 'data', 'id_map.json')) as fh:
    IDMAP = json.load(fh)
COUNTRY_IDS = sorted(int(k) for k in IDMAP['countries'])
COUNTRY_SET = set(COUNTRY_IDS)

AGES = list(range(8, 15))          # 15-19 ... 45-49
MEASURES = {1: 'Deaths', 2: 'DALYs', 6: 'Incidence'}
Z = 1.959963985
SEED = 20261108
N_BOOT_PRIMARY = 1000
N_BOOT_TREND = 500
TREND_YEARS = list(range(1990, 2024))

# ----------------------------------------------------------------------------- panels
POP_ALL = pd.read_pickle(os.path.join(PANELS, 'L_pop_sex3.pkl'))
POP_ALL = POP_ALL[(POP_ALL.measure == 44) & (POP_ALL.metric == 1) & (POP_ALL.sex == 3)]
POP = POP_ALL[POP_ALL.location.isin(COUNTRY_SET)]


def _panel(measure):
    d = pd.read_pickle(os.path.join(PANELS, f'E_country_age_m{measure}.pkl'))
    return d[(d.metric == 1) & (d.sex == 3) & (d.age.isin(AGES)) &
             (d.location.isin(COUNTRY_SET))]


def global_age_weights(year=2023):
    """Global population by 5-year age group (15-19 ... 45-49) in `year`."""
    g = POP_ALL[(POP_ALL.location == 1) & (POP_ALL.year == year) & (POP_ALL.age.isin(AGES))]
    w = g.groupby('age').val.sum()
    missing = set(AGES) - set(w.index)
    if missing:
        raise RuntimeError(f'Global age weights missing for ages {sorted(missing)}')
    return w.reindex(AGES)


def asr_by_country_year(measure, years, weights):
    """Directly age-standardised 15-49 rate (per 100 000) by country and year.

    ASR = sum_a w_a rate_a / sum_a w_a with rate_a = N_a / P_a.
    """
    num = _panel(measure)
    num = num[num.year.isin(years)]
    num = num.groupby(['location', 'year', 'age']).val.sum().unstack('age')
    den = POP[POP.year.isin(years)]
    den = den.groupby(['location', 'year', 'age']).val.sum().unstack('age')
    num = num.reindex(columns=AGES)
    den = den.reindex(columns=AGES)
    common = num.index.intersection(den.index)
    num, den = num.loc[common], den.loc[common]
    rate = num / den
    wv = weights.reindex(AGES).values.astype(float)
    asr = (rate * wv).sum(axis=1, skipna=False) / wv.sum() * 1e5
    out = asr.rename('y').reset_index()
    n_missing = int(out.y.isna().sum())
    if n_missing:
        print(f'  warning: {n_missing} country-year ASR cells undefined for measure {measure}')
    return out.dropna(subset=['y'])


def crude_rate_2023(measure, sex):
    """Published crude 15-49 rate (per 100 000), age 24, metric 3."""
    d = pd.read_pickle(os.path.join(PANELS, f'BCD_country_eo_{sex}.pkl'))
    d = d[(d.year == 2023) & (d.age == 24) & (d.metric == 3) &
          (d.measure == measure) & (d.location.isin(COUNTRY_SET))]
    return (d.groupby('location').val.first().rename('y').reset_index())


def metabolic_paf(year, sex='both'):
    """Metabolic-risk PAF for deaths (percent).  Panel stores a fraction -> x100."""
    d = pd.read_pickle(os.path.join(PANELS, f'GHI_risk104_country_{sex}.pkl'))
    d = d[(d.year == year) & (d.age == 24) & (d.measure == 1) & (d.metric == 2) &
          (d.location.isin(COUNTRY_SET))]
    d = d.groupby('location').val.first() * 100.0
    return d.rename('y').reset_index()


def metabolic_death_rate(year, sex='both'):
    """Metabolic-attributable death rate, crude 15-49 (per 100 000)."""
    d = pd.read_pickle(os.path.join(PANELS, f'GHI_risk104_country_{sex}.pkl'))
    d = d[(d.year == year) & (d.age == 24) & (d.measure == 1) & (d.metric == 3) &
          (d.location.isin(COUNTRY_SET))]
    return (d.groupby('location').val.first().rename('y').reset_index())


def population_weights(year):
    d = POP[(POP.year == year) & (POP.age == 24)]
    return d.groupby('location').val.sum()


def sdi_2023():
    s = pd.read_csv(os.path.join(AUX, 'sdi_quintiles_2023.csv'))
    s.columns = [c.strip() for c in s.columns]
    s = s.rename(columns={'Location ID': 'location_id', '2023 SDI Index Value': 'sdi'})
    s = s[s.location_id.isin(COUNTRY_SET)][['location_id', 'sdi']].dropna()
    return s.set_index('location_id').sdi


def sdi_annual():
    """Annual SDI for every country, 1990-2023 (age 22 all ages, both sexes)."""
    s = pd.read_csv(os.path.join(AUX, 'sdi_values_1950_2023.csv'))
    s = s[(s.covariate_name_short == 'sdi') & (s.age_group_id == 22) & (s.sex_id == 3)]
    s = s[s.location_id.isin(COUNTRY_SET) & s.year_id.isin(TREND_YEARS)]
    piv = s.pivot_table(index='location_id', columns='year_id', values='mean_value',
                        aggfunc='first')
    n_present = piv.notna().sum()
    if (n_present < len(COUNTRY_IDS)).any():
        bad = n_present[n_present < len(COUNTRY_IDS)]
        print(f'  warning: annual SDI missing countries in {len(bad)} year(s): '
              f'{bad.to_dict()}; proceeding with the available intersection')
    return piv


# ----------------------------------------------------------------------------- indices
def _point_indices(y, w, sdi):
    """Population-weighted CI, SII and RII (point estimates + analytic errors)."""
    order = np.argsort(sdi, kind='mergesort')
    y, w, sdi = y[order], w[order], sdi[order]
    cw = np.cumsum(w) / w.sum()
    r = cw - 0.5 * w / w.sum()
    mu = float(np.sum(w * y) / np.sum(w))
    rbar = float(np.sum(w * r) / np.sum(w))
    cov = float(np.sum(w * (y - mu) * (r - rbar)) / np.sum(w))
    ci = 2.0 * cov / mu if mu != 0 else np.nan

    X = sm.add_constant(r)
    wls = sm.WLS(y, X, weights=w).fit()
    sii = float(wls.params[1])
    sii_se = float(wls.bse[1])
    sii_p = float(wls.pvalues[1])

    try:
        glm = sm.GLM(y, X, family=sm.families.Gaussian(link=sm.families.links.Log()),
                     var_weights=w).fit()
        b = float(glm.params[1])
        se = float(glm.bse[1])
        rii = float(np.exp(b))
        # Correct ratio-scale CI: exp(b +/- 1.96 SE).  Never symmetric / <= 0.
        rii_lo = float(np.exp(b - Z * se))
        rii_hi = float(np.exp(b + Z * se))
        rii_p = float(glm.pvalues[1])
    except Exception:
        rii = rii_lo = rii_hi = rii_p = float('nan')

    return {'concentration_index': ci, 'sii': sii,
            'sii_ci_low': sii - Z * sii_se, 'sii_ci_high': sii + Z * sii_se,
            'sii_pvalue': sii_p, 'rii': rii, 'rii_ci_low': rii_lo,
            'rii_ci_high': rii_hi, 'rii_pvalue': rii_p,
            'weighted_mean': mu, 'n_countries': int(len(y))}


def compute_indices(y, w, sdi, n_boot=0, rng=None):
    """Indices with bootstrap percentile CIs for CI / SII (and RII, appended)."""
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    sdi = np.asarray(sdi, float)
    keep = np.isfinite(y) & np.isfinite(w) & np.isfinite(sdi) & (w > 0)
    y, w, sdi = y[keep], w[keep], sdi[keep]
    if len(y) < 10:
        return None
    out = _point_indices(y, w, sdi)
    out.update({'concentration_index_ci_low': np.nan,
                'concentration_index_ci_high': np.nan,
                'rii_ci_low_boot': np.nan, 'rii_ci_high_boot': np.nan,
                'n_boot': 0, 'n_boot_ok': 0})
    if n_boot and rng is not None:
        cis, siis, riis = [], [], []
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                pb = _point_indices(y[idx], w[idx], sdi[idx])
            except Exception:
                continue
            if np.isfinite(pb['concentration_index']):
                cis.append(pb['concentration_index'])
            if np.isfinite(pb['sii']):
                siis.append(pb['sii'])
            if np.isfinite(pb['rii']):
                riis.append(pb['rii'])
        out['n_boot'] = int(n_boot)
        out['n_boot_ok'] = int(len(riis))
        need = max(20, int(0.5 * n_boot))
        if len(cis) >= need:
            out['concentration_index_ci_low'] = float(np.percentile(cis, 2.5))
            out['concentration_index_ci_high'] = float(np.percentile(cis, 97.5))
        if len(siis) >= need:
            out['sii_ci_low'] = float(np.percentile(siis, 2.5))
            out['sii_ci_high'] = float(np.percentile(siis, 97.5))
        if len(riis) >= need:
            out['rii_ci_low_boot'] = float(np.percentile(riis, 2.5))
            out['rii_ci_high_boot'] = float(np.percentile(riis, 97.5))
    return out


def bh_qvalues(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return q


# ----------------------------------------------------------------------------- primary
def primary_2023():
    """Primary 2023 outcome set: both-sex ASR, sex-specific crude, metabolic."""
    pop = population_weights(2023)
    sdi = sdi_2023()
    base = pd.DataFrame({'pop': pop}).join(sdi.rename('sdi'), how='inner')
    base.index.name = 'location_id'
    if len(base) != len(COUNTRY_IDS):
        print(f'  warning: {len(base)} of {len(COUNTRY_IDS)} countries have both '
              f'population and 2023 SDI')

    weights = global_age_weights(2023)
    outcomes = []   # (outcome label, sex, rate_type, tidy frame)

    # both-sex directly age-standardised rates
    for measure in (6, 1, 2):
        asr = asr_by_country_year(measure, [2023], weights)
        outcomes.append((f'Total {MEASURES[measure]} (15-49 ASR)', 'both', 'asr', asr))

    # sex-specific crude 15-49 rates (age-specific numbers unavailable by sex)
    for sex in ('male', 'female'):
        for measure in (6, 1, 2):
            cr = crude_rate_2023(measure, sex)
            outcomes.append((f'Total {MEASURES[measure]} (15-49 rate)', sex, 'crude', cr))

    # metabolic PAF for deaths (both sexes + sex-specific)
    for sex in ('both', 'male', 'female'):
        outcomes.append(('Metabolic PAF (deaths, %)', sex, 'paf', metabolic_paf(2023, sex)))

    # metabolic-attributable death rate, both sexes
    outcomes.append(('Metabolic-attributable death rate', 'both', 'crude',
                     metabolic_death_rate(2023, 'both')))

    rng = np.random.default_rng(SEED)
    rows, country_rows = [], []
    for label, sex, rate_type, frame in outcomes:
        merged = base.merge(frame.rename(columns={'location': 'location_id'}),
                            left_index=True, right_on='location_id', how='inner')
        res = compute_indices(merged.y, merged['pop'], merged.sdi,
                              n_boot=N_BOOT_PRIMARY, rng=rng)
        if res is None:
            print(f'  warning: insufficient countries for {label} ({sex})')
            continue
        rows.append({'outcome': label, 'sex': sex, 'year': 2023,
                     'rate_type': rate_type, **res})
        # country-level data with ordinal and fractional SDI ranks
        c = merged.sort_values('sdi').copy()
        c['rank'] = np.arange(1, len(c) + 1)
        cw = np.cumsum(c['pop'].values) / c['pop'].sum()
        c['fractional_rank'] = cw - 0.5 * c['pop'].values / c['pop'].sum()
        c['outcome'] = label
        c['sex'] = sex
        c['rate_type'] = rate_type
        c['year'] = 2023
        country_rows.append(c[['outcome', 'sex', 'year', 'rate_type', 'location_id',
                               'y', 'pop', 'sdi', 'rank', 'fractional_rank']])

    out = pd.DataFrame(rows)
    out['q_value'] = bh_qvalues(out['rii_pvalue'].values)
    cols = ['outcome', 'sex', 'year', 'n_countries', 'rate_type',
            'concentration_index', 'concentration_index_ci_low',
            'concentration_index_ci_high', 'sii', 'sii_ci_low', 'sii_ci_high',
            'sii_pvalue', 'rii', 'rii_ci_low', 'rii_ci_high', 'rii_pvalue',
            'q_value', 'weighted_mean', 'n_boot', 'n_boot_ok',
            'rii_ci_low_boot', 'rii_ci_high_boot']
    out = out[cols]
    out.to_csv(os.path.join(RES, 'inequality_indices.csv'), index=False)
    cdata = pd.concat(country_rows, ignore_index=True)
    cdata.to_csv(os.path.join(RES, 'inequality_country_data.csv'), index=False)
    return out, cdata


# ----------------------------------------------------------------------------- trends
def trends_1990_2023():
    """Annual CI/SII/RII 1990-2023 for incidence ASR, deaths ASR and deaths PAF."""
    weights = global_age_weights(2023)
    sdi = sdi_annual()
    pop_piv = POP[POP.age == 24].groupby(['location', 'year']).val.sum().unstack('year')

    series = {}
    series['Total Incidence (15-49 ASR)'] = asr_by_country_year(6, TREND_YEARS, weights)
    series['Total Deaths (15-49 ASR)'] = asr_by_country_year(1, TREND_YEARS, weights)
    paf = pd.read_pickle(os.path.join(PANELS, 'GHI_risk104_country_both.pkl'))
    paf = paf[(paf.age == 24) & (paf.measure == 1) & (paf.metric == 2) &
              (paf.location.isin(COUNTRY_SET)) & (paf.year.isin(TREND_YEARS))]
    paf = paf.groupby(['location', 'year']).val.first().mul(100.0).rename('y').reset_index()
    series['Metabolic PAF (deaths, %)'] = paf

    rng = np.random.default_rng(SEED + 1)
    rows = []
    for label, frame in series.items():
        for year in TREND_YEARS:
            y = frame[frame.year == year].set_index('location').y
            w = pop_piv[year] if year in pop_piv.columns else pd.Series(dtype=float)
            s = sdi[year] if year in sdi.columns else pd.Series(dtype=float)
            df = pd.DataFrame({'y': y, 'w': w, 'sdi': s}).dropna()
            if len(df) < 10:
                continue
            res = compute_indices(df.y.values, df.w.values, df.sdi.values,
                                  n_boot=N_BOOT_TREND, rng=rng)
            if res is None:
                continue
            # trend table: RII interval is the bootstrap percentile when available
            if np.isfinite(res.get('rii_ci_low_boot', np.nan)):
                rii_lo, rii_hi = res['rii_ci_low_boot'], res['rii_ci_high_boot']
                method = 'bootstrap'
            else:
                rii_lo, rii_hi = res['rii_ci_low'], res['rii_ci_high']
                method = 'analytic'
            rows.append({'year': year, 'outcome': label,
                         'n_countries': res['n_countries'],
                         'concentration_index': res['concentration_index'],
                         'concentration_index_ci_low': res['concentration_index_ci_low'],
                         'concentration_index_ci_high': res['concentration_index_ci_high'],
                         'sii': res['sii'], 'sii_ci_low': res['sii_ci_low'],
                         'sii_ci_high': res['sii_ci_high'],
                         'rii': res['rii'], 'rii_ci_low': rii_lo, 'rii_ci_high': rii_hi,
                         'weighted_mean': res['weighted_mean'],
                         'n_boot': res['n_boot'], 'n_boot_ok': res['n_boot_ok'],
                         'ci_method': method})
    out = pd.DataFrame(rows)[['year', 'outcome', 'n_countries', 'concentration_index',
                              'concentration_index_ci_low', 'concentration_index_ci_high',
                              'sii', 'sii_ci_low', 'sii_ci_high',
                              'rii', 'rii_ci_low', 'rii_ci_high',
                              'weighted_mean', 'n_boot', 'n_boot_ok', 'ci_method']]
    out.to_csv(os.path.join(RES, 'inequality_trends.csv'), index=False)
    return out


def run():
    print('Primary 2023 inequality (1000-replicate country bootstrap) ...')
    prim, cdata = primary_2023()
    print(prim[['outcome', 'sex', 'rate_type', 'concentration_index', 'sii', 'rii',
                'rii_ci_low', 'rii_ci_high', 'q_value']].to_string(index=False))
    print(f'  inequality_indices.csv: {len(prim)} rows; '
          f'inequality_country_data.csv: {len(cdata)} rows')

    print('\nInequality trends 1990-2023 (500-replicate country bootstrap per year) ...')
    tr = trends_1990_2023()
    print(f'  inequality_trends.csv: {len(tr)} rows; '
          f'n_countries range {int(tr.n_countries.min())}-{int(tr.n_countries.max())}')
    for outcome in tr.outcome.unique():
        sub = tr[tr.outcome == outcome].set_index('year')
        for y in (1990, 2000, 2010, 2023):
            if y in sub.index:
                r = sub.loc[y]
                print(f'  {outcome:34s} {y}: C={r.concentration_index:+.4f} '
                      f'({r.concentration_index_ci_low:+.4f}, {r.concentration_index_ci_high:+.4f}) '
                      f'RII={r.rii:.3f} ({r.rii_ci_low:.3f}, {r.rii_ci_high:.3f})')


if __name__ == '__main__':
    run()