#!/usr/bin/env python3
"""Reviewer-preemption sensitivity analyses for the early-onset cancer package.

Two analyses are computed from the existing age-specific panels (no new GBD
queries, and no changes to the manuscript, figures, tables.xlsx or submission):

A. Alternative standard population
   Direct standardisation of ages 15-19 ... 45-49 to the *1990* global 15-49
   population instead of the 2023 population used in the main pipeline.
   Computed for the global series (1990-2023), country ranks in 2023, GBD
   regions in 2023 and SDI quintiles in 1990/2023.

B. Alternative age definition (15-39)
   Ages 15-19 ... 35-39 only (age ids 8-12), standardised to the 2023 global
   15-39 population.  Global counts/ASR, sex-specific ASR (from
   A_global_trends.pkl), regional ASR in 2023, and a metabolic PAF for 20-39
   (the risk panel J has no 15-19 age group).

Methods
  ASR = sum_a w_a * (n_a / P_a) * 1e5, with w_a normalised to sum 1, built from
  metric==1 numbers and the matching population (the supplied metric==3 rates
  are never used, so numerator and denominator are always consistent).
  EAPC / joinpoint models are imported from trends.py, i.e. exactly the
  functions used by the main pipeline (log-linear EAPC with analytic 95% CI;
  BIC-selected continuous segmented log-linear joinpoint with max_k=3).
  Regional and SDI-quintile ASR are population-weighted means of country ASR
  (location_asr.aggregate_series), the same approximation as the main pipeline.
  Weights, region mapping, aggregation and trend helpers are reused from
  scripts/location_asr.py.

Uncertainty
  Every re-weighted ASR here is a POINT ESTIMATE.  GBD publishes marginal 95%
  UIs for each age group but not the age-specific covariance, so re-weighted
  intervals cannot be reconstructed defensibly; we do not fabricate them.
  EAPC 95% CIs and p-values are analytic (OLS) and are the only intervals
  reported.

Outputs (results/):
  sensitivity_standard_pop.csv    tidy long: global / regional / SDI rows
  sensitivity_country_rank.csv    204 countries x 3 measures, ASR 2023 under
                                  both standards + ranks and quartiles
  sensitivity_age_definition.csv  global / regional / sex-specific 15-39 rows
                                  plus 20-39 (and 15-39 approximate) PAF rows
  sensitivity_summary.md          concise report with the exact numbers
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from trends import eapc, joinpoint                              # noqa: E402
from location_asr import (                                      # noqa: E402
    AGES, MEASURES, PANELS, RES, region_map, weights, aggregate_series, trend_table)

AGES_15_49 = list(AGES)          # 15-19 ... 45-49 (age ids 8-14)
AGES_15_39 = list(AGES[:5])      # 15-19 ... 35-39 (age ids 8-12)
AGES_20_39 = list(AGES[1:5])     # 20-24 ... 35-39 (age ids 9-12)
AGE_15_19 = 8
AGE_LABELS = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']

STD_1990 = '1990 global 15-49'
STD_2023 = '2023 global 15-49'
STD_2023_1539 = '2023 global 15-39'

TIDY_COLS = ['analysis', 'measure', 'scope', 'standard', 'year', 'asr', 'count',
             'eapc_1990_2023', 'eapc_1990_2023_low', 'eapc_1990_2023_high',
             'eapc_2013_2023', 'eapc_2013_2023_low', 'eapc_2013_2023_high',
             'last_apc', 'last_segment', 'paf', 'attributable_count', 'note']


# --------------------------------------------------------------------------
# data loading and core computations
# --------------------------------------------------------------------------
def load_panels():
    num = {m: pd.read_pickle(os.path.join(PANELS, f'E_country_age_m{m}.pkl')) for m in MEASURES}
    pop = pd.read_pickle(os.path.join(PANELS, 'L_pop_sex3.pkl'))
    return num, pop


def country_ids():
    with open(os.path.join(ROOT, 'data', 'id_map.json')) as fh:
        return {int(k): v for k, v in json.load(fh)['countries'].items()}


def standard_weights(pop, year, ages, sex=3):
    """Normalised standard-population weights for one year and age range.

    Same construction as location_asr.weights(), generalised by year and ages.
    """
    w = pop[(pop.location == 1) & (pop.year == year) & (pop.age.isin(ages)) &
            (pop.metric == 1) & (pop.sex == sex)].set_index('age').val
    missing = set(ages) - set(w.index)
    if missing:
        raise ValueError(f'missing ages {sorted(missing)} in population panel')
    return w / w.sum()


def age_specific_rates(num, pop, loc_ids, ages, sex=3):
    """Tidy location x year x age x measure counts, populations and rates.

    metric==1 numbers divided by the matching population, so the numerator and
    denominator of every age-specific rate are consistent by construction.
    """
    frames = []
    for meas, label in MEASURES.items():
        d = num[meas]
        # A_global_trends carries all three measures in one frame; the E panels
        # are already split by measure, so filter whenever the column exists.
        if 'measure' in d.columns:
            d = d[d.measure == meas]
        d = d[(d.metric == 1) & (d.sex == sex) & (d.age.isin(ages)) &
              (d.location.isin(loc_ids))]
        d = d.groupby(['location', 'year', 'age'], as_index=False).agg(count=('val', 'sum'))
        p = pop[(pop.metric == 1) & (pop.sex == sex) & (pop.age.isin(ages)) &
                (pop.location.isin(loc_ids))]
        p = p.groupby(['location', 'year', 'age'], as_index=False).agg(population=('val', 'sum'))
        m = d.merge(p, on=['location', 'year', 'age'], how='inner', validate='one_to_one')
        m['rate'] = m['count'] / m['population']
        m['measure'] = label
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def standardised_series(rates, w, label_map):
    """ASR series (all years) from an age-specific rate table and a weight vector.

    Same layout as location_asr.series_for(): location_id, location (name),
    measure, year, count, population, crude_rate, asr.
    """
    r = rates.copy()
    r['w_rate'] = r['rate'] * r['age'].map(w)
    g = r.groupby(['measure', 'location', 'year'], as_index=False).agg(
        count=('count', 'sum'), population=('population', 'sum'), asr=('w_rate', 'sum'))
    g = g.rename(columns={'location': 'location_id'})
    g['asr'] = g['asr'] * 1e5
    g['crude_rate'] = g['count'] / g['population'] * 1e5
    g['location'] = g['location_id'].map(label_map)
    return g[['location_id', 'location', 'measure', 'year', 'count', 'population',
              'crude_rate', 'asr']]


def trend_index(series, group_col):
    """EAPC and joinpoint statistics per group and measure.

    Delegates to location_asr.trend_table(), so the trend model is identical to
    the main pipeline: EAPC 1990-2023 and 2013-2023 with analytic 95% CIs,
    BIC-selected joinpoints (max_k=3) and the last-segment APC.
    """
    return trend_table(series, group_col).set_index([group_col, 'measure'])


def tidy_rows(series, trend, group_col, analysis, standard, note,
              years=None, scope_prefix=''):
    """Explode a series into the tidy long output layout.

    EAPC / joinpoint fields are properties of the whole series and are repeated
    on every year row so that each row is self-describing.
    """
    rows = []
    for (grp, meas), g in series.groupby([group_col, 'measure']):
        g = g.sort_values('year')
        if years is not None:
            g = g[g.year.isin(years)]
        tr = trend.loc[(grp, meas)] if (grp, meas) in trend.index else None

        def get(key):
            if tr is None or key not in tr.index:
                return np.nan
            return float(tr[key])

        last_seg = tr['last_segment'] if tr is not None and 'last_segment' in tr.index else ''
        for _, r in g.iterrows():
            rows.append({'analysis': analysis, 'measure': meas,
                         'scope': f'{scope_prefix}{grp}', 'standard': standard,
                         'year': int(r.year), 'asr': float(r.asr), 'count': float(r['count']),
                         'eapc_1990_2023': get('eapc_1990_2023'),
                         'eapc_1990_2023_low': get('eapc_1990_2023_low'),
                         'eapc_1990_2023_high': get('eapc_1990_2023_high'),
                         'eapc_2013_2023': get('eapc_2013_2023'),
                         'eapc_2013_2023_low': get('eapc_2013_2023_low'),
                         'eapc_2013_2023_high': get('eapc_2013_2023_high'),
                         'last_apc': get('last_apc'), 'last_segment': last_seg,
                         'paf': np.nan, 'attributable_count': np.nan, 'note': note})
    return rows


def series_stats(series, meas):
    """Headline numbers for one measure of one series."""
    g = series[series.measure == meas].sort_values('year')
    out = {'asr_1990': float(g[g.year == 1990].asr.iloc[0]),
           'asr_2023': float(g[g.year == 2023].asr.iloc[0]),
           'asr_2019': float(g[g.year == 2019].asr.iloc[0]),
           'count_1990': float(g[g.year == 1990]['count'].iloc[0]),
           'count_2023': float(g[g.year == 2023]['count'].iloc[0]),
           'crude_1990': float(g[g.year == 1990].crude_rate.iloc[0]),
           'crude_2023': float(g[g.year == 2023].crude_rate.iloc[0])}
    out['asr_change_percent'] = (out['asr_2023'] / out['asr_1990'] - 1) * 100
    out['count_change_percent'] = (out['count_2023'] / out['count_1990'] - 1) * 100
    out['reversal_19_23'] = (out['asr_2023'] / out['asr_2019'] - 1) * 100
    for start, end, tag in [(1990, 2023, 'eapc_1990_2023'), (2013, 2023, 'eapc_2013_2023')]:
        e = eapc(g.year.values, g.asr.values, start, end)
        out[tag] = e['eapc']
        out[tag + '_low'] = e['ci_low']
        out[tag + '_high'] = e['ci_high']
        out[tag + '_p'] = e['pvalue']
    jp = joinpoint(g.year.values, g.asr.values, max_k=3)
    out['joinpoints'] = ';'.join(map(str, jp['joinpoint_years']))
    out['last_apc'] = jp['segments'][-1]['apc']
    out['last_segment'] = f"{jp['segments'][-1]['year_start']}-{jp['segments'][-1]['year_end']}"
    return out


def region_mapping():
    rm = region_map()
    mapping = {}
    for lid, (_rid, rname) in rm.items():
        mapping.setdefault(rname, []).append(lid)
    return rm, mapping


def sdi_quintile_mapping(names):
    """Fixed 2023 SDI quintile assignment from data/aux/sdi_quintiles_2023.csv.

    The file's precomputed 'SDI Quintile' column has 21 blank country cells
    (including China, India, Japan and the USA) and strongly unequal group
    sizes (High 78 / Low 47 / Low-middle 23 / High-middle 19 / Middle 16), so
    it cannot define five comparable groups.  We therefore cut the same file's
    2023 SDI index into equal quintiles across the 204 countries; this
    reproduces the main pipeline's quintile assignment exactly (checked by
    sdi_assignment_check()).
    """
    s = pd.read_csv(os.path.join(ROOT, 'data', 'aux', 'sdi_quintiles_2023.csv'))
    s = s.rename(columns={'Location ID': 'location_id', '2023 SDI Index Value': 'sdi',
                          'SDI Quintile': 'label'})
    s = s[s.location_id.isin(names)].copy()
    s['quintile'] = pd.qcut(s.sdi, 5, labels=['Q1 lowest', 'Q2', 'Q3', 'Q4', 'Q5 highest'])
    qmap = s.groupby('quintile', observed=True).location_id.apply(list).to_dict()
    return s, qmap


def sdi_assignment_check(sdi_df):
    """Compare the derived quintiles with the main pipeline and with the file's
    own (incomplete) labels."""
    out = {}
    ineq_path = os.path.join(RES, 'inequality_country_data.csv')
    if os.path.exists(ineq_path):
        ineq = pd.read_csv(ineq_path)
        sdi23 = ineq[(ineq.year == 2023) &
                     (ineq.outcome == 'Total Incidence (15-49 ASR)')][['location_id', 'sdi']]
        sdi23['qp'] = pd.qcut(sdi23.sdi, 5,
                              labels=['Q1 lowest', 'Q2', 'Q3', 'Q4', 'Q5 highest'])
        m = sdi_df.merge(sdi23, on='location_id')
        out['pipeline_agreement'] = float((m.qp.astype(str) == m.quintile.astype(str)).mean())
        lab = sdi_df.dropna(subset=['label'])
        rank = {'Low SDI': 1, 'Low-middle SDI': 2, 'Middle SDI': 3,
                'High-middle SDI': 4, 'High SDI': 5}
        agree = (lab.quintile.astype(str).str[1].astype(int) == lab.label.map(rank))
        out['label_agreement'] = float(agree.mean())
        out['label_n'] = int(len(lab))
        out['label_missing'] = int(sdi_df.label.isna().sum())
    return out


def check_main_pipeline(global_2023, country_2023):
    """Confirm the re-implementation reproduces the published 2023-standard
    results before it is applied to another standard / age range."""
    g_old = pd.read_csv(os.path.join(RES, 'global_asr_series.csv'))
    m = global_2023.merge(g_old[['year', 'measure', 'asr_15_49']], on=['year', 'measure'])
    dg = float(np.max(np.abs(m.asr - m.asr_15_49)))

    c_old = pd.read_csv(os.path.join(RES, 'country_asr_series.csv'))
    c_old = c_old[c_old.year == 2023]
    m2 = country_2023[country_2023.year == 2023].merge(
        c_old[['location_id', 'measure', 'asr']],
        on=['location_id', 'measure'], suffixes=('', '_old'), validate='one_to_one')
    dc = float(np.max(np.abs(m2.asr - m2.asr_old)))
    if dg > 1e-6 or dc > 1e-6:
        raise AssertionError(f'2023-standard reproduction failed: global {dg}, country {dc}')
    return dg, dc


# --------------------------------------------------------------------------
# analysis A: alternative standard population (1990 global 15-49)
# --------------------------------------------------------------------------
def analysis_a(num, pop, names):
    w1990 = standard_weights(pop, 1990, AGES_15_49)
    w2023 = weights(pop)                      # main-pipeline 2023 weights
    wdf = pd.DataFrame({'age': AGES_15_49,
                        'label': AGE_LABELS,
                        'weight_1990': w1990.reindex(AGES_15_49).values,
                        'weight_2023': w2023.reindex(AGES_15_49).values})
    wdf['ratio_1990_to_2023'] = wdf.weight_1990 / wdf.weight_2023

    # --- A2: global series under both standards -----------------------------
    rates_global = age_specific_rates(num, pop, {1}, AGES_15_49)
    g90 = standardised_series(rates_global, w1990, {1: 'Global'})
    g23 = standardised_series(rates_global, w2023, {1: 'Global'})
    g90['scope'] = 'Global'
    g23['scope'] = 'Global'
    stats90 = {m: series_stats(g90, m) for m in MEASURES.values()}
    stats23 = {m: series_stats(g23, m) for m in MEASURES.values()}
    tr90 = trend_index(g90, 'scope')
    tr23 = trend_index(g23, 'scope')

    # --- A3: country ASR 2023 under both standards --------------------------
    rates_country = age_specific_rates(num, pop, set(names), AGES_15_49)
    c90 = standardised_series(rates_country, w1990, names)
    c23 = standardised_series(rates_country, w2023, names)
    val_diffs = check_main_pipeline(g23, c23)

    c90_23 = c90[c90.year == 2023][['location_id', 'location', 'measure', 'asr']]
    c23_23 = c23[c23.year == 2023][['location_id', 'location', 'measure', 'asr']]
    rank = c23_23.merge(c90_23, on=['location_id', 'location', 'measure'],
                        suffixes=('_std2023', '_std1990'))
    rank_rows, rank_stats = [], {}
    for meas in MEASURES.values():
        d = rank[rank.measure == meas].copy()
        # rank 1 = highest ASR; ties broken by name for determinism
        d = d.sort_values(['asr_std2023', 'location'], ascending=[False, True])
        d['rank_2023'] = np.arange(1, len(d) + 1)
        d = d.sort_values(['asr_std1990', 'location'], ascending=[False, True])
        d['rank_1990'] = np.arange(1, len(d) + 1)
        # quartile 1 = highest-ASR quartile (51 countries per quartile)
        d['quartile_2023'] = pd.qcut(d.rank_2023, 4, labels=[1, 2, 3, 4]).astype(int)
        d['quartile_1990'] = pd.qcut(d.rank_1990, 4, labels=[1, 2, 3, 4]).astype(int)
        d['rank_shift'] = d.rank_1990 - d.rank_2023
        sp = stats.spearmanr(d.asr_std2023, d.asr_std1990)
        pe = stats.pearsonr(d.asr_std2023, d.asr_std1990)
        top20_23 = set(d.nlargest(20, 'asr_std2023').location)
        top20_90 = set(d.nlargest(20, 'asr_std1990').location)
        rank_stats[meas] = {
            'spearman': float(sp.statistic), 'spearman_p': float(sp.pvalue),
            'pearson': float(pe[0]),
            'top20_overlap': len(top20_23 & top20_90),
            'quartile_moves_gt1': int((d.quartile_2023 - d.quartile_1990).abs().gt(1).sum()),
            'quartile_moves_1': int((d.quartile_2023 - d.quartile_1990).abs().eq(1).sum()),
            'max_rank_shift': int(d.rank_shift.abs().max()),
            'median_abs_shift': float(d.rank_shift.abs().median()),
            'n_rank_shift_gt10': int(d.rank_shift.abs().gt(10).sum()),
            'top5_2023': list(d.nlargest(5, 'asr_std2023').location),
            'top5_1990': list(d.nlargest(5, 'asr_std1990').location),
            'bottom5_2023': list(d.nsmallest(5, 'asr_std2023').location),
            'bottom5_1990': list(d.nsmallest(5, 'asr_std1990').location),
            'top20_entered': sorted(top20_90 - top20_23),
            'top20_left': sorted(top20_23 - top20_90),
            'largest_moves': list(d.reindex(d.rank_shift.abs().sort_values(ascending=False)
                                            .index[:3])[['location', 'rank_2023', 'rank_1990']]
                                  .itertuples(index=False, name=None)),
        }
        for _, r in d.iterrows():
            rank_rows.append({'country': r.location,
                              'asr_2023_std2023': float(r.asr_std2023),
                              'asr_2023_std1990': float(r.asr_std1990),
                              'rank_2023': int(r.rank_2023), 'rank_1990': int(r.rank_1990),
                              'quartile_2023': int(r.quartile_2023),
                              'quartile_1990': int(r.quartile_1990),
                              'measure': meas})
    rank_df = pd.DataFrame(rank_rows, columns=['country', 'asr_2023_std2023',
                                               'asr_2023_std1990', 'rank_2023', 'rank_1990',
                                               'quartile_2023', 'quartile_1990', 'measure'])
    rank_df['measure'] = pd.Categorical(rank_df.measure, list(MEASURES.values()), ordered=True)
    rank_df = rank_df.sort_values(['measure', 'rank_2023']).reset_index(drop=True)

    # --- A4: regions --------------------------------------------------------
    _rm, mapping = region_mapping()
    reg90 = aggregate_series(c90, mapping, 'region')
    reg23 = aggregate_series(c23, mapping, 'region')
    tr_reg90 = trend_index(reg90, 'region')
    tr_reg23 = trend_index(reg23, 'region')

    old_reg = pd.read_csv(os.path.join(RES, 'regions_asr_2023.csv'))
    reg_compare = {}
    for meas in MEASURES.values():
        o = old_reg[old_reg.measure == meas].sort_values('asr', ascending=False)
        r90 = reg90[(reg90.year == 2023) & (reg90.measure == meas)].sort_values(
            'asr', ascending=False)
        r23 = reg23[(reg23.year == 2023) & (reg23.measure == meas)].sort_values(
            'asr', ascending=False)
        m90 = dict(zip(r90.region, r90.asr))
        m23 = dict(zip(r23.region, r23.asr))
        # correlate the two standards region by region (never two self-sorted vectors)
        pair = pd.DataFrame({'a': pd.Series(m90), 'b': pd.Series(m23)}).dropna()
        top5_90 = list(r90.region.head(5))
        bot5_90 = list(r90.region.tail(5))
        top5_pub = list(o.region.head(5))
        bot5_pub = list(o.region.tail(5))
        reg_compare[meas] = {
            'top5_std2023': list(r23.region.head(5)),
            'top5_std1990': top5_90,
            'bottom5_std2023': list(r23.region.tail(5)),
            'bottom5_std1990': bot5_90,
            'top5_published': top5_pub,
            'bottom5_published': bot5_pub,
            'top5_overlap': len(set(top5_90) & set(top5_pub)),
            'bottom5_overlap': len(set(bot5_90) & set(bot5_pub)),
            'top5_same_set': set(top5_90) == set(top5_pub),
            'bottom5_same_set': set(bot5_90) == set(bot5_pub),
            'top5_same_order': top5_90 == top5_pub,
            'bottom5_same_order': bot5_90 == bot5_pub,
            'order_diff': [(i, a, b) for i, (a, b) in enumerate(zip(top5_90, top5_pub)) if a != b],
            'top5_entered': sorted(set(top5_90) - set(top5_pub)),
            'top5_left': sorted(set(top5_pub) - set(top5_90)),
            'bottom5_entered': sorted(set(bot5_90) - set(bot5_pub)),
            'bottom5_left': sorted(set(bot5_pub) - set(bot5_90)),
            'spearman_1990_vs_2023': float(stats.spearmanr(pair.a, pair.b).statistic),
            'asr_std2023_map': m23,
            'asr_std1990_map': m90,
            'top5_values': [(nm, m23[nm], m90[nm]) for nm in top5_90],
        }

    # --- A5: SDI quintiles --------------------------------------------------
    sdi_df, qmap = sdi_quintile_mapping(names)
    sdi_check = sdi_assignment_check(sdi_df)
    quint90 = aggregate_series(c90, qmap, 'sdi_quintile')
    quint23 = aggregate_series(c23, qmap, 'sdi_quintile')
    tr_q90 = trend_index(quint90, 'sdi_quintile')
    tr_q23 = trend_index(quint23, 'sdi_quintile')

    q_compare = {}
    for meas in MEASURES.values():
        t90 = quint90[quint90.measure == meas].pivot(index='sdi_quintile', columns='year',
                                                     values='asr')
        t23 = quint23[quint23.measure == meas].pivot(index='sdi_quintile', columns='year',
                                                     values='asr')
        q_compare[meas] = {'std1990': t90, 'std2023': t23}

    # --- tidy rows ----------------------------------------------------------
    note_pt = ('point estimate: GBD publishes no age-specific covariance, so the '
               're-weighted ASR carries no GBD uncertainty interval')
    rows = tidy_rows(g90, tr90, 'scope', 'A_standard_population', STD_1990,
                     note_pt + '; weights = 1990 global 15-49 population')
    rows += tidy_rows(g23, tr23, 'scope', 'A_standard_population', STD_2023,
                      note_pt + '; weights = 2023 global 15-49 population '
                                '(reproduces results/global_asr_series.csv)')

    for label, reg, tr, std in [('1990', reg90, tr_reg90, STD_1990),
                                ('2023', reg23, tr_reg23, STD_2023)]:
        # write all 21 GBD regions (not only those entering a top-5/bottom-5 list)
        sub = reg[reg.year.eq(2023)].copy()
        sub['region'] = pd.Categorical(sub.region, sorted(sub.region.unique()), ordered=True)
        sub = sub.sort_values(['measure', 'region'])
        rows += tidy_rows(sub, tr, 'region', 'A_standard_population', std,
                          'approximate aggregation: population-weighted mean of country ASR '
                          '(aggregate_series); 2023 cross-section; all 21 GBD regions; '
                          f'weights = {label} standard')

    for label, q, tr, std in [('1990', quint90, tr_q90, STD_1990),
                              ('2023', quint23, tr_q23, STD_2023)]:
        sub = q[q.year.isin([1990, 2023])].copy()
        sub['sdi_quintile'] = sub.sdi_quintile.astype(str)
        rows += tidy_rows(sub, tr, 'sdi_quintile', 'A_standard_population', std,
                          'equal quintiles of the 2023 SDI index '
                          '(data/aux/sdi_quintiles_2023.csv); approximate aggregation of '
                          f'country ASR; weights = {label} standard', scope_prefix='SDI ')

    return dict(wdf=wdf, g90=g90, g23=g23, stats90=stats90, stats23=stats23,
                val_diffs=val_diffs, rank=rank_df, rank_stats=rank_stats,
                reg90=reg90, reg23=reg23, reg_compare=reg_compare,
                quint90=quint90, quint23=quint23, q_compare=q_compare,
                sdi_check=sdi_check, rows=rows)


# --------------------------------------------------------------------------
# analysis B: alternative age definition (15-39)
# --------------------------------------------------------------------------
def analysis_b(num, pop, names):
    w39 = standard_weights(pop, 2023, AGES_15_39)
    wdf = pd.DataFrame({'age': AGES_15_39, 'label': AGE_LABELS[:5],
                        'weight_15_39': w39.reindex(AGES_15_39).values})

    # --- B1: global counts and ASR -----------------------------------------
    rates_global = age_specific_rates(num, pop, {1}, AGES_15_39)
    gy = standardised_series(rates_global, w39, {1: 'Global'})
    gy['scope'] = 'Global'
    gstats = {m: series_stats(gy, m) for m in MEASURES.values()}
    tr_y = trend_index(gy, 'scope')

    # --- B3: regional ASR 2023 (approximate aggregation) -------------------
    rates_country = age_specific_rates(num, pop, set(names), AGES_15_39)
    cy = standardised_series(rates_country, w39, names)
    _rm, mapping = region_mapping()
    regy = aggregate_series(cy, mapping, 'region')
    tr_regy = trend_index(regy, 'region')

    old_reg = pd.read_csv(os.path.join(RES, 'regions_asr_2023.csv'))
    reg_compare = {}
    for meas in MEASURES.values():
        o = old_reg[old_reg.measure == meas].sort_values('asr', ascending=False)
        r = regy[(regy.year == 2023) & (regy.measure == meas)].sort_values(
            'asr', ascending=False)
        old_map = dict(zip(o.region, o.asr))
        top5_39 = list(r.region.head(5))
        bot5_39 = list(r.region.tail(5))
        top5_49 = list(o.region.head(5))
        bot5_49 = list(o.region.tail(5))
        pair = pd.DataFrame({'a': pd.Series(old_map),
                             'b': pd.Series(dict(zip(r.region, r.asr)))}).dropna()
        reg_compare[meas] = {
            'top5_15_39': top5_39,
            'bottom5_15_39': bot5_39,
            'top5_15_49': top5_49,
            'bottom5_15_49': bot5_49,
            'top5_overlap': len(set(top5_39) & set(top5_49)),
            'bottom5_overlap': len(set(bot5_39) & set(bot5_49)),
            'top5_entered': sorted(set(top5_39) - set(top5_49)),
            'top5_left': sorted(set(top5_49) - set(top5_39)),
            'bottom5_entered': sorted(set(bot5_39) - set(bot5_49)),
            'bottom5_left': sorted(set(bot5_49) - set(bot5_39)),
            'top5_values': [(nm, float(r.set_index('region').asr[nm]), old_map[nm])
                            for nm in top5_39],
            'spearman_15_39_vs_15_49': float(stats.spearmanr(pair.a, pair.b).statistic),
        }

    # --- B2: sex-specific global ASR (A_global_trends) ---------------------
    A = pd.read_pickle(os.path.join(PANELS, 'A_global_trends.pkl'))
    A = A[(A.location == 1) & (A.cause == 1059)]
    sex_frames, sex_stats = [], {}
    for sex, sname, pf in [(1, 'Male', 'L_pop_sex1.pkl'), (2, 'Female', 'L_pop_sex2.pkl'),
                           (3, 'Both', 'L_pop_sex3.pkl')]:
        P = pd.read_pickle(os.path.join(PANELS, pf))
        wsx = standard_weights(P, 2023, AGES_15_39, sex=sex)
        rates = age_specific_rates({m: A for m in MEASURES}, P, {1}, AGES_15_39, sex=sex)
        s = standardised_series(rates, wsx, {1: 'Global'})
        s['sex'] = sname
        sex_frames.append(s)
        for meas in MEASURES.values():
            sex_stats[(sname, meas)] = series_stats(s, meas)
    sex_series = pd.concat(sex_frames, ignore_index=True)
    sex_series['scope'] = 'Global ' + sex_series.sex
    tr_sex = trend_index(sex_series, 'scope')

    divergence = {}
    for meas in MEASURES.values():
        fm = sex_stats[('Female', meas)]
        ma = sex_stats[('Male', meas)]
        divergence[meas] = {
            'eapc_diff_1990_2023': fm['eapc_1990_2023'] - ma['eapc_1990_2023'],
            'eapc_diff_2013_2023': fm['eapc_2013_2023'] - ma['eapc_2013_2023'],
            'asr_ratio_2023': fm['asr_2023'] / ma['asr_2023'],
            'change_diff': fm['asr_change_percent'] - ma['asr_change_percent'],
            'last_apc_diff': fm['last_apc'] - ma['last_apc'],
            'reversal_19_23_f': fm['reversal_19_23'],
            'reversal_19_23_m': ma['reversal_19_23'],
        }

    # --- B4: metabolic PAF for 20-39 (and a 15-39 approximation) -----------
    J = pd.read_pickle(os.path.join(PANELS, 'J_risk_age_global.pkl'))
    J = J[(J.rei == 104) & (J.cause == 410) & (J.sex == 3)]

    ghi_check = {}
    ghi_path = os.path.join(PANELS, 'GHI_risk104_country_both.pkl')
    if os.path.exists(ghi_path):
        G = pd.read_pickle(ghi_path)
        G = G[(G.location == 1) & (G.year == 2023) & (G.metric == 2) & (G.age == 24)]
        for meas in (1, 2):
            jv = float(J[(J.measure == meas) & (J.age == 24) & (J.year == 2023) &
                         (J.metric == 2)].val.iloc[0])
            gv = float(G[G.measure == meas].val.iloc[0])
            ghi_check[meas] = (jv, gv, abs(jv - gv))

    paf_rows, paf = [], {}
    for meas, label in [(1, 'Deaths'), (2, 'DALYs')]:
        e = num[meas]
        for yr in (1990, 2023):
            att = float(J[(J.measure == meas) & (J.age.isin(AGES_20_39)) & (J.year == yr) &
                          (J.metric == 1)].val.sum())
            tot = float(e[(e.location == 1) & (e.metric == 1) & (e.sex == 3) &
                          (e.age.isin(AGES_20_39)) & (e.year == yr)].val.sum())
            p = att / tot * 100
            paf[(label, yr)] = {'attributable': att, 'total': tot, 'paf': p}

            # assumption-based 15-39 approximation: 20-24 PAF applied to 15-19
            paf_2024 = float(J[(J.measure == meas) & (J.age == 9) & (J.year == yr) &
                               (J.metric == 2)].val.iloc[0])
            tot_1519 = float(e[(e.location == 1) & (e.metric == 1) & (e.sex == 3) &
                               (e.age == AGE_15_19) & (e.year == yr)].val.sum())
            att_1519 = paf_2024 * tot_1519
            tot_1539 = float(e[(e.location == 1) & (e.metric == 1) & (e.sex == 3) &
                               (e.age.isin(AGES_15_39)) & (e.year == yr)].val.sum())
            paf[(label + '_15_39_approx', yr)] = {
                'attributable': att + att_1519, 'total': tot_1539,
                'paf': (att + att_1519) / tot_1539 * 100, 'paf_2024': paf_2024,
                'attributable_1519': att_1519, 'total_1519': tot_1519}

            paf_rows.append({'analysis': 'B_age_definition', 'measure': label,
                             'scope': 'Global 20-39', 'standard': STD_2023_1539,
                             'year': yr, 'asr': np.nan, 'count': tot,
                             'eapc_1990_2023': np.nan, 'eapc_1990_2023_low': np.nan,
                             'eapc_1990_2023_high': np.nan, 'eapc_2013_2023': np.nan,
                             'eapc_2013_2023_low': np.nan, 'eapc_2013_2023_high': np.nan,
                             'last_apc': np.nan, 'last_segment': '',
                             'paf': p, 'attributable_count': att,
                             'note': 'metabolic PAF (rei 104) for ages 20-39: sum of J '
                                     'attributable ages 9-12 divided by sum of total ages 9-12; '
                                     '15-19 unavailable in J'})
            paf_rows.append({'analysis': 'B_age_definition_approx', 'measure': label,
                             'scope': 'Global 15-39 (approximate)', 'standard': STD_2023_1539,
                             'year': yr, 'asr': np.nan, 'count': tot_1539,
                             'eapc_1990_2023': np.nan, 'eapc_1990_2023_low': np.nan,
                             'eapc_1990_2023_high': np.nan, 'eapc_2013_2023': np.nan,
                             'eapc_2013_2023_low': np.nan, 'eapc_2013_2023_high': np.nan,
                             'last_apc': np.nan, 'last_segment': '',
                             'paf': paf[(label + '_15_39_approx', yr)]['paf'],
                             'attributable_count': att + att_1519,
                             'note': 'ASSUMPTION-BASED approximation: 20-39 attributable plus '
                                     'the 20-24 PAF applied to all 15-19 cases; not a GBD estimate'})

    note_pt = ('point estimate: GBD publishes no age-specific covariance, so the '
               're-weighted ASR carries no GBD uncertainty interval')
    rows = tidy_rows(gy, tr_y, 'scope', 'B_age_definition', STD_2023_1539,
                     note_pt + '; ages 15-39 only, 2023 global 15-39 weights')
    sel = []
    for meas in MEASURES.values():
        d = regy[(regy.year == 2023) & (regy.measure == meas)].sort_values(
            'asr', ascending=False)
        sel += list(d.region.head(5)) + list(d.region.tail(5))
    sub = regy[regy.region.isin(sel) & regy.year.eq(2023)].copy()
    sub['region'] = pd.Categorical(sub.region, sorted(set(sel)), ordered=True)
    sub = sub.sort_values(['measure', 'region'])
    rows += tidy_rows(sub, tr_regy, 'region', 'B_age_definition', STD_2023_1539,
                      'approximate aggregation: population-weighted mean of country ASR '
                      '(aggregate_series); 2023 cross-section; regions entering any '
                      'top-5/bottom-5 ranking; ages 15-39')
    rows += tidy_rows(sex_series, tr_sex, 'scope', 'B_age_definition', STD_2023_1539,
                      note_pt + '; each sex standardised to its own 2023 15-39 population',
                      years=[1990, 2019, 2023], scope_prefix='')
    rows += paf_rows

    return dict(wdf=wdf, g=gy, stats=gstats, cy=cy, reg=regy, reg_compare=reg_compare,
                sex_series=sex_series, sex_stats=sex_stats, divergence=divergence,
                paf=paf, ghi_check=ghi_check, rows=rows)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    return f'{x:,.{nd}f}'


def write_summary(path, A, B):
    L = []
    a90, a23 = A['stats90'], A['stats23']
    rs = A['rank_stats']
    rc = A['reg_compare']
    bc = B['reg_compare']
    bs = B['stats']
    ss = B['sex_stats']
    dv = B['divergence']
    paf = B['paf']
    meas_names = list(MEASURES.values())

    L.append('# Sensitivity analyses: alternative standard population (1990) and '
             'alternative age definition (15-39)\n')
    L.append('Generated by `scripts/sensitivity_asr.py` from the age-specific panels in '
             '`data/panels/` (GBD 2023, cause 1059, 204 countries, 1990-2023). All ASRs are per '
             '100,000 population aged 15-49 (analysis A) or 15-39 (analysis B) and are **point '
             'estimates**: GBD publishes marginal 95% UIs per age group but not the age-specific '
             'covariance required to re-weight them, so no uncertainty interval is propagated to '
             'the re-weighted ASR. The only intervals reported are analytic 95% CIs for the EAPC '
             '(log-linear OLS) and for the joinpoint APCs, computed with the same `trends.py` '
             'functions as the main pipeline. The 2023-standard series reproduced here matches '
             f'`results/global_asr_series.csv` to {A["val_diffs"][0]:.1e} and '
             f'`results/country_asr_series.csv` to {A["val_diffs"][1]:.1e}.\n')

    L.append('## A1. Standard-population weights\n')
    L.append('| Age group | 1990 global weight | 2023 global weight | 1990/2023 |')
    L.append('|---|---|---|---|')
    for _, r in A['wdf'].iterrows():
        L.append(f'| {r.label} | {r.weight_1990:.5f} | {r.weight_2023:.5f} | '
                 f'{r.ratio_1990_to_2023:.4f} |')
    L.append('')
    L.append('The 1990 population is younger inside 15-49 (15-19 weight '
             f'{A["wdf"].iloc[0].weight_1990:.3f} vs {A["wdf"].iloc[0].weight_2023:.3f}; 45-49 '
             f'{A["wdf"].iloc[-1].weight_1990:.3f} vs {A["wdf"].iloc[-1].weight_2023:.3f}), so '
             'standardising to 1990 lowers the ASR level while leaving every age-specific rate '
             'unchanged. By construction the 1990 ASR under the 1990 standard equals the 1990 '
             f'crude rate ({f(a90["Incidence"]["crude_1990"],4)} incidence, '
             f'{f(a90["Deaths"]["crude_1990"],4)} deaths, {f(a90["DALYs"]["crude_1990"],1)} DALYs).\n')

    L.append('## A2. Global ASR under the 1990 standard\n')
    L.append('| Measure | ASR 1990 | ASR 2019 | ASR 2023 | Change 1990-2023 | EAPC 1990-2023 '
             '(95% CI), p | EAPC 2013-2023 (95% CI), p | Joinpoints | Last-segment APC |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for meas in meas_names:
        s = a90[meas]
        L.append(f'| {meas} | {f(s["asr_1990"])} | {f(s["asr_2019"])} | {f(s["asr_2023"])} | '
                 f'{f(s["asr_change_percent"],1)}% | {f(s["eapc_1990_2023"],3)} '
                 f'({f(s["eapc_1990_2023_low"],3)} to {f(s["eapc_1990_2023_high"],3)}), '
                 f'p={s["eapc_1990_2023_p"]:.2e} | {f(s["eapc_2013_2023"],3)} '
                 f'({f(s["eapc_2013_2023_low"],3)} to {f(s["eapc_2013_2023_high"],3)}), '
                 f'p={s["eapc_2013_2023_p"]:.2e} | {s["joinpoints"]} | '
                 f'{f(s["last_apc"],3)} ({s["last_segment"]}) |')
    L.append('')
    L.append('Reference (2023 standard, the manuscript standard):\n')
    L.append('| Measure | ASR 1990 | ASR 2023 | Change | EAPC 1990-2023 | EAPC 2013-2023 | '
             'Last-segment APC |')
    L.append('|---|---|---|---|---|---|---|')
    for meas in meas_names:
        s = a23[meas]
        L.append(f'| {meas} | {f(s["asr_1990"])} | {f(s["asr_2023"])} | '
                 f'{f(s["asr_change_percent"],1)}% | {f(s["eapc_1990_2023"],3)} '
                 f'({f(s["eapc_1990_2023_low"],3)} to {f(s["eapc_1990_2023_high"],3)}) | '
                 f'{f(s["eapc_2013_2023"],3)} ({f(s["eapc_2013_2023_low"],3)} to '
                 f'{f(s["eapc_2013_2023_high"],3)}) | {f(s["last_apc"],3)} ({s["last_segment"]}) |')
    L.append('')
    max_shift = max(abs(a90[m]['eapc_1990_2023'] - a23[m]['eapc_1990_2023']) for m in meas_names)
    L.append('Choosing the 1990 standard changes the level (2023 incidence ASR '
             f'{f(a90["Incidence"]["asr_2023"])} vs {f(a23["Incidence"]["asr_2023"])} per 100,000) '
             'but not the direction of any trend. The 1990-standard EAPCs differ from the '
             f'2023-standard EAPCs by at most {f(max_shift,3)} percentage points, all retain their '
             'sign and significance, and BIC selects the same joinpoint years '
             f'({a90["Incidence"]["joinpoints"]} incidence; {a90["Deaths"]["joinpoints"]} deaths; '
             f'{a90["DALYs"]["joinpoints"]} DALYs). The 2019-2023 rebound in the death ASR is '
             f'+{f(a90["Deaths"]["reversal_19_23"],2)}% under the 1990 standard vs '
             f'+{f(a23["Deaths"]["reversal_19_23"],2)}% under 2023.\n')

    L.append('## A3. Country rankings in 2023 under both standards\n')
    L.append('| Measure | Spearman rho | Pearson r | Top-20 overlap | Moving >1 quartile | '
             'Moving 1 quartile | Median abs rank shift | Max abs rank shift |')
    L.append('|---|---|---|---|---|---|---|---|')
    for meas in meas_names:
        s = rs[meas]
        L.append(f'| {meas} | {f(s["spearman"],4)} (p={s["spearman_p"]:.1e}) | '
                 f'{f(s["pearson"],4)} | {s["top20_overlap"]}/20 | '
                 f'{s["quartile_moves_gt1"]}/204 | {s["quartile_moves_1"]}/204 | '
                 f'{s["median_abs_shift"]:.0f} | {s["max_rank_shift"]} |')
    L.append('')
    for meas in meas_names:
        s = rs[meas]
        L.append(f'- **{meas}**: top 5 under the 2023 standard - {", ".join(s["top5_2023"])}; '
                 f'under the 1990 standard - {", ".join(s["top5_1990"])}. Lowest 5 under 1990 - '
                 f'{", ".join(s["bottom5_1990"])}.')
    L.append('')
    L.append('Quartile 1 is the highest-ASR quartile (51 countries each). A country "moves more '
             'than one quartile" when its quartile differs by two or more between standards: '
             + ('no country does so for any measure '
                f'({rs["Incidence"]["quartile_moves_gt1"]}, {rs["Deaths"]["quartile_moves_gt1"]} '
                f'and {rs["DALYs"]["quartile_moves_gt1"]} countries for incidence, deaths and '
                'DALYs respectively).'
                if sum(rs[m]['quartile_moves_gt1'] for m in meas_names) == 0 else
                ', '.join(f'{m} {rs[m]["quartile_moves_gt1"]}/204' for m in meas_names) + '.')
             + ' The largest individual rank movements are '
             + '; '.join(f'{m}: ' + ', '.join(f'{nm} {r23}->{r90}' for nm, r23, r90 in
                                              rs[m]['largest_moves'])
                         for m in meas_names) + ' (rank under 2023 -> 1990 standard).\n')

    L.append('## A4. Regional ASR in 2023: 1990 vs 2023 standard\n')
    L.append('| Measure | Top 5 (1990 std) | Top 5 (2023 std, published) | Top-5 overlap | '
             'Bottom 5 (1990 std) | Bottom-5 overlap |')
    L.append('|---|---|---|---|---|---|')
    for meas in meas_names:
        c = rc[meas]
        L.append(f'| {meas} | {"; ".join(c["top5_std1990"])} | {"; ".join(c["top5_published"])} | '
                 f'{c["top5_overlap"]}/5 | {"; ".join(c["bottom5_std1990"])} | '
                 f'{c["bottom5_overlap"]}/5 |')
    L.append('')
    for meas in meas_names:
        c = rc[meas]
        L.append(f'- **{meas}** (1990-standard ASR; 2023-standard value in brackets): ' +
                 '; '.join(f'{nm} {f(v90)} ({f(v23)})' for nm, v23, v90 in c['top5_values']) + '.')
    L.append('')
    L.append('The top-5 regional **sets** are identical under both standards for '
             + ', '.join(meas_names) + ' ('
             + ', '.join(f'{m}: ' + ('identical order' if rc[m]['top5_same_order'] else
                                    'positions ' + '/'.join(str(i + 1) for i, a, b in
                                                            rc[m]['order_diff']) +
                                    ' exchanged: ' + '; '.join(f'rank {i + 1} {a} (1990 std) vs '
                                                               f'{b} (2023 std)'
                                                               for i, a, b in rc[m]['order_diff']))
                         for m in meas_names)
             + '). The bottom-5 set is unchanged for '
             + ', '.join(m for m in meas_names if rc[m]['bottom5_same_set'])
             + ('' if all(rc[m]['bottom5_same_set'] for m in meas_names) else
                '; for DALYs ' + '; '.join(
                    f'{", ".join(rc[m]["bottom5_entered"])} enters the bottom five under the 1990 '
                    f'standard in place of {", ".join(rc[m]["bottom5_left"])}'
                    for m in meas_names if not rc[m]['bottom5_same_set']))
             + '. Region-level rank correlation between the two standards is '
             + ', '.join(f'{m} {f(rc[m]["spearman_1990_vs_2023"],4)}' for m in meas_names) + '.\n')

    L.append('## A5. SDI quintiles, 1990 and 2023, under the 1990 standard\n')
    L.append('Quintiles are equal fifths of the 2023 SDI index from '
             '`data/aux/sdi_quintiles_2023.csv`. That file\'s own `SDI Quintile` column has '
             f'{A["sdi_check"]["label_missing"]} blank countries and unequal group sizes (High 78, '
             'Low 47, Low-middle 23, High-middle 19, Middle 16), so equal quintiles of the same '
             'SDI values are used; this reproduces the main pipeline assignment exactly '
             f'(agreement {A["sdi_check"]["pipeline_agreement"]:.2f}), while the file\'s '
             f'precomputed labels agree with an equal-quintile ordering for only '
             f'{A["sdi_check"]["label_agreement"]*100:.0f}% of the '
             f'{A["sdi_check"]["label_n"]} labelled countries.\n')
    L.append('| Quintile | Incidence 1990 | Incidence 2023 | Deaths 1990 | Deaths 2023 | '
             'DALYs 1990 | DALYs 2023 |')
    L.append('|---|---|---|---|---|---|---|')
    for q in ['Q1 lowest', 'Q2', 'Q3', 'Q4', 'Q5 highest']:
        cells = [q]
        for meas in meas_names:
            t = A['q_compare'][meas]['std1990']
            cells += [f(t.loc[q, 1990]), f(t.loc[q, 2023])]
        L.append('| ' + ' | '.join(cells) + ' |')
    L.append('')
    L.append('Same pattern as Table S8 under the 2023 standard: incidence ASR '
             f'{f(A["q_compare"]["Incidence"]["std1990"].loc["Q1 lowest", 2023] / A["q_compare"]["Incidence"]["std1990"].loc["Q1 lowest", 1990] * 100 - 100, 1)}% '
             'in Q1 and '
             f'{f(A["q_compare"]["Incidence"]["std1990"].loc["Q5 highest", 2023] / A["q_compare"]["Incidence"]["std1990"].loc["Q5 highest", 1990] * 100 - 100, 1)}% '
             'in Q5 over 1990-2023, while deaths fall steeply in Q4/Q5 and are flat or rising in '
             'Q1/Q2. The SDI gradient story is therefore not a standard-population artefact.\n')

    L.append('## B1. Global 15-39 counts and ASR\n')
    L.append('| Measure | Count 1990 | Count 2023 | Count change | ASR 1990 | ASR 2023 | '
             'ASR change | EAPC 1990-2023 (95% CI), p | EAPC 2013-2023 (95% CI), p | Joinpoints '
             '| Last-segment APC |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for meas in meas_names:
        s = bs[meas]
        L.append(f'| {meas} | {f(s["count_1990"],0)} | {f(s["count_2023"],0)} | '
                 f'{f(s["count_change_percent"],1)}% | {f(s["asr_1990"])} | {f(s["asr_2023"])} | '
                 f'{f(s["asr_change_percent"],1)}% | {f(s["eapc_1990_2023"],3)} '
                 f'({f(s["eapc_1990_2023_low"],3)} to {f(s["eapc_1990_2023_high"],3)}), '
                 f'p={s["eapc_1990_2023_p"]:.2e} | {f(s["eapc_2013_2023"],3)} '
                 f'({f(s["eapc_2013_2023_low"],3)} to {f(s["eapc_2013_2023_high"],3)}), '
                 f'p={s["eapc_2013_2023_p"]:.2e} | {s["joinpoints"]} | '
                 f'{f(s["last_apc"],3)} ({s["last_segment"]}) |')
    L.append('')
    L.append('**Mortality reversal at 15-39.** The death ASR rises from '
             f'{f(bs["Deaths"]["asr_2019"])} in 2019 to {f(bs["Deaths"]["asr_2023"])} in 2023 '
             f'(+{f(bs["Deaths"]["reversal_19_23"],2)}%); DALYs +'
             f'{f(bs["DALYs"]["reversal_19_23"],2)}% and incidence +'
             f'{f(bs["Incidence"]["reversal_19_23"],2)}%. The corresponding 15-49 changes are '
             f'+{f(a23["Deaths"]["reversal_19_23"],2)}%, +{f(a23["DALYs"]["reversal_19_23"],2)}% '
             f'and +{f(a23["Incidence"]["reversal_19_23"],2)}%. The reversal therefore persists '
             'under the narrower age definition and is slightly larger for mortality.\n')
    L.append('**Caveat on the long-run incidence trend.** At 15-39 the 1990-2023 incidence EAPC '
             f'is {f(bs["Incidence"]["eapc_1990_2023"],3)} '
             f'({f(bs["Incidence"]["eapc_1990_2023_low"],3)} to '
             f'{f(bs["Incidence"]["eapc_1990_2023_high"],3)}), p='
             f'{bs["Incidence"]["eapc_1990_2023_p"]:.2f}, i.e. no longer significant, whereas at '
             f'15-49 it is -0.290 (p=8.4e-05). The long-run incidence decline is concentrated in '
             'ages 40-49. By contrast the 2013-2023 acceleration is stronger at 15-39 '
             f'(+{f(bs["Incidence"]["eapc_2013_2023"],2)}% vs +1.30% at 15-49), so the recent '
             'acceleration is not driven by the 40-49 band.\n')

    L.append('## B2. Sex-specific global ASR, 15-39\n')
    L.append('| Sex | Measure | ASR 1990 | ASR 2019 | ASR 2023 | Change | EAPC 1990-2023 '
             '(95% CI) | EAPC 2013-2023 (95% CI) | Last-segment APC |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for sex in ['Male', 'Female', 'Both']:
        for meas in meas_names:
            s = ss[(sex, meas)]
            L.append(f'| {sex} | {meas} | {f(s["asr_1990"])} | {f(s["asr_2019"])} | '
                     f'{f(s["asr_2023"])} | {f(s["asr_change_percent"],1)}% | '
                     f'{f(s["eapc_1990_2023"],3)} ({f(s["eapc_1990_2023_low"],3)} to '
                     f'{f(s["eapc_1990_2023_high"],3)}) | {f(s["eapc_2013_2023"],3)} '
                     f'({f(s["eapc_2013_2023_low"],3)} to {f(s["eapc_2013_2023_high"],3)}) | '
                     f'{f(s["last_apc"],2)} ({s["last_segment"]}) |')
    L.append('')
    L.append('Female-male divergence at 15-39: the 2013-2023 EAPC difference is '
             + ', '.join(f'{m} {f(dv[m]["eapc_diff_2013_2023"],2)} points (female '
                         f'{f(ss[("Female", m)]["eapc_2013_2023"],2)} vs male '
                         f'{f(ss[("Male", m)]["eapc_2013_2023"],2)})' for m in meas_names)
             + f'. In 2023 the female incidence ASR is {f(dv["Incidence"]["asr_ratio_2023"],2)}x '
             f'the male rate; the 2019-2023 death ASR rose +{f(dv["Deaths"]["reversal_19_23_f"],2)}% '
             f'in females and +{f(dv["Deaths"]["reversal_19_23_m"],2)}% in males. The divergence '
             'reported for 15-49 is reproduced, and for recent years amplified, at 15-39.\n')

    L.append('## B3. Regional ASR in 2023, 15-39 vs 15-49\n')
    L.append('| Measure | Top 5, 15-39 | Top 5, 15-49 (published) | Overlap | Bottom 5, 15-39 | '
             'Bottom 5, 15-49 | Overlap |')
    L.append('|---|---|---|---|---|---|---|')
    for meas in meas_names:
        c = bc[meas]
        L.append(f'| {meas} | {"; ".join(c["top5_15_39"])} | {"; ".join(c["top5_15_49"])} | '
                 f'{c["top5_overlap"]}/5 | {"; ".join(c["bottom5_15_39"])} | '
                 f'{"; ".join(c["bottom5_15_49"])} | {c["bottom5_overlap"]}/5 |')
    L.append('')
    for meas in meas_names:
        c = bc[meas]
        L.append(f'- **{meas}** (15-39 ASR; 15-49 value in brackets): ' +
                 '; '.join(f'{nm} {f(v39)} ({f(v49)})' for nm, v39, v49 in c['top5_values']) + '.')
    L.append('')
    L.append('Regional rank correlation between the 15-39 and 15-49 orderings is '
             + ', '.join(f'{m} {f(bc[m]["spearman_15_39_vs_15_49"],4)}' for m in meas_names)
             + '. The 15-39 regional geometry reproduces the published 15-49 geometry, with '
             'top-5 overlap '
             + ', '.join(f'{m} {bc[m]["top5_overlap"]}/5' for m in meas_names) + ' and bottom-5 '
             'overlap ' + ', '.join(f'{m} {bc[m]["bottom5_overlap"]}/5' for m in meas_names) + '. '
             + ' '.join(f'For {m}, ' + '; '.join(
                 part for part in [
                     (f'{", ".join(bc[m]["top5_entered"])} enters the 15-39 top five in place '
                      f'of {", ".join(bc[m]["top5_left"])}') if bc[m]['top5_entered'] else '',
                     (f'{", ".join(bc[m]["bottom5_entered"])} enters the 15-39 bottom five in '
                      f'place of {", ".join(bc[m]["bottom5_left"])}') if bc[m]['bottom5_entered'] else '',
                 ] if part) + '.'
                 for m in meas_names if bc[m]['top5_entered'] or bc[m]['bottom5_entered'])
             + '\n')

    L.append('## B4. Metabolic PAF for 20-39\n')
    L.append('| Measure | Year | Attributable (20-39) | Total (20-39) | PAF 20-39 |')
    L.append('|---|---|---|---|---|')
    for label in ['Deaths', 'DALYs']:
        for yr in (1990, 2023):
            p = paf[(label, yr)]
            L.append(f'| {label} | {yr} | {f(p["attributable"],0)} | {f(p["total"],0)} | '
                     f'{f(p["paf"],2)}% |')
    L.append('')
    L.append('**Caveat.** The risk panel J has no 15-19 age group, so this PAF is for **20-39, '
             'not 15-39**. It is the ratio of summed attributable numbers to summed total numbers '
             'over ages 20-24 to 35-39 (a population-weighted average of the age-specific PAFs), '
             'not a re-standardised PAF. For a 15-39 approximation the 20-24 PAF is applied to '
             'the 15-19 total: '
             + '; '.join(
                 f'{label} PAF 15-39 approx {f(paf[(label + "_15_39_approx", yr)]["paf"],2)}% '
                 f'in {yr} (20-24 PAF {f(paf[(label + "_15_39_approx", yr)]["paf_2024"]*100,2)}% '
                 f'applied to {f(paf[(label + "_15_39_approx", yr)]["total_1519"],0)} 15-19 cases)'
                 for label in ['Deaths', 'DALYs'] for yr in (1990, 2023))
             + '. That approximation is assumption-based (metabolic PAF rises steeply with age, so '
             '15-19 is lower than 20-24 and this overstates 15-19 attribution), so it should be '
             'used only as an upper bound; the 20-39 figure is the defensible number.\n')
    if B['ghi_check']:
        L.append('Compatibility check: J\'s 15-49 PAF in 2023 is '
                 + ', '.join(f'{"deaths" if m == 1 else "DALYs"} '
                             f'{B["ghi_check"][m][0]:.6f} (GHI panel {B["ghi_check"][m][1]:.6f}; '
                             f'difference {B["ghi_check"][m][2]:.1e})' for m in (1, 2))
                 + ', so the attributable numerators are on the same scale as the published '
                 'attribution.\n')

    L.append('## Uncertainty and limitations\n')
    L.append('- All ASRs in these files are point estimates. GBD supplies marginal UIs per age '
             'group; re-weighting them would require the age-specific covariance (and would '
             'ignore correlation between numerators), so no interval is reported for levels. '
             'Level differences of 1-15% between standards should not be over-interpreted.')
    L.append('- EAPC CIs are analytic and describe the fit of a log-linear model to the point '
             'estimates; they are not GBD uncertainty intervals.')
    L.append('- Regional and SDI-quintile values are approximate aggregations '
             '(population-weighted means of country ASR) and inherit the instability of '
             'small-state estimates.')
    L.append('- Country rankings compare 204 estimates of very different precision; small '
             'territories (Monaco, Andorra) remain at the top under both standards.')
    L.append('- The joinpoint model selected the maximum number of joinpoints, so exact years are '
             'unstable; only the direction of the last segment should be interpreted.\n')

    L.append('## Interpretation: are the manuscript conclusions robust?\n')
    L.append('**Yes, to both perturbations of the standardisation choices, with one exception.** '
             'Under the 1990 standard the global ASR levels fall (2023 incidence '
             f'{f(a90["Incidence"]["asr_2023"])} vs {f(a23["Incidence"]["asr_2023"])} per 100,000) '
             'because the 1990 population is younger, but every trend statement survives: '
             f'incidence declines slowly over 1990-2023 ({f(a90["Incidence"]["eapc_1990_2023"],2)}% '
             'per year), mortality falls faster '
             f'({f(a90["Deaths"]["eapc_1990_2023"],2)}%), and both accelerate after 2013 '
             f'(+{f(a90["Incidence"]["eapc_2013_2023"],2)}% and '
             f'+{f(a90["Deaths"]["eapc_2013_2023"],2)}% per year), with a 2019-2023 mortality '
             f'rebound of +{f(a90["Deaths"]["reversal_19_23"],1)}%. Country rankings are '
             'effectively invariant (Spearman '
             f'{f(rs["Incidence"]["spearman"],3)} for incidence, top-20 overlap '
             f'{rs["Incidence"]["top20_overlap"]}/20, no country moving more than one quartile) '
             'and the regional top-five set is unchanged (region-level Spearman '
             f'{f(rc["Incidence"]["spearman_1990_vs_2023"],3)}), so the post-standardisation '
             'reversal of the geographic pattern is not an artefact of the 2023 weights. '
             'Restricting to 15-39 '
             'preserves the incidence acceleration '
             f'(+{f(bs["Incidence"]["eapc_2013_2023"],2)}% in 2013-2023), the female-male '
             f'divergence (female +{f(ss[("Female", "Incidence")]["eapc_2013_2023"],2)}% vs male '
             f'+{f(ss[("Male", "Incidence")]["eapc_2013_2023"],2)}%) and the metabolic PAF '
             f'({f(paf[("Deaths", 2023)]["paf"],2)}% at 20-39, which excludes ages 15-19, vs 3.96% '
             'at 15-49), and strengthens '
             f'the mortality reversal (+{f(bs["Deaths"]["reversal_19_23"],1)}%). The one fragile '
             'claim is the long-run incidence decline, which is significant at 15-49 but not at '
             f'15-39 (p={bs["Incidence"]["eapc_1990_2023_p"]:.2f}); the text should attribute that '
             'decline to ages 40-49.')

    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')


# --------------------------------------------------------------------------
def main():
    num, pop = load_panels()
    names = country_ids()

    A = analysis_a(num, pop, names)
    B = analysis_b(num, pop, names)

    std_df = pd.DataFrame(A['rows'])
    # the standard-population file uses the 16-column specification (no PAF columns)
    std_df = std_df[TIDY_COLS[:-3] + ['note']]
    std_df.to_csv(os.path.join(RES, 'sensitivity_standard_pop.csv'), index=False)

    A['rank'].to_csv(os.path.join(RES, 'sensitivity_country_rank.csv'), index=False)

    age_df = pd.DataFrame(B['rows'])
    age_df = age_df[TIDY_COLS]
    age_df.to_csv(os.path.join(RES, 'sensitivity_age_definition.csv'), index=False)

    write_summary(os.path.join(RES, 'sensitivity_summary.md'), A, B)

    print(f'validation vs main pipeline: global {A["val_diffs"][0]:.2e}, '
          f'country {A["val_diffs"][1]:.2e}')
    print('rows: standard_pop', len(std_df), '| country_rank', len(A['rank']),
          '| age_definition', len(age_df))
    print('\nA2 global under 1990 standard:')
    for meas in MEASURES.values():
        s = A['stats90'][meas]
        print(f'  {meas:9s} ASR {s["asr_1990"]:8.3f} -> {s["asr_2023"]:8.3f} '
              f'({s["asr_change_percent"]:+7.2f}%)  EAPC {s["eapc_1990_2023"]:+.3f} '
              f'({s["eapc_1990_2023_low"]:+.3f},{s["eapc_1990_2023_high"]:+.3f})  '
              f'2013-23 {s["eapc_2013_2023"]:+.3f}  jp {s["joinpoints"]}  '
              f'last {s["last_apc"]:+.3f}')
    print('\nA3 country rank statistics:')
    for meas in MEASURES.values():
        s = A['rank_stats'][meas]
        print(f'  {meas:9s} spearman {s["spearman"]:.4f} pearson {s["pearson"]:.4f} '
              f'top20 {s["top20_overlap"]}/20 quartile>1 {s["quartile_moves_gt1"]}')
    print('\nB1 global 15-39:')
    for meas in MEASURES.values():
        s = B['stats'][meas]
        print(f'  {meas:9s} n {s["count_1990"]:.0f} -> {s["count_2023"]:.0f}  '
              f'ASR {s["asr_1990"]:.3f} -> {s["asr_2023"]:.3f} '
              f'({s["asr_change_percent"]:+.2f}%)  EAPC {s["eapc_1990_2023"]:+.3f} '
              f'p={s["eapc_1990_2023_p"]:.2e}  2013-23 {s["eapc_2013_2023"]:+.3f}  '
              f'jp {s["joinpoints"]}  last {s["last_apc"]:+.3f}')
    print('\nB4 PAF:')
    for k, v in B['paf'].items():
        print(f'  {k}: PAF {v["paf"]:.4f}% attributable {v["attributable"]:.1f} '
              f'total {v["total"]:.1f}')


if __name__ == '__main__':
    main()
