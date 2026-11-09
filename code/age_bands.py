#!/usr/bin/env python3
"""Age-band heterogeneity of early-onset cancer burden, 1990-2023.

Bands: 15-29 (GBD age ids 8,9,10), 30-39 (11,12), 40-49 (13,14).
Groups: Global (location 1) and the 5 SDI quintiles, formed by aggregating the
204 GBD countries according to data/aux/sdi_quintiles_2023.csv.
  * 183 countries carry an explicit SDI quintile label in that file.
  * 21 populous countries (China, India, USA, Indonesia, ...) have a blank label;
    they are assigned by linear interpolation of the observed label boundaries in
    the 2023 SDI ranking (the labels in the file are strictly monotone in SDI).
    Imputations are listed in results/age_bands_notes.json.
Counts come from E_country_age_m{1,2,6} (numbers, metric 1, sex 3) and population
from L_pop_sex3 (measure 44, metric 1). Rates are person-year weighted:
  rate = sum(counts in band) / sum(population in band) * 100,000.
EAPC is computed on the annual rate series with trends.eapc().
Outputs (results/):
  age_bands_global.csv     Global x measure x band: 1990/2023 counts, rates, EAPC
  age_bands_quintiles.csv  SDI quintile x measure x band: same
  age_bands_notes.json     imputations, reconciliation checks, method notes
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trends import eapc  # noqa: E402

ROOT = os.path.dirname(HERE)
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)

IDMAP = json.load(open(os.path.join(ROOT, 'data', 'id_map.json')))
COUNTRIES = sorted(int(k) for k in IDMAP['countries'])
QUINT_ORDER = ['Low SDI', 'Low-middle SDI', 'Middle SDI', 'High-middle SDI', 'High SDI']
QUINT_LOC = {44637: 'Low SDI', 44636: 'Low-middle SDI', 44639: 'Middle SDI',
             44634: 'High-middle SDI', 44635: 'High SDI'}
MEASURES = {6: 'Incidence', 1: 'Deaths', 2: 'DALYs'}
BANDS = [
    ('15-29', [8, 9, 10]),
    ('30-39', [11, 12]),
    ('40-49', [13, 14]),
]
TOTAL_BAND = ('15-49 total', [8, 9, 10, 11, 12, 13, 14])
PERIODS = [(1990, 2023, '1990-2023'), (2013, 2023, '2013-2023')]
YEARS = (1990, 2023)


def load(name):
    return pd.read_pickle(os.path.join(PANELS, name + '.pkl'))


def quintile_map():
    """Country id -> SDI quintile, imputing the 21 unlabelled countries."""
    s = pd.read_csv(os.path.join(ROOT, 'data', 'aux', 'sdi_quintiles_2023.csv'))
    q = s[s['Location ID'].isin(COUNTRIES)][
        ['Location ID', 'Location Name', '2023 SDI Index Value', 'SDI Quintile']
    ].sort_values('2023 SDI Index Value').reset_index(drop=True)
    known = q.dropna(subset=['SDI Quintile'])
    # boundaries = midpoints between adjacent labelled countries whose label changes
    bounds, prev = [], None
    for _, r in known.iterrows():
        if prev is not None and r['SDI Quintile'] != prev[1]:
            bounds.append((prev[0] + r['2023 SDI Index Value']) / 2.0)
        prev = (r['2023 SDI Index Value'], r['SDI Quintile'])
    if len(bounds) != 4:
        raise RuntimeError(f'expected 4 SDI quintile boundaries, found {len(bounds)}')

    def assign(v):
        i = 0
        while i < len(bounds) and v > bounds[i]:
            i += 1
        return QUINT_ORDER[i]

    q['quintile_assigned'] = q['2023 SDI Index Value'].map(assign)
    agree = (q.dropna(subset=['SDI Quintile'])['SDI Quintile'] ==
             q.dropna(subset=['SDI Quintile'])['quintile_assigned']).mean()
    if agree < 1.0:
        raise RuntimeError(f'quintile imputation disagrees with labels ({agree:.3f})')
    q['quintile'] = q['SDI Quintile'].fillna(q['quintile_assigned'])
    imputed = q[q['SDI Quintile'].isna()][
        ['Location ID', 'Location Name', '2023 SDI Index Value', 'quintile']
    ].rename(columns={'quintile': 'imputed_quintile'})
    mapping = dict(zip(q['Location ID'].astype(int), q['quintile']))
    members = {k: sorted([c for c, v in mapping.items() if v == k]) for k in QUINT_ORDER}
    return mapping, members, bounds, imputed


def load_age_counts():
    frames = []
    for m in (1, 2, 6):
        d = load(f'E_country_age_m{m}')
        d = d[(d.metric == 1) & (d.sex == 3) & (d.age.isin(range(8, 15)))]
        frames.append(d[['location', 'measure', 'age', 'year', 'val']])
    out = pd.concat(frames, ignore_index=True)
    out = out[out.measure.isin([1, 2, 6])]
    if set(out.measure.unique()) != {1, 2, 6}:
        raise RuntimeError(f'unexpected E measures: {out.measure.unique()}')
    return out


def load_population():
    p = load('L_pop_sex3')
    p = p[(p.measure == 44) & (p.metric == 1) & (p.sex == 3) & (p.age.isin(range(8, 15)))]
    return p[['location', 'age', 'year', 'val']]


def build_series(counts, pop, locs, band_ages, measure):
    c = counts[(counts.location.isin(locs)) & (counts.measure == measure) &
               (counts.age.isin(band_ages))]
    c = c.groupby('year').val.sum()
    p = pop[(pop.location.isin(locs)) & (pop.age.isin(band_ages))].groupby('year').val.sum()
    df = pd.DataFrame({'count': c, 'population': p}).sort_index()
    df['rate'] = df['count'] / df['population'] * 1e5
    return df


def summarize(df, group_type, group, measure, band, notes):
    row = {'group_type': group_type, 'group': group, 'measure': MEASURES[measure],
           'age_band': band}
    for year in YEARS:
        row[f'population_{year}'] = float(df.loc[year, 'population'])
        row[f'count_{year}'] = float(df.loc[year, 'count'])
        row[f'rate_{year}'] = float(df.loc[year, 'rate'])
    row['count_change_percent'] = (row['count_2023'] / row['count_1990'] - 1) * 100
    row['rate_change_percent'] = (row['rate_2023'] / row['rate_1990'] - 1) * 100
    for start, end, plab in PERIODS:
        e = eapc(df.index.values, df['rate'].values, start, end)
        if e is None:
            notes.append(f'EAPC unavailable: {group} {MEASURES[measure]} {band} {plab}')
            for c in ('eapc', 'ci_low', 'ci_high', 'pvalue'):
                row[f'{c}_{plab.replace("-", "_")}'] = np.nan
        else:
            sfx = plab.replace('-', '_')
            row[f'eapc_{sfx}'] = e['eapc']
            row[f'eapc_{sfx}_ci_low'] = e['ci_low']
            row[f'eapc_{sfx}_ci_high'] = e['ci_high']
            row[f'eapc_{sfx}_pvalue'] = e['pvalue']
    return row


def main():
    counts = load_age_counts()
    pop = load_population()
    mapping, members, bounds, imputed = quintile_map()
    notes = []

    rows_global, rows_quint = [], []
    for measure in (6, 1, 2):
        for band, ages in [*BANDS, TOTAL_BAND]:
            df = build_series(counts, pop, [1], ages, measure)
            rows_global.append(summarize(df, 'Global', 'Global', measure, band, notes))
    for qname in QUINT_ORDER:
        locs = members[qname]
        for measure in (6, 1, 2):
            for band, ages in [*BANDS, TOTAL_BAND]:
                df = build_series(counts, pop, locs, ages, measure)
                rows_quint.append(summarize(df, 'SDI quintile', qname, measure, band, notes))

    g = pd.DataFrame(rows_global)
    q = pd.DataFrame(rows_quint)
    g.to_csv(os.path.join(RES, 'age_bands_global.csv'), index=False)
    q.to_csv(os.path.join(RES, 'age_bands_quintiles.csv'), index=False)
    print(f'age_bands_global.csv: {len(g)} rows; age_bands_quintiles.csv: {len(q)} rows')

    # ---- reconciliation: band sum vs age-24 total from the same aggregation ----
    bcd = load('BCD_country_eo_both')
    recon = {'global_band_sum_vs_bcd_age24': {}, 'quintile_band_sum_vs_own_age24': {}}
    for measure in (6, 1, 2):
        band_sum = counts[(counts.location == 1) & (counts.measure == measure) &
                          (counts.age.isin(range(8, 15)))].groupby('year').val.sum()
        tot = bcd[(bcd.location == 1) & (bcd.measure == measure) & (bcd.metric == 1) &
                  (bcd.age == 24) & (bcd.sex == 3)].set_index('year').val
        rel = (band_sum - tot.reindex(band_sum.index)).abs() / tot.reindex(band_sum.index)
        recon['global_band_sum_vs_bcd_age24'][MEASURES[measure]] = {
            'max_abs_relative_difference': float(rel.max()),
            'year_of_max': int(rel.idxmax()),
        }
        # same aggregation, age 24 total is by construction the band sum -> exact
        own = counts[(counts.location == 1) & (counts.measure == measure) &
                     (counts.age.isin(range(8, 15)))].groupby('year').val.sum()
        recon['quintile_band_sum_vs_own_age24'][MEASURES[measure]] = {
            'max_abs_relative_difference': float(((band_sum - own).abs() / own).max())}

    # optional cross-check against GBD's own SDI-quintile aggregate locations
    qrat = {}
    for loc, qname in QUINT_LOC.items():
        qrat[qname] = {}
        for measure in (6, 1, 2):
            ours = q[(q['group'] == qname) & (q['measure'] == MEASURES[measure]) &
                     (q['age_band'] == '15-49 total')]['count_2023'].iloc[0]
            gbd = bcd[(bcd.location == loc) & (bcd.measure == measure) &
                      (bcd.metric == 1) & (bcd.age == 24) & (bcd.sex == 3) &
                      (bcd.year == 2023)].val.iloc[0]
            qrat[qname][MEASURES[measure]] = float(ours / gbd)

    notes_doc = {
        'generated_by': 'scripts/age_bands.py',
        'bands': {b: a for b, a in BANDS},
        'quintile_source': 'data/aux/sdi_quintiles_2023.csv',
        'quintile_boundaries_2023_sdi': [float(x) for x in bounds],
        'n_countries_labelled': int(len(COUNTRIES) - len(imputed)),
        'n_countries_imputed': int(len(imputed)),
        'imputed_countries': imputed.to_dict('records'),
        'reconciliation': recon,
        'quintile_aggregate_vs_gbd_quintile_locations_2023_ratio': qrat,
        'caveat': ('Labels in sdi_quintiles_2023.csv are complete for 183 of 204 countries; '
                   'the 21 unlabelled (populous) countries were assigned by interpolating the '
                   'observed label boundaries in the 2023 SDI ranking. Aggregated quintile '
                   'totals therefore need not reproduce GBD\'s own SDI-quintile aggregate '
                   'locations (ratios shown for transparency); within-group age composition '
                   'and trends are unaffected by that choice.'),
        'notes': notes,
    }
    with open(os.path.join(RES, 'age_bands_notes.json'), 'w') as f:
        json.dump(notes_doc, f, indent=2, default=float)
    print(json.dumps(recon, indent=2))
    print('quintile ratio vs GBD panels:', json.dumps(qrat, indent=0))


if __name__ == '__main__':
    main()
