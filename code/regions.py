#!/usr/bin/env python3
"""Regional (21 GBD regions) early-onset cancer burden and metabolic attribution.

For the 21 GBD regions plus Global (location 1), both sexes, age 15-49 (age id 24):
  - 1990 and 2023 incidence / deaths / DALYs counts and crude 15-49 rates
  - 1990 and 2023 metabolic-attributable deaths / DALYs (risk 104): counts, rates,
    and PAF (deaths & DALYs; GHI metric 2 is a fraction, converted to percent)
  - 1990-2023 and 2013-2023 EAPC for every rate and PAF series
  - Joinpoint (BIC-selected segmented log-linear) last-segment APC for incidence
    and death rates, reusing trends.joinpoint()
Outputs (results/):
  regions_2023.csv        one row per region (+ Global) with 1990 & 2023 values
  regions_trends.csv      region x outcome x period EAPC / last-segment APC
  regions_key_findings.json  top/bottom regions and reconciliation checks
Inputs: BCD_country_eo_both.pkl (cause 1059), GHI_risk104_country_both.pkl (cause 410).
Note: the GHI panels stored on disk have already been re-based to the cause-1059
denominator by scripts/fix_paf.py, so PAF = attributable 1059 / total 1059.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trends import eapc, joinpoint  # noqa: E402

ROOT = os.path.dirname(HERE)
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)

IDMAP = json.load(open(os.path.join(ROOT, 'data', 'id_map.json')))
REGIONS = [(int(k), v) for k, v in IDMAP['regions'].items()]
MEASURES = {6: 'Incidence', 1: 'Deaths', 2: 'DALYs'}
ATTR_MEASURES = {1: 'Deaths', 2: 'DALYs'}
YEARS = (1990, 2023)
PERIODS = [(1990, 2023, '1990-2023'), (2013, 2023, '2013-2023')]


def load(name):
    return pd.read_pickle(os.path.join(PANELS, name + '.pkl'))


def series(df, loc, measure, metric, age=24):
    d = df[(df.location == loc) & (df.measure == measure) &
           (df.metric == metric) & (df.age == age) & (df.sex == 3)].sort_values('year')
    return d


def point(df, loc, measure, metric, year, age=24):
    d = series(df, loc, measure, metric, age)
    d = d[d.year == year]
    if not len(d):
        return np.nan, np.nan, np.nan
    r = d.iloc[0]
    return float(r.val), float(r.lower), float(r.upper)


def main():
    bcd = load('BCD_country_eo_both')
    risk = load('GHI_risk104_country_both')

    targets = [('Global', 1)] + [(name, loc) for loc, name in REGIONS]
    rows = []
    for name, loc in targets:
        row = {'location_id': loc, 'location': name,
               'level': 'Global' if loc == 1 else 'GBD region'}
        # ---- total burden ----
        for measure, mlab in MEASURES.items():
            for year in YEARS:
                n, nlo, nhi = point(bcd, loc, measure, 1, year)
                r, rlo, rhi = point(bcd, loc, measure, 3, year)
                row[f'{mlab.lower()}_number_{year}'] = n
                row[f'{mlab.lower()}_number_{year}_low'] = nlo
                row[f'{mlab.lower()}_number_{year}_high'] = nhi
                row[f'{mlab.lower()}_rate_{year}'] = r
                row[f'{mlab.lower()}_rate_{year}_low'] = rlo
                row[f'{mlab.lower()}_rate_{year}_high'] = rhi
        # population implied by incidence number / crude rate (per 100,000)
        for year in YEARS:
            n = row[f'incidence_number_{year}']
            r = row[f'incidence_rate_{year}']
            row[f'population_{year}'] = n / r * 1e5 if r and np.isfinite(r) and r > 0 else np.nan
        # ---- metabolic-attributable burden (risk 104) ----
        for measure, mlab in ATTR_MEASURES.items():
            key = f'attributable_{mlab.lower()}'
            for year in YEARS:
                n, nlo, nhi = point(risk, loc, measure, 1, year)
                r, rlo, rhi = point(risk, loc, measure, 3, year)
                p, plo, phi = point(risk, loc, measure, 2, year)  # fraction
                row[f'{key}_number_{year}'] = n
                row[f'{key}_number_{year}_low'] = nlo
                row[f'{key}_number_{year}_high'] = nhi
                row[f'{key}_rate_{year}'] = r
                row[f'{key}_rate_{year}_low'] = rlo
                row[f'{key}_rate_{year}_high'] = rhi
                row[f'paf_{mlab.lower()}_{year}'] = p * 100 if np.isfinite(p) else np.nan
                row[f'paf_{mlab.lower()}_{year}_low'] = plo * 100 if np.isfinite(plo) else np.nan
                row[f'paf_{mlab.lower()}_{year}_high'] = phi * 100 if np.isfinite(phi) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values('incidence_rate_2023', ascending=False, na_position='last').reset_index(drop=True)
    out.insert(0, 'rank_incidence_rate_2023', np.arange(1, len(out) + 1))
    out.to_csv(os.path.join(RES, 'regions_2023.csv'), index=False)
    print(f'regions_2023.csv: {len(out)} rows')

    # ---------------- trends ----------------
    # (outcome label, panel, measure id, metric id, measure label)
    outcomes = []
    for measure, mlab in MEASURES.items():
        outcomes.append((f'{mlab} rate', bcd, measure, 3, mlab))
    for measure, mlab in ATTR_MEASURES.items():
        outcomes.append((f'Attributable {mlab.lower()} rate', risk, measure, 3, mlab))
        outcomes.append((f'PAF {mlab.lower()}', risk, measure, 2, mlab))

    trows = []
    for name, loc in targets:
        for outcome, df, measure, metric, mlab in outcomes:
            d = series(df, loc, measure, metric, 24)
            if not len(d):
                continue
            vals = d.val.values * (100 if metric == 2 else 1)
            years = d.year.values
            for start, end, plab in PERIODS:
                e = eapc(years, vals, start, end)
                if e is None:
                    continue
                trows.append({
                    'location_id': loc, 'location': name,
                    'level': 'Global' if loc == 1 else 'GBD region',
                    'outcome': outcome, 'measure': mlab,
                    'unit': 'percent_of_deaths_or_dalys' if metric == 2 else 'per_100000',
                    'period': plab, 'method': 'EAPC',
                    'n_years': e['n_years'], 'year_start': e['year_start'],
                    'year_end': e['year_end'], 'estimate': e['eapc'],
                    'ci_low': e['ci_low'], 'ci_high': e['ci_high'], 'pvalue': e['pvalue'],
                    'n_joinpoints': np.nan, 'joinpoint_years': '',
                })
            # joinpoint last-segment APC for total incidence / deaths rates only
            if metric == 3 and outcome in ('Incidence rate', 'Deaths rate'):
                jp = joinpoint(years, vals)
                if jp is not None:
                    last = jp['segments'][-1]
                    trows.append({
                        'location_id': loc, 'location': name,
                        'level': 'Global' if loc == 1 else 'GBD region',
                        'outcome': outcome, 'measure': outcome.split(' ')[0],
                        'unit': 'per_100000',
                        'period': f"{last['year_start']}-{last['year_end']}",
                        'method': 'joinpoint_last_segment_APC',
                        'n_years': last['year_end'] - last['year_start'] + 1,
                        'year_start': last['year_start'], 'year_end': last['year_end'],
                        'estimate': last['apc'], 'ci_low': last['ci_low'],
                        'ci_high': last['ci_high'], 'pvalue': np.nan,
                        'n_joinpoints': jp['n_joinpoints'],
                        'joinpoint_years': ';'.join(map(str, jp['joinpoint_years'])),
                    })
    t = pd.DataFrame(trows)
    t.to_csv(os.path.join(RES, 'regions_trends.csv'), index=False)
    print(f'regions_trends.csv: {len(t)} rows')

    # ---------------- reconciliation checks ----------------
    reg = out[out.level == 'GBD region']
    glob = out[out.level == 'Global'].iloc[0]
    checks = {}
    for mlab in ['incidence', 'deaths', 'dalys']:
        g = glob[f'{mlab}_number_2023']
        s = reg[f'{mlab}_number_2023'].sum()
        checks[f'{mlab}_count_2023_sum_over_global'] = float(s / g)
        checks[f'{mlab}_count_2023_sum'] = float(s)
        checks[f'{mlab}_count_2023_global'] = float(g)
        ga = glob[f'attributable_{mlab}_number_2023'] if f'attributable_{mlab}_number_2023' in glob else np.nan
        if np.isfinite(ga):
            sa = reg[f'attributable_{mlab}_number_2023'].sum()
            checks[f'attributable_{mlab}_count_2023_sum_over_global'] = float(sa / ga)

    # ---------------- key findings ----------------
    rr = reg.sort_values('incidence_rate_2023', ascending=False)
    top_inc = [{'region': r.location, 'location_id': int(r.location_id),
                'incidence_rate_2023': float(r.incidence_rate_2023),
                'incidence_rate_1990': float(r.incidence_rate_1990),
                'incidence_number_2023': float(r.incidence_number_2023)}
               for r in rr.head(5).itertuples()]
    paf = reg.sort_values('paf_deaths_2023', ascending=False)
    top_paf = [{'region': r.location, 'location_id': int(r.location_id),
                'paf_deaths_2023_percent': float(r.paf_deaths_2023),
                'attributable_deaths_number_2023': float(r.attributable_deaths_number_2023),
                'attributable_deaths_rate_2023': float(r.attributable_deaths_rate_2023)}
               for r in paf.head(5).itertuples()]
    low_paf = [{'region': r.location, 'location_id': int(r.location_id),
                'paf_deaths_2023_percent': float(r.paf_deaths_2023),
                'attributable_deaths_number_2023': float(r.attributable_deaths_number_2023)}
               for r in paf.tail(3).iloc[::-1].itertuples()]
    jp_last = t[(t.method == 'joinpoint_last_segment_APC') &
                (t.period.str.endswith('2023'))]
    findings = {
        'generated_by': 'scripts/regions.py',
        'definition': ('GBD 2023, cause 1059 total cancers excluding NMSC, both sexes, '
                       'age 15-49 (age id 24). Rates are crude 15-49 per 100,000. '
                       'PAF = metabolic-risk-attributable burden / total 1059 burden x 100 '
                       '(GHI risk 104, cause-410 panel re-based to 1059 by fix_paf.py).'),
        'n_regions': int(len(reg)),
        'global_reference': {
            'incidence_rate_2023': float(glob.incidence_rate_2023),
            'incidence_number_2023': float(glob.incidence_number_2023),
            'deaths_rate_2023': float(glob.deaths_rate_2023),
            'deaths_number_2023': float(glob.deaths_number_2023),
            'paf_deaths_2023_percent': float(glob.paf_deaths_2023),
            'attributable_deaths_number_2023': float(glob.attributable_deaths_number_2023),
            'paf_dalys_2023_percent': float(glob.paf_dalys_2023),
        },
        'top5_incidence_rate_2023': top_inc,
        'top5_paf_deaths_2023': top_paf,
        'lowest3_paf_deaths_2023': low_paf,
        'joinpoint_last_segment_2023': [
            {'region': r.location, 'outcome': r.outcome, 'segment': r.period,
             'apc': float(r.estimate), 'ci_low': float(r.ci_low),
             'ci_high': float(r.ci_high), 'n_joinpoints': int(r.n_joinpoints)}
            for r in jp_last.itertuples() if r.location == 'Global'],
        'reconciliation': checks,
    }
    with open(os.path.join(RES, 'regions_key_findings.json'), 'w') as f:
        json.dump(findings, f, indent=2)
    print('regions_key_findings.json written')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
