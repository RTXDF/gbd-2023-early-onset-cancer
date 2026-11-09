#!/usr/bin/env python3
"""Detection-effect sensitivity: all-cancer incidence excluding diagnostic-intensity-
sensitive sites.

Sites are subtracted in age-standardised space.  Both the aggregate and each site
are directly standardised to the same 2023 global population weights within
15-49, so the difference is the ASR of the aggregate cancer set with those sites
removed (age standardisation is linear in age-specific rates).

Exclusion sets (pre-specified):
  thyroid_only        thyroid
  core                thyroid, prostate, kidney, melanoma
  core_plus_breast    core + breast (female-relevant)

Outputs (results/):
  detection_excluded_series.csv   annual residual ASR by sex and scenario
  detection_excluded_summary.csv  levels, changes, EAPC/joinpoint by scenario
  detection_site_contrib.csv      site-level contribution to the 2013-2023 change
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trends import eapc, joinpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
AGES = list(range(8, 15))
YEARS = list(range(1990, 2024))
SITES = {
    480: 'Thyroid cancer', 438: 'Prostate cancer', 471: 'Kidney cancer',
    459: 'Malignant skin melanoma', 429: 'Breast cancer',
}
SCENARIOS = {
    'Thyroid only': {'both': [480], 'Male': [480], 'Female': [480]},
    'Core detection-sensitive': {'both': [480, 438, 471, 459],
                                 'Male': [480, 438, 471, 459],
                                 'Female': [480, 471, 459]},
    'Core plus breast': {'both': [480, 438, 471, 459, 429],
                         'Male': [480, 438, 471, 459],
                         'Female': [480, 471, 459, 429]},
}
SEXMAP = {'both': 3, 'Male': 1, 'Female': 2}


def load_inputs():
    frames = []
    for sex in (1, 2, 3):
        d = pd.read_pickle(os.path.join(PANELS, f'F_sites_age_inc_m{sex}.pkl'))
        d = d[(d.location == 1) & (d.measure == 6) & (d.metric == 1) &
              (d.age.isin(AGES)) & (d.year.isin(YEARS))]
        frames.append(d)
    inc = pd.concat(frames, ignore_index=True)
    pop = {}
    for sex, sid in SEXMAP.items():
        p = pd.read_pickle(os.path.join(PANELS, f'L_pop_sex{sid}.pkl'))
        p = p[(p.location == 1) & (p.measure == 44) & (p.metric == 1) &
              (p.age.isin(AGES)) & (p.year.isin(YEARS))]
        pop[sex] = p
    return inc, pop


def weights(pop, sex):
    p = pop[sex]
    w = p[(p.year == 2023)].groupby('age').val.sum().reindex(AGES)
    return w / w.sum()


def site_asr(inc, pop, sex, cause):
    d = inc[(inc.sex == SEXMAP[sex]) & (inc.cause == cause)]
    p = pop[sex]
    rows = []
    for yr in YEARS:
        dd = d[d.year == yr].set_index('age').val.reindex(AGES)
        pp = p[p.year == yr].set_index('age').val.reindex(AGES)
        rate = dd / pp * 1e5
        rows.append({'year': yr, 'asr': float((rate * weights(pop, sex)).sum())})
    return pd.DataFrame(rows)


def aggregate_asr(sex):
    if sex == 'both':
        g = pd.read_csv(os.path.join(RES, 'global_asr_series.csv'))
        g = g[g.measure == 'Incidence'].sort_values('year')
        return g.set_index('year').asr_15_49
    s = pd.read_csv(os.path.join(RES, 'sex_asr_series.csv'))
    s = s[(s.measure == 'Incidence') & (s.sex == sex)].sort_values('year')
    return s.set_index('year').asr_15_49


def trend_block(years, values):
    e_full = eapc(years, values, 1990, 2023) or {}
    e_recent = eapc(years, values, 2013, 2023) or {}
    jp = joinpoint(years, values, max_k=3)
    last = {}
    if jp:
        s = jp['segments'][-1]
        last = {'joinpoints': ';'.join(str(x) for x in jp['joinpoint_years']),
                'last_segment': f"{s['year_start']}-{s['year_end']}",
                'last_apc': s['apc'],
                'last_apc_low': s['ci_low'],
                'last_apc_high': s['ci_high']}
    return {
        'eapc_1990_2023': e_full.get('eapc'),
        'eapc_1990_2023_low': e_full.get('ci_low'),
        'eapc_1990_2023_high': e_full.get('ci_high'),
        'eapc_1990_2023_p': e_full.get('pvalue'),
        'eapc_2013_2023': e_recent.get('eapc'),
        'eapc_2013_2023_low': e_recent.get('ci_low'),
        'eapc_2013_2023_high': e_recent.get('ci_high'),
        'eapc_2013_2023_p': e_recent.get('pvalue'),
        **last,
    }


def run():
    inc, pop = load_inputs()
    name_map = (inc[['cause', 'cause_name']].drop_duplicates()
                .set_index('cause').cause_name.to_dict())
    series_rows, summary_rows, contrib_rows = [], [], []

    for sex in ('both', 'Male', 'Female'):
        agg = aggregate_asr(sex).reindex(YEARS)
        all_series = {c: site_asr(inc, pop, sex, c) for c in sorted(inc.cause.unique())}
        # coverage check: all named sites except NMSC relative to the aggregate
        covered = sum(v.set_index('year').asr for c, v in all_series.items() if c != 462)
        coverage_2023 = float(covered.loc[2023] / agg.loc[2023])

        for scenario, by_sex in SCENARIOS.items():
            excl = by_sex[sex]
            removed = sum(all_series[c].set_index('year').asr for c in excl)
            resid = (agg - removed).reindex(YEARS)
            for yr in YEARS:
                series_rows.append({'sex': sex, 'scenario': scenario, 'year': yr,
                                    'asr': float(resid.loc[yr]),
                                    'asr_all': float(agg.loc[yr]),
                                    'asr_removed': float(removed.loc[yr])})
            t = trend_block(YEARS, resid.values)
            base = {
                'sex': sex, 'scenario': scenario,
                'asr_1990': float(resid.loc[1990]), 'asr_2013': float(resid.loc[2013]),
                'asr_2023': float(resid.loc[2023]),
                'pct_change_1990_2023': float((resid.loc[2023] / resid.loc[1990] - 1) * 100),
                'pct_change_2013_2023': float((resid.loc[2023] / resid.loc[2013] - 1) * 100),
                'share_of_2013_2023_rise_removed': float(
                    (removed.loc[2023] - removed.loc[2013]) /
                    (agg.loc[2023] - agg.loc[2013]) * 100),
                'coverage_of_aggregate_2023': coverage_2023,
                'excluded_sites': ','.join(SITES[c] for c in excl),
            }
            summary_rows.append({**base, **t})

        # site contributions to the 2013-2023 change
        denom = float(agg.loc[2023] - agg.loc[2013])
        ref = trend_block(YEARS, agg.values)
        summary_rows.append({
            'sex': sex, 'scenario': 'All cancers (reference)',
            'asr_1990': float(agg.loc[1990]), 'asr_2013': float(agg.loc[2013]),
            'asr_2023': float(agg.loc[2023]),
            'pct_change_1990_2023': float((agg.loc[2023] / agg.loc[1990] - 1) * 100),
            'pct_change_2013_2023': float((agg.loc[2023] / agg.loc[2013] - 1) * 100),
            'share_of_2013_2023_rise_removed': 0.0,
            'coverage_of_aggregate_2023': 1.0,
            'excluded_sites': '',
            **ref,
        })
        for c, v in all_series.items():
            if c == 462:
                continue
            s = v.set_index('year').asr
            contribution = float(s.loc[2023] - s.loc[2013])
            contrib_rows.append({
                'sex': sex, 'cause_id': c, 'site': name_map.get(c, 'Other'),
                'asr_2013': float(s.loc[2013]), 'asr_2023': float(s.loc[2023]),
                'change_2013_2023': contribution,
                'share_of_total_change_pct': contribution / denom * 100 if denom else np.nan,
                'detection_sensitive': c in (480, 438, 471, 459, 429),
            })

    pd.DataFrame(series_rows).to_csv(
        os.path.join(RES, 'detection_excluded_series.csv'), index=False)
    out = pd.DataFrame(summary_rows)
    out.to_csv(os.path.join(RES, 'detection_excluded_summary.csv'), index=False)
    pd.DataFrame(contrib_rows).to_csv(
        os.path.join(RES, 'detection_site_contrib.csv'), index=False)
    cols = ['sex', 'scenario', 'asr_1990', 'asr_2013', 'asr_2023',
            'pct_change_2013_2023', 'eapc_2013_2023', 'last_apc',
            'share_of_2013_2023_rise_removed']
    print(out[cols].to_string(index=False))
    return out


if __name__ == '__main__':
    run()
