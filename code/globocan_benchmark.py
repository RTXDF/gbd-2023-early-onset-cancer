"""External benchmark of the GBD 2023 early-onset estimates against GLOBOCAN 2024.

GLOBOCAN 2024 (IARC Global Cancer Observatory, Cancer Today v3 API) provides
incidence and mortality estimates for 186 countries in 2024 by 5-year age group.
The comparison uses the same case definition as the manuscript: all cancers
excluding non-melanoma skin cancer (GLOBOCAN code 40; GBD cause 1059), ages
15-49 (GLOBOCAN age indices 3-9, token 3_9).

Outputs (results/):
  globocan_global.csv        global and sex-specific counts, crude rates and ratios
  globocan_country.csv       country-level comparison of crude rates
  globocan_sites.csv         site-level incidence comparison
  globocan_summary.json      concordance statistics
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw')
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
RAWG = os.path.join(RAW, 'globocan2024_3_9')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_asr import region_map

AGES = list(range(8, 15))

# GBD cause id -> GLOBOCAN cancer id (2024 labels)
SITE_MAP = {
    411: [6], 414: [7], 417: [11], 423: [14], 426: [15], 429: [20],
    432: [23], 435: [24], 438: [27], 441: [8, 9], 444: [1], 447: [4],
    450: [3, 5], 453: [12], 456: [13], 459: [16], 465: [25], 468: [28],
    471: [29], 474: [30], 477: [31], 480: [32], 483: [18], 484: [33],
    485: [34], 486: [35], 487: [36], 489: [37, 38],
}

# GBD 489 ("other malignant neoplasms") bundles a different set of residual
# categories from GLOBOCAN 37+38, so it is shown but excluded from concordance.
NOT_COMPARABLE = {489}

# country-name harmonisation for names that differ between the two sources
ALIASES = {
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Iran (Islamic Republic of)': 'Iran',
    "Lao People's Democratic Republic": 'Lao PDR',
    "Democratic People's Republic of Korea": 'North Korea',
    'Republic of Korea': 'South Korea',
    'Republic of Moldova': 'Moldova',
    'Russian Federation': 'Russia',
    'Syrian Arab Republic': 'Syria',
    'United Republic of Tanzania': 'Tanzania',
    'United States of America': 'United States',
    'Viet Nam': 'Vietnam',
    'Türkiye': 'Turkey',
    'Czechia': 'Czech Republic',
    'Cabo Verde': 'Cape Verde',
    'Congo': 'Republic of the Congo',
    'Democratic Republic of the Congo': 'DR Congo',
    'Micronesia (Federated States of)': 'Micronesia',
}


def norm(s: str) -> str:
    s = ALIASES.get(str(s).strip(), str(s).strip())
    return (s.lower().replace('.', '').replace('-', ' ').replace('’', "'")
            .replace('  ', ' ').strip())


def gbd_global_counts():
    d = pd.read_pickle(os.path.join(PANELS, 'A_global_trends.pkl'))
    d = d[(d.location == 1) & (d.metric == 1) & (d.year == 2023) &
          (d.age.isin(AGES))]
    rows = []
    for sex_id, label in ((3, 'Both sexes'), (1, 'Male'), (2, 'Female')):
        for meas_id, meas in ((6, 'Incidence'), (1, 'Deaths')):
            v = d[(d.sex == sex_id) & (d.measure == meas_id)].val.sum()
            rows.append({'sex': label, 'measure': meas, 'gbd_count': float(v)})
    out = pd.DataFrame(rows)
    # population and crude rate from the published global series
    g = pd.read_csv(os.path.join(RES, 'global_asr_series.csv'))
    g = g[g.year == 2023].set_index('measure')
    for meas in ('Incidence', 'Deaths'):
        i = out[(out.sex == 'Both sexes') & (out.measure == meas)].index[0]
        out.loc[i, 'gbd_pop'] = float(g.loc[meas, 'population'])
        out.loc[i, 'gbd_crude'] = float(g.loc[meas, 'crude_rate'])
    s = pd.read_csv(os.path.join(RES, 'sex_asr_series.csv'))
    s = s[s.year == 2023]
    for (sex, meas), grp in s.groupby(['sex', 'measure']):
        i = out[(out.sex == sex) & (out.measure == meas)].index
        if len(i):
            out.loc[i[0], 'gbd_crude'] = float(grp.asr_15_49.iloc[0])
    return out


def load_globocan(pattern):
    frames = [pd.DataFrame(json.load(open(f))['dataset'])
              for f in sorted(glob.glob(os.path.join(RAWG, pattern)))]
    return pd.concat(frames, ignore_index=True)


def load_globocan_global():
    return load_globocan('world_type*_sex[012].json')


def load_globocan_country():
    d = load_globocan('countries_type*_sex[012].json')
    d = d[d.sex == 0]  # country comparison is both sexes only
    meta = pd.DataFrame(json.load(open(os.path.join(RAWG, 'meta_populations.json'))))
    meta = meta[meta.country_iso3.notna() & (meta.country_iso3.astype(str).str.len() == 3)]
    meta = meta[['country_code', 'label', 'country_iso3']].drop_duplicates('country_code')
    return d.merge(meta, on='country_code', how='inner')


def load_gbd_country():
    d = pd.read_csv(os.path.join(RES, 'country_asr_series.csv'))
    d = d[(d.year == 2023) & (d.measure.isin(['Incidence', 'Deaths']))]
    return d[['location_id', 'location', 'measure', 'count', 'population', 'crude_rate']]


def main():
    gbd_g = gbd_global_counts()
    gc_g = load_globocan_global()

    rows = []
    for sex_code, label in ((0, 'Both sexes'), (1, 'Male'), (2, 'Female')):
        for typ, meas in ((0, 'Incidence'), (1, 'Deaths')):
            g = gc_g[(gc_g.sex == sex_code) & (gc_g.type == typ)].iloc[0]
            rows.append({'sex': label, 'measure': meas,
                         'globocan_count': float(g.total),
                         'globocan_pop': float(g.total_pop),
                         'globocan_crude': float(g.crude_rate),
                         'globocan_asr': float(g.asr)})
    global_df = pd.DataFrame(rows).merge(gbd_g, on=['sex', 'measure'], how='left')
    # sex-specific GBD population from the 15-49 age-band population panel
    pop = pd.read_pickle(os.path.join(PANELS, 'L_pop_sex3.pkl'))
    pop23 = pop[(pop.location == 1) & (pop.measure == 44) & (pop.metric == 1) &
                (pop.year == 2023) & (pop.age.isin(AGES))].val.sum()
    for sex in ('Male', 'Female'):
        p = pd.read_pickle(os.path.join(PANELS, f"L_pop_sex{1 if sex == 'Male' else 2}.pkl"))
        pv = p[(p.location == 1) & (p.measure == 44) & (p.metric == 1) &
               (p.year == 2023) & (p.age.isin(AGES))].val.sum()
        i = global_df[(global_df.sex == sex)].index
        global_df.loc[i, 'gbd_pop'] = float(pv)
    i = global_df[(global_df.sex == 'Both sexes')].index
    global_df.loc[i, 'gbd_pop'] = float(pop23)
    global_df['count_ratio'] = global_df.globocan_count / global_df.gbd_count
    global_df['rate_ratio'] = global_df.globocan_crude / global_df.gbd_crude
    global_df.to_csv(os.path.join(RES, 'globocan_global.csv'), index=False)

    # ---------------- country level ----------------
    gbd_c = load_gbd_country()
    gc_c = load_globocan_country()
    gc_c = gc_c.rename(columns={'total': 'globocan_count', 'crude_rate': 'globocan_crude',
                                'asr': 'globocan_asr', 'label': 'location',
                                'total_pop': 'globocan_pop'})
    gc_c['measure'] = gc_c.type.map({0: 'Incidence', 1: 'Deaths'})
    gbd_c['key'] = gbd_c.location.map(norm)
    gc_c['key'] = gc_c.location.map(norm)
    m = gbd_c.merge(gc_c[['key', 'country_iso3', 'measure', 'globocan_count',
                          'globocan_pop', 'globocan_crude', 'globocan_asr']],
                    on=['key', 'measure'], how='inner', suffixes=('', '_gc'))
    m['ratio'] = m.globocan_crude / m.crude_rate
    out_c = m[['location_id', 'location', 'country_iso3', 'measure', 'count',
               'population', 'crude_rate', 'globocan_count', 'globocan_pop',
               'globocan_crude', 'globocan_asr', 'ratio']].rename(columns={
                   'count': 'gbd_count', 'population': 'population',
                   'crude_rate': 'gbd_crude'})
    out_c.to_csv(os.path.join(RES, 'globocan_country.csv'), index=False)

    stats_country = {}
    for meas in ('Incidence', 'Deaths'):
        sub = out_c[out_c.measure == meas].dropna(subset=['globocan_crude', 'gbd_crude'])
        rho = stats.spearmanr(sub.globocan_crude, sub.gbd_crude).statistic
        top20 = len(set(sub.nlargest(20, 'globocan_crude').location) &
                    set(sub.nlargest(20, 'gbd_crude').location))
        big = sub.nlargest(50, 'population')
        stats_country[meas] = {
            'n_countries': int(len(sub)),
            'spearman_rho': round(float(rho), 3),
            'spearman_rho_50_largest': round(float(stats.spearmanr(
                big.globocan_crude, big.gbd_crude).statistic), 3),
            'median_ratio': round(float(np.median(sub.ratio)), 3),
            'iqr_ratio': [round(float(np.percentile(sub.ratio, 25)), 3),
                          round(float(np.percentile(sub.ratio, 75)), 3)],
            'gbd_higher_share_pct': round(float(np.mean(sub.ratio < 1) * 100), 1),
            'top20_overlap': int(top20),
            'share_within_25pct': round(float(np.mean((sub.ratio > 0.75) & (sub.ratio < 1.25)) * 100), 1),
        }

    # ---------------- region level ----------------
    reg_of = {lid: v[1] for lid, v in region_map().items()}
    out_c['region'] = out_c.location_id.map(reg_of)
    reg_rows = []
    for meas in ('Incidence', 'Deaths'):
        s = out_c[(out_c.measure == meas) & out_c.region.notna()]
        g = s.groupby('region').agg(
            gbd_count=('gbd_count', 'sum'), gbd_pop=('population', 'sum'),
            globocan_count=('globocan_count', 'sum'),
            globocan_pop=('globocan_pop', 'sum'))
        g['gbd_crude'] = g.gbd_count / g.gbd_pop * 1e5
        g['globocan_crude'] = g.globocan_count / g.globocan_pop * 1e5
        g['ratio'] = g.globocan_crude / g.gbd_crude
        g['measure'] = meas
        reg_rows.append(g.reset_index())
    region_df = pd.concat(reg_rows, ignore_index=True)
    region_df.to_csv(os.path.join(RES, 'globocan_region.csv'), index=False)
    stats_region = {}
    for meas in ('Incidence', 'Deaths'):
        s = region_df[region_df.measure == meas]
        stats_region[meas] = {
            'n_regions': int(len(s)),
            'spearman_rho': round(float(stats.spearmanr(s.globocan_crude,
                                                        s.gbd_crude).statistic), 3),
            'median_ratio': round(float(np.median(s.ratio)), 3),
            'iqr_ratio': [round(float(np.percentile(s.ratio, 25)), 3),
                          round(float(np.percentile(s.ratio, 75)), 3)],
        }

    # ---------------- site level ----------------
    gbd_s = pd.read_csv(os.path.join(RES, 'sites_global_2023.csv'))
    gbd_s = gbd_s[gbd_s.metric == 'Number']
    gc_s = load_globocan('world_type*_sex0_cancer*.json')
    gc_s = gc_s[gc_s.sex == 0]
    rows = []
    for meas, typ in (('Incidence', 0), ('Deaths', 1)):
        g_all = gbd_s[gbd_s.measure == meas]
        sub_all = gc_s[gc_s.type == typ]
        for cause, ids in SITE_MAP.items():
            g = g_all[g_all.cause_id == cause]
            sub = sub_all[sub_all.cancer_code.isin(ids)]
            if not len(g) or not len(sub):
                continue
            gv = float(g.val.iloc[0])
            gcv = float(sub.total.sum())
            rows.append({'measure': meas, 'cause_id': cause,
                         'site': g.cause.iloc[0], 'gbd_2023': gv,
                         'globocan_2024': gcv, 'ratio': gcv / gv,
                         'comparable': cause not in NOT_COMPARABLE})
    site_df = pd.DataFrame(rows)
    for meas in ('Incidence', 'Deaths'):
        m = site_df.measure == meas
        site_df.loc[m, 'gbd_share'] = (site_df.loc[m, 'gbd_2023'] /
                                       site_df.loc[m, 'gbd_2023'].sum() * 100)
        site_df.loc[m, 'globocan_share'] = (site_df.loc[m, 'globocan_2024'] /
                                            site_df.loc[m, 'globocan_2024'].sum() * 100)
        site_df.loc[m, 'share_difference_pp'] = (site_df.loc[m, 'globocan_share'] -
                                                 site_df.loc[m, 'gbd_share'])
    site_df = site_df.sort_values(['measure', 'gbd_2023'], ascending=[True, False])
    site_df.to_csv(os.path.join(RES, 'globocan_sites.csv'), index=False)

    site_stats = {}
    for meas in ('Incidence', 'Deaths'):
        s = site_df[(site_df.measure == meas) & site_df.comparable]
        site_stats[meas] = {
            'n_sites': int(len(s)),
            'spearman_counts': round(float(stats.spearmanr(s.gbd_2023,
                                                           s.globocan_2024).statistic), 3),
            'spearman_shares': round(float(stats.spearmanr(s.gbd_share,
                                                           s.globocan_share).statistic), 3),
            'median_ratio': round(float(np.median(s.ratio)), 3),
            'share_within_25pct': round(float(np.mean((s.ratio > 0.75) &
                                                      (s.ratio < 1.25)) * 100), 1),
            'largest_ratio': s.nlargest(3, 'ratio')[['site', 'ratio']].round(2).to_dict('records'),
            'smallest_ratio': s.nsmallest(3, 'ratio')[['site', 'ratio']].round(2).to_dict('records'),
        }
    # sites ranked in the same top 10 by both sources
    s_inc = site_df[(site_df.measure == 'Incidence') & site_df.comparable]
    site_stats['Incidence']['top10_overlap'] = int(
        len(set(s_inc.nlargest(10, 'gbd_2023').cause_id) &
            set(s_inc.nlargest(10, 'globocan_2024').cause_id)))
    site_stats['Incidence']['top5_gbd'] = s_inc.nlargest(5, 'gbd_2023').site.tolist()
    site_stats['Incidence']['top5_globocan'] = s_inc.nlargest(5, 'globocan_2024').site.tolist()

    # site-level discrepancies (signed); because the all-cancer totals agree
    # closely these largely offset, so we rank by absolute difference
    gap = s_inc[['site', 'gbd_2023', 'globocan_2024']].copy()
    gap['difference'] = gap.globocan_2024 - gap.gbd_2023
    gap['absolute_difference'] = gap.difference.abs()
    gap['ratio'] = gap.globocan_2024 / gap.gbd_2023
    gap.sort_values('absolute_difference', ascending=False).to_csv(
        os.path.join(RES, 'globocan_gap_by_site.csv'), index=False)

    summary = {
        'source': 'GLOBOCAN 2024 (IARC GCO, Cancer Today v3 API)',
        'case_definition': 'All cancers excluding non-melanoma skin cancer, ages 15-49',
        'n_countries_globocan': int(gc_c.country_iso3.nunique()),
        'global': global_df.round(3).to_dict(orient='records'),
        'country': stats_country,
        'region': stats_region,
        'sites': site_stats,
        'incidence_difference_by_site': {
            'net_difference': round(float(gap.difference.sum()), 1),
            'sum_absolute_difference': round(float(gap.absolute_difference.sum()), 1),
            'top_contributors': gap.nlargest(5, 'absolute_difference')
            [['site', 'difference', 'ratio']].round(2).to_dict('records'),
        },
    }

    with open(os.path.join(RES, 'globocan_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(global_df.round(3).to_string(index=False))
    print(json.dumps(stats_country, indent=2))
    print(site_df.round(3).to_string(index=False))
    print(json.dumps(site_stats, indent=2))
    print(gap.nlargest(8, 'absolute_difference').round(1).to_string(index=False))


if __name__ == '__main__':
    main()
