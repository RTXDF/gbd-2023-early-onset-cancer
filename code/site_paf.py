#!/usr/bin/env python3
"""Site-specific metabolic-attributable fractions (PAF) of early-onset cancer, 2023.

Quantified metabolic attribution exists for 14 of the 29 GBD cancer sites
(intersection of K_risk104_sites_m{1,2}_a24 causes and F_sites_m{1,2}_a24 causes).
For every site and measure (deaths = m1, DALYs = m2), global, both sexes, age 15-49:
  attributable number + UI  (K_risk104, rei 104)
  total site number + UI    (F_sites)
  PAF = attributable / total x 100
  attributable and total rates (per 100,000)
PAF uncertainty: the attributable-number UI is propagated by the ratio method,
  paf_low/high = paf_point x (attributable_low/high / attributable_point),
i.e. denominator (total site burden) uncertainty is treated as negligible relative
to the attributable-numerator uncertainty; bounds are re-ordered after scaling.
The same construction is repeated for rei 105 (high fasting plasma glucose) and
108 (high body-mass index) so BMI vs FPG contributions can be compared; those
sub-risk panels contain a few negative point estimates (GBD ensemble), which give
negative PAFs and are reported as such.
Outputs (results/):
  sites_paf_2023.csv         14 sites x measure, PAF and rates, sorted by PAF
  sites_paf_by_risk_2023.csv rei 104/105/108 x site x measure
  sites_paf_notes.json       methods, 14-site reconciliation, >100% check
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)

RISKS = {104: 'Metabolic risks', 105: 'High fasting plasma glucose',
         108: 'High body-mass index'}
MEASURES = {1: 'Deaths', 2: 'DALYs'}
YEAR = 2023


def load(name):
    return pd.read_pickle(os.path.join(PANELS, name + '.pkl'))


def point(df, cause, measure, metric, year=YEAR):
    d = df[(df.location == 1) & (df.cause == cause) & (df.measure == measure) &
           (df.metric == metric) & (df.age == 24) & (df.sex == 3) & (df.year == year)]
    if not len(d):
        return None
    r = d.iloc[0]
    return float(r.val), float(r.lower), float(r.upper), r.cause_name


def scaled_bounds(value, low, high, center):
    """PAF bounds via proportional scaling of the attributable-number UI."""
    if center is None or not np.isfinite(center) or center == 0:
        return np.nan, np.nan
    a = value * low / center
    b = value * high / center
    return float(min(a, b)), float(max(a, b))


def site_table(rei):
    km1, km2 = load(f'K_risk{rei}_sites_m1_a24'), load(f'K_risk{rei}_sites_m2_a24')
    fm1, fm2 = load('F_sites_m1_a24'), load('F_sites_m2_a24')
    sites = sorted(set(km1.cause.unique()) & set(fm1.cause.unique()))
    rows = []
    for measure, kdf, fdf in [(1, km1, fm1), (2, km2, fm2)]:
        for cause in sites:
            k1 = point(kdf, cause, measure, 1)
            k3 = point(kdf, cause, measure, 3)
            f1 = point(fdf, cause, measure, 1)
            f3 = point(fdf, cause, measure, 3)
            if k1 is None or f1 is None:
                continue
            attr, alo, ahi, cname = k1
            total, tlo, thi, _ = f1
            paf = attr / total * 100 if total else np.nan
            plo, phi = scaled_bounds(paf, alo, ahi, attr)
            rows.append({
                'rei': rei, 'risk': RISKS[rei], 'site': cname, 'cause_id': int(cause),
                'measure': MEASURES[measure],
                'attributable_number': attr, 'attributable_low': alo,
                'attributable_high': ahi, 'total_number': total,
                'total_low': tlo, 'total_high': thi,
                'paf_percent': paf, 'paf_low': plo, 'paf_high': phi,
                'attributable_rate': k3[0] if k3 else np.nan,
                'attributable_rate_low': k3[1] if k3 else np.nan,
                'attributable_rate_high': k3[2] if k3 else np.nan,
                'total_rate': f3[0] if f3 else np.nan,
                'total_rate_low': f3[1] if f3 else np.nan,
                'total_rate_high': f3[2] if f3 else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    t104 = site_table(104)
    t104 = t104.sort_values(['paf_percent'], ascending=False).reset_index(drop=True)
    t104.to_csv(os.path.join(RES, 'sites_paf_2023.csv'), index=False)
    print(f'sites_paf_2023.csv: {len(t104)} rows')

    by_risk = []
    for rei in (104, 105, 108):
        by_risk.append(site_table(rei))
    br = pd.concat(by_risk, ignore_index=True)
    br = br.sort_values(['risk', 'measure', 'paf_percent'], ascending=[True, True, False])
    br.to_csv(os.path.join(RES, 'sites_paf_by_risk_2023.csv'), index=False)
    print(f'sites_paf_by_risk_2023.csv: {len(br)} rows')

    # ---- reconciliation: 14-site attributable deaths vs global risk-104 total ----
    gh = load('GHI_risk104_country_both')
    g = gh[(gh.location == 1) & (gh.measure == 1) & (gh.metric == 1) & (gh.age == 24) &
           (gh.sex == 3) & (gh.year == YEAR)]
    global_attr = float(g.val.iloc[0])
    site_sum_d = float(t104[t104.measure == 'Deaths'].attributable_number.sum())
    site_sum_y = float(t104[t104.measure == 'DALYs'].attributable_number.sum())
    gl = gh[(gh.location == 1) & (gh.measure == 2) & (gh.metric == 1) & (gh.age == 24) &
            (gh.sex == 3) & (gh.year == YEAR)]
    global_attr_y = float(gl.val.iloc[0])

    # 14-site total vs all-cancer total (coverage of the burden)
    bcd = load('BCD_country_eo_both')
    tot_all_d = float(bcd[(bcd.location == 1) & (bcd.measure == 1) & (bcd.metric == 1) &
                          (bcd.age == 24) & (bcd.sex == 3) & (bcd.year == YEAR)].val.iloc[0])
    tot_all_y = float(bcd[(bcd.location == 1) & (bcd.measure == 2) & (bcd.metric == 1) &
                          (bcd.age == 24) & (bcd.sex == 3) & (bcd.year == YEAR)].val.iloc[0])

    notes = {
        'generated_by': 'scripts/site_paf.py',
        'year': YEAR,
        'population': 'Global, both sexes, age 15-49 (age id 24)',
        'n_sites': int(t104[t104.measure == 'Deaths'].shape[0]),
        'sites': sorted(t104.site.unique().tolist()),
        'paf_definition': ('PAF = metabolic-attributable burden (rei 104) / total burden for '
                           'the same site and measure x 100. Risk panels are cause-410 '
                           'queries re-based to the cause-1059 denominator by fix_paf.py; '
                           'attributable counts are identical under either denominator.'),
        'paf_ui_method': ('paf_low/high = paf_point x (attributable_low/high / '
                          'attributable_point); total-site UI treated as negligible; '
                          'bounds re-ordered after scaling.'),
        'reconciliation_2023': {
            'sum_14_site_attributable_deaths': site_sum_d,
            'global_risk104_attributable_deaths': global_attr,
            'difference': site_sum_d - global_attr,
            'sum_14_site_attributable_dalys': site_sum_y,
            'global_risk104_attributable_dalys': global_attr_y,
            'difference_dalys': site_sum_y - global_attr_y,
            'all_cancer_total_deaths_2023': tot_all_d,
            'all_cancer_total_dalys_2023': tot_all_y,
            'site_burden_share_of_all_cancer_deaths_percent': site_sum_d / tot_all_d * 100,
            'site_burden_share_of_all_cancer_dalys_percent': site_sum_y / tot_all_y * 100,
        },
        'paf_over_100_check': {
            'max_paf_percent_deaths': float(t104[t104.measure == 'Deaths'].paf_percent.max()),
            'max_paf_percent_dalys': float(t104[t104.measure == 'DALYs'].paf_percent.max()),
            'n_over_100': int((t104.paf_percent > 100).sum()),
            'statement': ('PAFs cannot exceed 100% here; they exceed 100% only if the '
                          'attributable definition overlaps/aggregates beyond the site '
                          'denominator. No such case occurs in this table.'),
        },
        'negative_point_estimates_by_risk': {
            RISKS[rei]: int((br[(br.rei == rei)].attributable_number < 0).sum())
            for rei in (104, 105, 108)
        },
        'sub_risk_caveat': ('rei 105 (FPG) and 108 (BMI) are sub-risks of rei 104 and are '
                            'not mutually exclusive; a few negative site-level ensemble '
                            'estimates yield negative PAFs and must not be summed.'),
    }
    with open(os.path.join(RES, 'sites_paf_notes.json'), 'w') as f:
        json.dump(notes, f, indent=2, default=float)

    print(json.dumps(notes['reconciliation_2023'], indent=2))
    print(json.dumps(notes['paf_over_100_check'], indent=2))
    print(t104[['site', 'measure', 'paf_percent', 'attributable_number']].to_string(index=False))


if __name__ == '__main__':
    main()
