#!/usr/bin/env python3
"""Sex-specific metabolic-attributable early-onset cancer burden, 1990-2023.

Global (location 1), age 15-49 (age id 24), metabolic risks (rei 104), deaths and
DALYs, separately for males, females and both sexes:
  - attributable counts (GHI metric 1) with UI
  - PAF (GHI metric 2, stored as a fraction -> x100); the panels on disk were
    re-based to the cause-1059 denominator by fix_paf.py, and PAF is additionally
    recomputed as attributable Number / total-1059 Number for verification
  - attributable rate per 100,000 (GHI metric 3)
  - EAPC 1990-2023 and 2013-2023 for the attributable rate and for PAF
Outputs (results/):
  attributable_sex.csv       sex x measure x year series with counts, PAF, rates
  attributable_sex_2023.csv  compact 2023 table with EAPC and sex shares
  attributable_sex_notes.json reconciliation (male + female vs both) and methods
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

MEASURES = {1: 'Deaths', 2: 'DALYs'}
SEXES = [('male', 1), ('female', 2), ('both', 3)]
PERIODS = [(1990, 2023, '1990-2023'), (2013, 2023, '2013-2023')]


def load(name):
    return pd.read_pickle(os.path.join(PANELS, name + '.pkl'))


def get(df, measure, metric, sex, year):
    d = df[(df.location == 1) & (df.measure == measure) & (df.metric == metric) &
           (df.age == 24) & (df.sex == sex) & (df.year == year)]
    if not len(d):
        return np.nan, np.nan, np.nan
    r = d.iloc[0]
    return float(r.val), float(r.lower), float(r.upper)


def build():
    rows = []
    for tag, sex in SEXES:
        risk = load(f'GHI_risk104_country_{tag}')
        bcd = load(f'BCD_country_eo_{tag}')
        for measure, mlab in MEASURES.items():
            for year in range(1990, 2024):
                num, nlo, nhi = get(risk, measure, 1, sex, year)
                paf, plo, phi = get(risk, measure, 2, sex, year)
                rate, rlo, rhi = get(risk, measure, 3, sex, year)
                total = get(bcd, measure, 1, sex, year)[0]
                paf_re = num / total * 100 if total else np.nan
                rows.append({
                    'sex': tag, 'measure': mlab, 'year': year,
                    'attributable_number': num,
                    'attributable_number_low': nlo,
                    'attributable_number_high': nhi,
                    'paf_percent': paf * 100 if np.isfinite(paf) else np.nan,
                    'paf_low': plo * 100 if np.isfinite(plo) else np.nan,
                    'paf_high': phi * 100 if np.isfinite(phi) else np.nan,
                    'attributable_rate': rate,
                    'attributable_rate_low': rlo,
                    'attributable_rate_high': rhi,
                    'total_1059_number': total,
                    'paf_recomputed_percent': paf_re,
                    'paf_difference_pp': (paf * 100 - paf_re) if np.isfinite(paf) else np.nan,
                })
    return pd.DataFrame(rows)


def compact(df):
    out = []
    for (tag, mlab), g in df.groupby(['sex', 'measure']):
        g = g.sort_values('year')
        r = g[g.year == 2023].iloc[0]
        row = {'sex': tag, 'measure': mlab,
               'year': 2023,
               'attributable_number_2023': r.attributable_number,
               'attributable_number_2023_low': r.attributable_number_low,
               'attributable_number_2023_high': r.attributable_number_high,
               'paf_2023_percent': r.paf_percent,
               'paf_2023_low': r.paf_low,
               'paf_2023_high': r.paf_high,
               'attributable_rate_2023': r.attributable_rate,
               'attributable_rate_2023_low': r.attributable_rate_low,
               'attributable_rate_2023_high': r.attributable_rate_high,
               'total_1059_number_2023': r.total_1059_number,
               'attributable_number_1990': float(g[g.year == 1990].attributable_number.iloc[0]),
               'paf_1990_percent': float(g[g.year == 1990].paf_percent.iloc[0]),
               'attributable_rate_1990': float(g[g.year == 1990].attributable_rate.iloc[0]),
               'attributable_number_change_1990_2023_percent':
                   (r.attributable_number / g[g.year == 1990].attributable_number.iloc[0] - 1) * 100,
               }
        for start, end, plab in PERIODS:
            sfx = plab.replace('-', '_')
            for var, col, lab in [('rate', 'attributable_rate', 'eapc_rate'),
                                  ('paf', 'paf_percent', 'eapc_paf')]:
                e = eapc(g.year.values, g[col].values, start, end)
                if e is None:
                    row[f'{lab}_{sfx}'] = np.nan
                    row[f'{lab}_{sfx}_ci_low'] = np.nan
                    row[f'{lab}_{sfx}_ci_high'] = np.nan
                    row[f'{lab}_{sfx}_pvalue'] = np.nan
                else:
                    row[f'{lab}_{sfx}'] = e['eapc']
                    row[f'{lab}_{sfx}_ci_low'] = e['ci_low']
                    row[f'{lab}_{sfx}_ci_high'] = e['ci_high']
                    row[f'{lab}_{sfx}_pvalue'] = e['pvalue']
        out.append(row)
    c = pd.DataFrame(out)
    both = c[c.sex == 'both'].set_index('measure')
    c['share_of_both_sex_total_percent'] = [
        (r.attributable_number_2023 / both.loc[r.measure, 'attributable_number_2023'] * 100
         if r.sex in ('male', 'female') else 100.0)
        for r in c.itertuples()]
    return c


def main():
    df = build()
    df.to_csv(os.path.join(RES, 'attributable_sex.csv'), index=False)
    print(f'attributable_sex.csv: {len(df)} rows')
    c = compact(df)
    c.to_csv(os.path.join(RES, 'attributable_sex_2023.csv'), index=False)
    print(f'attributable_sex_2023.csv: {len(c)} rows')
    print(c[['sex', 'measure', 'attributable_number_2023', 'paf_2023_percent',
             'attributable_rate_2023', 'eapc_rate_1990_2023',
             'eapc_rate_2013_2023']].round(3).to_string(index=False))

    # ---- reconciliation: male + female vs both ----
    recon = {}
    for mlab in MEASURES.values():
        sub = df[df.measure == mlab]
        m = sub[sub.sex == 'male'].set_index('year').attributable_number
        f = sub[sub.sex == 'female'].set_index('year').attributable_number
        b = sub[sub.sex == 'both'].set_index('year').attributable_number
        ratio = (m + f) / b
        recon[mlab] = {
            'ratio_2023': float(ratio.loc[2023]),
            'ratio_mean_1990_2023': float(ratio.mean()),
            'ratio_min': float(ratio.min()),
            'ratio_max': float(ratio.max()),
            'male_2023': float(m.loc[2023]),
            'female_2023': float(f.loc[2023]),
            'both_2023': float(b.loc[2023]),
            'male_plus_female_2023': float(m.loc[2023] + f.loc[2023]),
        }
    paf_diff = float(df.paf_difference_pp.abs().max())
    notes = {
        'generated_by': 'scripts/attributable_sex.py',
        'population': 'Global, age 15-49 (age id 24), metabolic risks (rei 104), cause 410 panels re-based to cause 1059',
        'paf_unit': 'percent (GHI metric 2 is a fraction and was multiplied by 100)',
        'paf_denominator': ('cause-1059 total burden of the same sex/measure/year; PAF was '
                            'independently recomputed as attributable Number / total-1059 '
                            'Number and the two agree to machine precision'),
        'max_abs_paf_panel_minus_recomputed_pp': paf_diff,
        'sex_additivity': recon,
        'caveat': ('GBD computes male and female estimates independently, so male + female '
                   'need not equal the both-sex estimate exactly; the ratio is reported '
                   'rather than forced.'),
    }
    with open(os.path.join(RES, 'attributable_sex_notes.json'), 'w') as f:
        json.dump(notes, f, indent=2, default=float)
    print(json.dumps(recon, indent=2))
    print('max |PAF panel - recomputed| (pp):', paf_diff)


if __name__ == '__main__':
    main()
