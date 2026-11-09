#!/usr/bin/env python3
"""Correct PAFs in risk panels from cause-410 (all neoplasms incl. benign) denominator
to cause-1059 (total cancers excluding NMSC) denominator.

GBD 2023 'Neoplasms' (410) aggregates malignant + benign/in-situ neoplasms; the study
outcome is malignant cancer (cause 1059). Attributable NUMBERS are identical under both
causes (verified against the API); only PAF (Percent) denominators differ.

- age 24 (15-49) and age-specific rows: PAF = attributable Number / total-1059 Number
  (exact); UI bounds scaled by the point-estimate correction ratio (denominator
  uncertainty is negligible vs numerator uncertainty; verified ratio of bounds ~1.015).
- age 27 (ASR): Number is suppressed, so PAF is scaled by the age-24 correction ratio
  for the same location/year/measure/sex (documented approximation; effect <2%).
Run AFTER load_data.py (which rebuilds panels from raw cause-410 risk queries).
"""
import os, json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')

def load(n):
    p = os.path.join(PANELS, n + '.pkl')
    return pd.read_pickle(p) if os.path.exists(p) else None

def total_lookup(df):
    """(location, year, measure, sex, age) -> total Number (metric 1)."""
    d = df[df.metric == 1]
    return {(int(r.location), int(r.year), int(r.measure), int(r.sex), int(r.age)): float(r.val)
            for r in d.itertuples()}

def correct_panel(df, totals, label):
    """df: risk panel with metric 1 (Number) & 2 (Percent). totals: cause-1059 counts."""
    df = df.copy()
    num = {(int(r.location), int(r.year), int(r.measure), int(r.sex), int(r.age), int(r.rei)): (float(r.val), float(r.lower), float(r.upper))
           for r in df[df.metric == 1].itertuples()}
    # first pass: exact correction for non-ASR rows; collect ratios
    ratios = {}   # (loc, year, measure, sex) -> ratio new/old
    idxs = df.index[df.metric == 2]
    exact = scaled = kept = 0
    for i in idxs:
        r = df.loc[i]
        key = (int(r.location), int(r.year), int(r.measure), int(r.sex), int(r.age))
        nkey = key + (int(r.rei),)
        old = float(r.val)
        if int(r.age) != 27 and key in totals and nkey in num and old > 0:
            new = num[nkey][0] / totals[key]
            ratio = new / old
            df.loc[i, ['val', 'lower', 'upper']] = [new, float(r.lower) * ratio, float(r.upper) * ratio]
            ratios[(key[0], key[1], key[2], key[3])] = ratio
            exact += 1
    # fallback ratio per (measure, sex, year): median across locations
    fallback = {}
    for (loc, yr, m, s), v in ratios.items():
        fallback.setdefault((m, s, yr), []).append(v)
    fallback = {k: float(np.median(v)) for k, v in fallback.items()}
    for i in idxs:
        r = df.loc[i]
        key = (int(r.location), int(r.year), int(r.measure), int(r.sex), int(r.age))
        nkey = key + (int(r.rei),)
        if int(r.age) == 27:
            ratio = ratios.get((key[0], key[1], key[2], key[3])) or fallback.get((key[2], key[3], key[1]))
            if ratio:
                df.loc[i, ['val', 'lower', 'upper']] = [float(r.val) * ratio, float(r.lower) * ratio, float(r.upper) * ratio]
                scaled += 1
            else:
                kept += 1
        elif key not in totals or nkey not in num:
            ratio = fallback.get((key[2], key[3], key[1]))
            if ratio and float(r.val) > 0:
                df.loc[i, ['val', 'lower', 'upper']] = [float(r.val) * ratio, float(r.lower) * ratio, float(r.upper) * ratio]
                scaled += 1
            else:
                kept += 1
    df.to_pickle(os.path.join(PANELS, label + '.pkl'))
    return {'panel': label, 'exact': exact, 'scaled': scaled, 'uncorrected': kept}

def main():
    A = load('A_global_trends')
    report = []
    # country/region/quintile/global totals from BCD (age 24), per sex
    bcd_totals = {}
    for sex, tag in [(3, 'both'), (1, 'male'), (2, 'female')]:
        b = load(f'BCD_country_eo_{tag}')
        if b is not None:
            bcd_totals[sex] = total_lookup(b[b.age == 24])
    # E panels: age-specific country totals (measures 1,2,6; all sexes)
    e_totals = {}
    for m in [1, 2, 6]:
        e = load(f'E_country_age_m{m}')
        if e is not None:
            e_totals.update(total_lookup(e))
    for rei in [104, 105, 108]:
        for tag, sex in [('both', 3), ('male', 1), ('female', 2)]:
            name = f'GHI_risk{rei}_country_{tag}'
            df = load(name)
            if df is None: continue
            totals = dict(bcd_totals.get(sex, {}))
            totals.update(e_totals)
            if A is not None:
                totals.update(total_lookup(A))
            report.append(correct_panel(df, totals, name))
    j = load('J_risk_age_global')
    if j is not None and A is not None:
        report.append(correct_panel(j, total_lookup(A), 'J_risk_age_global'))
    json.dump(report, open(os.path.join(ROOT, 'results', 'paf_correction_report.json'), 'w'), indent=2)
    for r in report: print(r)

if __name__ == '__main__':
    main()