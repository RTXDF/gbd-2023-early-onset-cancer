#!/usr/bin/env python3
"""Metabolic-risk-attributable early-onset cancer burden (Direction A core novelty).

Risk-context panels GHI_risk{104,105,108}_country_{both,male,female}:
  metric Number(1) = attributable count; Percent(2) = PAF as FRACTION (0.039 = 3.9%); Rate(3) = attributable rate.
  measures Deaths(1), DALYs(2); ages 15-49(24), Age-standardized(27).
Outputs (results/):
  attributable_global.csv     global attributable deaths/DALYs + PAF, 1990-2023, by risk/sex/age
  attributable_country_2023.csv  country PAF + attributable rate/number (2023) for maps/inequality
  attributable_trends_eapc.csv   EAPC of attributable rate & PAF
"""
import os, json, math
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results'); os.makedirs(RES, exist_ok=True)
idmap = json.load(open(os.path.join(ROOT, 'data', 'id_map.json')))
COUNTRIES = set(int(k) for k in idmap['countries'])
RISKS = {104: 'Metabolic risks', 105: 'High fasting plasma glucose', 108: 'High body-mass index'}
MEASURES = {1: 'Deaths', 2: 'DALYs'}

def load_risk_panels():
    frames = []
    for rei in RISKS:
        for sex in ['both', 'male', 'female']:
            p = os.path.join(PANELS, f'GHI_risk{rei}_country_{sex}.pkl')
            if os.path.exists(p):
                d = pd.read_pickle(p); d['risk'] = RISKS[rei]; d['rei'] = rei; d['sex_query'] = sex
                frames.append(d)
    if not frames: return None
    return pd.concat(frames, ignore_index=True)

def run_global(df):
    g = df[(df.location == 1)].copy()
    rows = []
    for (rei, measure, sex, age, metric), gg in g.groupby(['rei', 'measure', 'sex_query', 'age', 'metric']):
        gg = gg.sort_values('year')
        for _, r in gg.iterrows():
            rows.append({'risk': RISKS[rei], 'measure': MEASURES.get(measure, measure), 'sex': sex,
                         'age': r.age_name, 'metric': r.metric_name, 'year': int(r.year),
                         'val': r.val, 'lower': r.lower, 'upper': r.upper})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'attributable_global.csv'), index=False)
    print(f"attributable_global rows: {len(out)}")
    # headline 2023 both-sex 15-49 deaths PAF & number
    h = g[(g.rei == 104) & (g.measure == 1) & (g.sex_query == 'both') & (g.age == 24) & (g.year == 2023)]
    for _, r in h.iterrows():
        print(f"  2023 metabolic-attributable EO cancer deaths [{r.metric_name}]: {r.val:.3f} ({r.lower:.3f}-{r.upper:.3f})")

def run_country(df):
    c = df[(df.location.isin(COUNTRIES)) & (df.year == 2023) & (df.sex_query == 'both') & (df.age == 24)].copy()
    rows = []
    for (rei, measure, metric), gg in c.groupby(['rei', 'measure', 'metric']):
        for _, r in gg.iterrows():
            rows.append({'location_id': int(r.location), 'location': r.location_name, 'risk': RISKS[rei],
                         'measure': MEASURES.get(measure, measure), 'metric': r.metric_name,
                         'val': r.val, 'lower': r.lower, 'upper': r.upper})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'attributable_country_2023.csv'), index=False)
    print(f"attributable_country_2023 rows: {len(out)}")

def run_trends(df):
    from trends import eapc
    g = df[(df.location == 1) & (df.age == 24) & (df.sex_query == 'both')].copy()
    rows = []
    for (rei, measure, metric), gg in g.groupby(['rei', 'measure', 'metric']):
        gg = gg.sort_values('year')
        for (s, e, lab) in [(1990, 2023, '1990-2023'), (2013, 2023, '2013-2023')]:
            res = eapc(gg.year.values, gg.val.values, s, e)
            if res:
                rows.append({'risk': RISKS[rei], 'measure': MEASURES.get(measure, measure), 'metric': gg.metric_name.iloc[0],
                             'period': lab, **res})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'attributable_trends_eapc.csv'), index=False)
    print(f"attributable_trends_eapc rows: {len(out)}")

if __name__ == '__main__':
    df = load_risk_panels()
    if df is None:
        print('No risk panels available yet.'); raise SystemExit(0)
    run_global(df); run_country(df); run_trends(df)