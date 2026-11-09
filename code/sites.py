#!/usr/bin/env python3
"""Site-specific early-onset cancer burden, trends, and metabolic attribution (29 sites).

Inputs: F_sites_m{1,2,6}_a{24,27} (global+SDI+regions), K_risk{104,105,108}_sites_m{1,2}_a{24,27}.
Outputs (results/): sites_global_2023.csv, sites_eapc.csv, sites_attributable_2023.csv
Figure: fig9_sites_bar.png (top sites by incidence & deaths, 15-49, 2023).
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results'); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
MEASURES = {1: 'Deaths', 2: 'DALYs', 6: 'Incidence'}

def load_sites():
    frames = []
    for m in [1, 2, 6]:
        for a in [24, 27]:
            p = os.path.join(PANELS, f'F_sites_m{m}_a{a}.pkl')
            if os.path.exists(p):
                d = pd.read_pickle(p); frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None

def run():
    df = load_sites()
    if df is None: print('No site panels yet.'); return
    g = df[(df.location == 1) & (df.age == 24) & (df.year == 2023) & (df.sex == 3)]
    rows = []
    for (cause, measure, metric), gg in g.groupby(['cause', 'measure', 'metric']):
        r = gg.iloc[0]
        rows.append({'cause_id': int(cause), 'cause': r.cause_name, 'measure': MEASURES.get(measure, measure),
                     'metric': r.metric_name, 'val': r.val, 'lower': r.lower, 'upper': r.upper})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'sites_global_2023.csv'), index=False)
    # EAPC by site (incidence & deaths rate, 15-49)
    from trends import eapc
    d = df[(df.location == 1) & (df.age == 24) & (df.metric == 3) & (df.sex == 3)]
    er = []
    for (cause, measure), gg in d.groupby(['cause', 'measure']):
        gg = gg.sort_values('year')
        e = eapc(gg.year.values, gg.val.values, 1990, 2023)
        if e: er.append({'cause': gg.cause_name.iloc[0], 'measure': MEASURES.get(measure, measure), **e})
    pd.DataFrame(er).to_csv(os.path.join(RES, 'sites_eapc.csv'), index=False)
    # metabolic attribution by site (2023, deaths PAF & number)
    ar = []
    for rei in [104, 105, 108]:
        p = os.path.join(PANELS, f'K_risk{rei}_sites_m1_a24.pkl')
        if not os.path.exists(p): continue
        kd = pd.read_pickle(p)
        kd = kd[(kd.location == 1) & (kd.year == 2023) & (kd.sex == 3) & (kd.age == 24)]
        for (cause, metric), gg in kd.groupby(['cause', 'metric']):
            r = gg.iloc[0]
            ar.append({'rei': rei, 'cause': r.cause_name, 'metric': r.metric_name, 'val': r.val, 'lower': r.lower, 'upper': r.upper})
    pd.DataFrame(ar).to_csv(os.path.join(RES, 'sites_attributable_2023.csv'), index=False)
    # figure: top sites
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, measure, title in [(axes[0], 'Incidence', 'Incidence'), (axes[1], 'Deaths', 'Deaths')]:
        s = out[(out.measure == measure) & (out.metric == 'Number')].sort_values('val', ascending=True).tail(10)
        ax.barh(s.cause, s.val / 1e3, color='#0072B2' if measure == 'Incidence' else '#D55E00')
        ax.set_xlabel('Number (thousands), 2023'); ax.set_title(f'Early-onset {title} by site', fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig9_sites_bar.png')); plt.close(fig)
    print(f"sites: global2023={len(out)}, eapc={len(er)}, attributable={len(ar)}; fig9 saved")

if __name__ == '__main__':
    run()