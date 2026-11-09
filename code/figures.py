#!/usr/bin/env python3
"""Publication-quality figures (300 dpi) for Direction A manuscript.
Reads results/*.csv; writes figures/*.png. Skips figures whose inputs are absent.
Style: clean serif-free, colorblind-safe palette, consistent fonts.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                 'axes.spines.right': False, 'axes.labelsize': 11, 'xtick.labelsize': 9,
                 'ytick.labelsize': 9, 'legend.fontsize': 9, 'figure.dpi': 120, 'savefig.dpi': 300,
                 'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.25})
C = {'Incidence': '#0072B2', 'Deaths': '#D55E00', 'DALYs': '#009E73'}
SEX_C = {'Both': '#000000', 'Male': '#0072B2', 'Female': '#CC79A7'}

def _read(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None

def fig1_global_trends():
    d = _read('attributable_global.csv')  # placeholder; use A panel via trends? build from global eapc source
    # Use raw global panel for numbers/rates
    p = os.path.join(ROOT, 'data', 'panels', 'A_global_trends.pkl')
    if not os.path.exists(p): return
    df = pd.read_pickle(p)
    df = df[(df.location == 1) & (df.cause == 1059) & (df.age == 24)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, (measure, mlabel) in enumerate([(6, 'Incidence'), (1, 'Deaths'), (2, 'DALYs')]):
        # numbers (metric 1)
        ax = axes[0, j]
        for sex in ['Both', 'Male', 'Female']:
            g = df[(df.measure == measure) & (df.metric == 1) & (df.sex_name == sex)].sort_values('year')
            ax.plot(g.year, g.val / 1e6, label=sex, color=SEX_C[sex], lw=1.8)
            ax.fill_between(g.year, g.lower / 1e6, g.upper / 1e6, color=SEX_C[sex], alpha=0.12)
        ax.set_title(f'{mlabel} (millions)', fontweight='bold'); ax.set_xlabel('Year'); ax.set_ylabel('Number (millions)')
        if j == 0: ax.legend(frameon=False)
        # rates (metric 3)
        ax = axes[1, j]
        for sex in ['Both', 'Male', 'Female']:
            g = df[(df.measure == measure) & (df.metric == 3) & (df.sex_name == sex)].sort_values('year')
            ax.plot(g.year, g.val, label=sex, color=SEX_C[sex], lw=1.8)
            ax.fill_between(g.year, g.lower, g.upper, color=SEX_C[sex], alpha=0.12)
        ax.set_title(f'{mlabel} rate (15-49)', fontweight='bold'); ax.set_xlabel('Year'); ax.set_ylabel('Rate per 100,000')
    fig.suptitle('Global early-onset cancer (ages 15-49) burden, 1990-2023', fontweight='bold', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(FIG, 'fig1_global_trends.png')); plt.close(fig)
    print('fig1_global_trends.png')

def fig2_eapc_forest():
    d = _read('trends_global_eapc.csv')
    if d is None: return
    d = d[(d.age == '15-49') & (d.period == '1990-2023')]
    fig, ax = plt.subplots(figsize=(7, 5))
    labels, ys = [], []
    y = 0
    for measure in ['Incidence', 'Deaths', 'DALYs']:
        for sex in ['Both', 'Male', 'Female']:
            r = d[(d.measure == measure) & (d.sex == sex)]
            if len(r) == 0: continue
            r = r.iloc[0]
            ax.errorbar(r.eapc, y, xerr=[[r.eapc - r.ci_low], [r.ci_high - r.eapc]], fmt='o',
                        color=C[measure], capsize=3, ms=5)
            labels.append(f'{measure} - {sex}'); ys.append(y); y += 1
    ax.axvline(0, color='grey', ls='--', lw=1)
    ax.set_yticks(ys); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel('EAPC (%) , 1990-2023'); ax.set_title('Estimated annual percentage change (15-49)', fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig2_eapc_forest.png')); plt.close(fig)
    print('fig2_eapc_forest.png')

def fig3_apc():
    d = _read('apc_ie_effects.csv')
    if d is None: return
    d = d[(d.location == 'Global') & (d.measure == 'Incidence')]
    if len(d) == 0: return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, etype, title, xlab in zip(axes, ['age', 'period', 'cohort'],
                                       ['Age effect', 'Period effect', 'Cohort effect'],
                                       ['Age group', 'Period', 'Birth cohort']):
        g = d[d.effect_type == etype].copy()
        g = g.sort_values('level')
        ax.plot(g.level, g.rate_ratio, 'o-', color='#0072B2', lw=1.8)
        ax.axhline(1, color='grey', ls='--', lw=1)
        ax.set_title(title, fontweight='bold'); ax.set_xlabel(xlab); ax.set_ylabel('Rate ratio')
        ax.tick_params(axis='x', rotation=45)
    fig.suptitle('Age-period-cohort intrinsic estimator (global incidence, 15-49)', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(FIG, 'fig3_apc_effects.png')); plt.close(fig)
    print('fig3_apc_effects.png')

def fig4_attributable():
    d = _read('attributable_global.csv')
    if d is None: return
    d = d[(d.risk == 'Metabolic risks') & (d.sex == 'both') & (d.age == '15-49')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    g = d[(d.measure == 'Deaths') & (d.metric == 'Number')].sort_values('year')
    l1, = ax.plot(g.year, g.val / 1e3, label='Deaths', color=C['Deaths'], lw=1.8)
    f1 = ax.fill_between(g.year, g.lower / 1e3, g.upper / 1e3, color=C['Deaths'], alpha=0.12)
    ax.set_ylabel('Attributable deaths (thousands)', color=C['Deaths'])
    ax.tick_params(axis='y', labelcolor=C['Deaths'])
    ax2 = ax.twinx()
    g = d[(d.measure == 'DALYs') & (d.metric == 'Number')].sort_values('year')
    l2, = ax2.plot(g.year, g.val / 1e6, label='DALYs', color=C['DALYs'], lw=1.8)
    f2 = ax2.fill_between(g.year, g.lower / 1e6, g.upper / 1e6, color=C['DALYs'], alpha=0.12)
    ax2.set_ylabel('Attributable DALYs (millions)', color=C['DALYs'])
    ax2.tick_params(axis='y', labelcolor=C['DALYs'])
    ax.set_title('Metabolic-attributable EO cancer burden', fontweight='bold')
    ax.set_xlabel('Year'); ax.legend([l1, l2], ['Deaths', 'DALYs'], frameon=False, loc='upper left')
    ax = axes[1]
    for measure in ['Deaths', 'DALYs']:
        g = d[(d.measure == measure) & (d.metric == 'Percent')].sort_values('year')
        ax.plot(g.year, g.val * 100, label=f'{measure} PAF', color=C[measure], lw=1.8)
        ax.fill_between(g.year, g.lower * 100, g.upper * 100, color=C[measure], alpha=0.12)
    ax.set_title('Population attributable fraction', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('PAF (%)'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig4_metabolic_attributable.png')); plt.close(fig)
    print('fig4_metabolic_attributable.png')

def fig5_sdi_trends():
    p = os.path.join(ROOT, 'data', 'panels', 'BCD_country_eo_both.pkl')
    idmap = json.load(open(os.path.join(ROOT, 'data', 'id_map.json')))
    if not os.path.exists(p): return
    df = pd.read_pickle(p)
    quint = {int(k): v for k, v in idmap['sdi_quintiles'].items()}
    df = df[(df.location.isin(quint)) & (df.measure == 6) & (df.metric == 3) & (df.age == 24)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap('viridis')
    order = ['Low SDI', 'Low-middle SDI', 'Middle SDI', 'High-middle SDI', 'High SDI']
    for i, qname in enumerate(order):
        qid = next((k for k, v in quint.items() if v == qname), None)
        if qid is None: continue
        g = df[df.location == qid].sort_values('year')
        ax.plot(g.year, g.val, label=qname, color=cmap(i / 4), lw=1.8)
    ax.set_xlabel('Year'); ax.set_ylabel('Incidence rate per 100,000 (15-49)')
    ax.set_title('Early-onset cancer incidence by SDI quintile', fontweight='bold'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig5_sdi_quintile_trends.png')); plt.close(fig)
    print('fig5_sdi_quintile_trends.png')

def fig6_inequality():
    d = _read('inequality_indices.csv')
    if d is None: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    dd = d[d.sex == 'both']
    y = np.arange(len(dd))
    ax.barh(y, dd.concentration_index, color='#0072B2')
    ax.set_yticks(y); ax.set_yticklabels(dd.outcome, fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color='grey', ls='--'); ax.set_xlabel('Concentration index')
    ax.set_title('Socioeconomic concentration (both sexes)', fontweight='bold')
    ax = axes[1]
    dd = d[d.sex == 'both'].dropna(subset=['rii'])
    y = np.arange(len(dd))
    ax.barh(y, dd.rii, color='#D55E00')
    ax.set_yticks(y); ax.set_yticklabels(dd.outcome, fontsize=8); ax.invert_yaxis()
    ax.axvline(1, color='grey', ls='--'); ax.set_xlabel('Relative inequality index (RII)')
    ax.set_title('Relative inequality (high vs low SDI)', fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig6_inequality.png')); plt.close(fig)
    print('fig6_inequality.png')

def fig7_decomposition():
    d = _read('decomposition_global.csv')
    if d is None: return
    d = d[d.period == '1990-2023']
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for _, r in d.iterrows():
        if r.outcome == 'total_deaths':
            rows.append(('Total EO cancer deaths', r.population_effect, r.rate_effect, 0))
        else:
            rows.append(('Metabolic-attributable deaths', r.population_effect, r.rate_effect, r.paf_effect))
    x = np.arange(len(rows)); w = 0.6
    pop = [r[1] / 1e3 for r in rows]; rate = [r[2] / 1e3 for r in rows]; paf = [r[3] / 1e3 for r in rows]
    ax.bar(x, pop, w, label='Population growth', color='#0072B2')
    ax.bar(x, rate, w, bottom=pop, label='Baseline rate change', color='#009E73')
    bottom2 = [pop[i] + rate[i] for i in range(len(rows))]
    ax.bar(x, paf, w, bottom=bottom2, label='Risk exposure (PAF) change', color='#D55E00')
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel('Contribution to change (thousands of deaths)')
    ax.set_title('Das Gupta decomposition, 1990-2023 (global)', fontweight='bold'); ax.legend(frameon=False)
    ax.axhline(0, color='grey', lw=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig7_decomposition.png')); plt.close(fig)
    print('fig7_decomposition.png')

def fig8_projections():
    d = _read('projections_number.csv')
    r = _read('projections_rate.csv')
    if d is None and r is None: return
    measures = ['Incidence', 'Deaths', 'DALYs']
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.5))
    for i, measure in enumerate(measures):
        ax = axes[i][0]
        if r is not None:
            g = r[(r.location == 'Global') & (r.age == '15-49') & (r.measure == measure)].sort_values('year')
            if len(g):
                ax.plot(g.year, g.projected_rate, color=C[measure], lw=1.8)
                ax.fill_between(g.year, g.pi_low, g.pi_high, color=C[measure], alpha=0.15)
        ax.set_title(f'{measure}: projected rate to 2050', fontweight='bold')
        ax.set_ylabel('Rate per 100,000'); ax.set_xlabel('Year')
        ax = axes[i][1]
        if d is not None:
            g = d[(d.location == 'Global') & (d.measure == measure)].sort_values('year')
            if len(g):
                ax.plot(g.year, g.projected_number / 1e6, color=C[measure], lw=1.8)
                ax.fill_between(g.year, g.pi_low / 1e6, g.pi_high / 1e6, color=C[measure], alpha=0.15)
        ax.set_title(f'{measure}: projected number to 2050', fontweight='bold')
        ax.set_ylabel('Number (millions)'); ax.set_xlabel('Year')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig8_projections_2050.png')); plt.close(fig)
    print('fig8_projections_2050.png')

if __name__ == '__main__':
    for fn in [fig1_global_trends, fig2_eapc_forest, fig3_apc, fig4_attributable,
               fig5_sdi_trends, fig6_inequality, fig7_decomposition, fig8_projections]:
        try: fn()
        except Exception as e:
            print(f'{fn.__name__} skipped: {e}')