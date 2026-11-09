#!/usr/bin/env python3
"""Regenerate the nine display figures for the GBD 2023 early-onset cancer manuscript.

Every plotted value is read from results/*.csv. Style follows the journal spec:
300 dpi RGB PNG, 180 mm width, DejaVu Sans 7-9 pt, Okabe-Ito palette,
panel labels A/B/C/D bold 10 pt at top-left, no figure titles, thin axes.

Six are main figures and three are supplementary; the file names below encode the
final submission numbering (see manuscript/renumbering_map.md):
Outputs: figures/fig1_global_burden.png, figures/fig2_acceleration.png,
figures/fig3_regions.png, figures/fig4_attribution.png, figures/fig5_inequality.png,
figures/fig6_decomposition.png, figures/figS1_sites.png,
figures/figS2_apc_ie.png, figures/figS3_projections.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito
BLUE = '#0072B2'
VERM = '#D55E00'
GREEN = '#009E73'
PINK = '#CC79A7'
ORANGE = '#E69F00'
SKY = '#56B4E9'
YELLOW = '#F0E442'
BLACK = '#000000'
GREY = '#7f7f7f'
LIGHTBLUE = '#9ecae1'

MEAS_C = {'Incidence': BLUE, 'Deaths': VERM, 'DALYs': GREEN}

rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7.5,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'lines.linewidth': 0.8,
    'lines.markersize': 2.8,
    'patch.linewidth': 0.6,
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'axes.grid': False,
    'mathtext.default': 'regular',
})

MM = 1 / 25.4


def mm(w, h):
    return (w * MM, h * MM)


def read(name):
    return pd.read_csv(os.path.join(RES, name))


def panel(ax, s, x=-0.17, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='top', ha='left', clip_on=False)


def num(v, nd=1):
    return f'{v:.{nd}f}'.replace('-', '\u2212')


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    # vector companion for line art (journal accepts PDF)
    pdf = os.path.splitext(p)[0] + '.pdf'
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    im = Image.open(p)
    im.convert('RGB').save(p, dpi=(300, 300), optimize=True)
    print('wrote', name, '+ pdf')


# ---------------------------------------------------------------- figure 1
def fig1():
    g = read('global_asr_series.csv')
    s = read('sex_asr_series.csv')
    fig, axs = plt.subplots(2, 2, figsize=mm(196, 152))
    fig.subplots_adjust(left=0.080, right=0.925, top=0.955, bottom=0.085,
                        wspace=0.42, hspace=0.42)

    # A -------------------------------------------------------------
    ax = axs[0, 0]
    ax2 = ax.twinx()
    inc = g[g.measure == 'Incidence'].sort_values('year')
    dea = g[g.measure == 'Deaths'].sort_values('year')
    dal = g[g.measure == 'DALYs'].sort_values('year')
    ax.fill_between(inc.year, inc.count_low / 1e6, inc.count_high / 1e6,
                    color=BLUE, alpha=0.18, lw=0)
    ax.fill_between(dea.year, dea.count_low / 1e6, dea.count_high / 1e6,
                    color=VERM, alpha=0.18, lw=0)
    ax.plot(inc.year, inc['count'] / 1e6, color=BLUE, lw=0.8, label='Incidence')
    ax.plot(dea.year, dea['count'] / 1e6, color=VERM, lw=0.8, label='Deaths')
    ax2.plot(dal.year, dal['count'] / 1e6, color=GREEN, lw=0.8, label='DALYs')
    # upper limit leaves room for the 2023 incidence ribbon (3.82 million)
    ax.set_ylim(0, 4.0)
    ax2.set_ylim(0, 70)
    ax.set_xlim(1990, 2024)
    ax.set_xlabel('Year')
    ax.set_ylabel('Cases or deaths (millions)')
    ax2.set_ylabel('DALYs (millions)')
    ax2.spines['right'].set_linewidth(0.6)
    h, l = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h + h2, l + l2, loc='lower right', handlelength=1.4,
              borderaxespad=0.3, labelspacing=0.25)
    panel(ax, 'A')

    # B -------------------------------------------------------------
    ax = axs[0, 1]
    ax.fill_between(inc.year, inc.crude_low, inc.crude_high, color=ORANGE,
                    alpha=0.14, lw=0)
    ax.fill_between(inc.year, inc.asr_low, inc.asr_high, color=BLUE,
                    alpha=0.14, lw=0)
    ax.plot(inc.year, inc.crude_rate, color=ORANGE, lw=0.8, label='Crude rate')
    ax.plot(inc.year, inc.asr_15_49, color=BLUE, lw=0.8,
            label='Age-standardised (15\u201349)')
    i0 = inc.iloc[0]
    i1 = inc.iloc[-1]
    ax.annotate(num(i0.crude_rate, 1), (i0.year, i0.crude_rate),
                xytext=(4, -7), textcoords='offset points', fontsize=6,
                color=ORANGE)
    ax.annotate(num(i0.asr_15_49, 1), (i0.year, i0.asr_15_49),
                xytext=(4, 3), textcoords='offset points', fontsize=6,
                color=BLUE)
    ax.annotate(num(i1.asr_15_49, 1) + ' (both)', (i1.year, i1.asr_15_49),
                xytext=(-6, 5), textcoords='offset points', fontsize=6,
                ha='right', color=BLACK)
    ax.set_xlim(1990, 2023)
    ax.set_xlabel('Year')
    ax.set_ylabel('Incidence rate (per 100,000)')
    ax.legend(loc='lower left', handlelength=1.6, borderaxespad=0.3,
              labelspacing=0.25, frameon=True, facecolor='white',
              framealpha=0.75, edgecolor='none')
    panel(ax, 'B')

    # C -------------------------------------------------------------
    ax = axs[1, 0]
    idx = {}
    for m in ['Incidence', 'Deaths', 'DALYs']:
        sub = g[g.measure == m].sort_values('year')
        v = sub.asr_15_49 / sub.asr_15_49.iloc[0] * 100
        idx[m] = v
        ax.plot(sub.year, v, color=MEAS_C[m], lw=0.8, label=m)
    ax.axhline(100, color=GREY, lw=0.6, ls=(0, (4, 2)))
    for yr in (2013, 2019):
        yv = idx['Incidence'][g[g.measure == 'Incidence'].year.values == yr].iloc[0]
        ax.plot([yr], [yv], 'o', ms=2.6, color=BLUE, zorder=5)
        ax.annotate(num(yv, 1), (yr, yv), xytext=(0, 5),
                    textcoords='offset points', fontsize=5.8, ha='center',
                    color=BLUE)
    ax.set_xlim(1990, 2023)
    ax.set_xlabel('Year')
    ax.set_ylabel('ASR indexed to 1990 = 100')
    ax.legend(loc='lower left', handlelength=1.4, borderaxespad=0.3,
              labelspacing=0.25)
    panel(ax, 'C')

    # D -------------------------------------------------------------
    ax = axs[1, 1]
    styles = {('Incidence', 'male'): ('Male incidence', BLUE, '-'),
              ('Incidence', 'female'): ('Female incidence', VERM, '-'),
              ('Deaths', 'male'): ('Male deaths', BLUE, (0, (3, 1.6))),
              ('Deaths', 'female'): ('Female deaths', VERM, (0, (3, 1.6)))}
    for (m, sx), (lab, c, ls) in styles.items():
        sub = s[(s.measure == m) & (s.sex == sx.capitalize())].sort_values('year')
        ax.plot(sub.year, sub.asr_15_49, color=c, ls=ls, lw=0.8, label=lab)
    ax.set_xlim(1990, 2023)
    ax.set_ylim(22, 122)
    ax.set_xlabel('Year')
    ax.set_ylabel('ASR (per 100,000)')
    ax.legend(loc='upper left', handlelength=1.8, borderaxespad=0.3,
              labelspacing=0.2, fontsize=6.2)
    panel(ax, 'D')
    save(fig, 'fig1_global_burden.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    g = read('global_asr_series.csv')
    seg = read('asr_joinpoint_segments.csv')
    tr = read('asr_trends.csv')
    strd = read('sex_asr_trends.csv')
    fig, axs = plt.subplots(2, 2, figsize=mm(180, 162))
    fig.subplots_adjust(left=0.175, right=0.975, top=0.955, bottom=0.085,
                        wspace=0.40, hspace=0.45)

    def asr_panel(ax, measure, jp, color):
        sub = g[g.measure == measure].sort_values('year')
        ax.fill_between(sub.year, sub.asr_low, sub.asr_high, color=color,
                        alpha=0.2, lw=0)
        ax.plot(sub.year, sub.asr_15_49, color=color, lw=0.8)
        for y in jp:
            ax.axvline(y, color=GREY, lw=0.6, ls=(0, (1.5, 1.5)))
        segs = seg[(seg.measure == measure) & (seg.sex == 'Both')].sort_values('segment')
        top = sub.asr_high.max()
        bottom = sub.asr_low.min()
        ax.set_ylim(bottom - 0.06 * (top - bottom), top + 0.30 * (top - bottom))
        for _, r in segs.iterrows():
            xm = (r.year_start + r.year_end) / 2
            txt = ('+' if r.apc >= 0 else '\u2212') + f'{abs(r.apc):.2f}'
            ax.text(xm, top + 0.13 * (top - bottom), txt, ha='center',
                    va='bottom', fontsize=6)
        ax.set_xlim(1990, 2023)
        ax.set_xlabel('Year')
        ax.set_ylabel('ASR (per 100,000)')

    asr_panel(axs[0, 0], 'Incidence', [1999, 2013, 2019], BLUE)
    panel(axs[0, 0], 'A')
    asr_panel(axs[0, 1], 'Deaths', [2000, 2012, 2019], VERM)
    panel(axs[0, 1], 'B')

    # C -------------------------------------------------------------
    ax = axs[1, 0]
    rows = []
    for m in ['Incidence', 'Deaths', 'DALYs']:
        r = tr[tr.measure == m].iloc[0]
        rows.append((m, '1990\u20132023', r.eapc_1990_2023,
                     r.eapc_1990_2023_low, r.eapc_1990_2023_high, BLUE))
        rows.append((m, '2013\u20132023', r.eapc_2013_2023,
                     r.eapc_2013_2023_low, r.eapc_2013_2023_high, SKY))
    ys = np.arange(len(rows))[::-1]
    for y, (m, per, v, lo, hi, c) in zip(ys, rows):
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt='o', ms=2.8,
                    color=c, ecolor=c, elinewidth=0.7, capsize=1.8)
    ax.axvline(0, color=BLACK, lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{m}  {per}' for m, per, *_ in rows], fontsize=6.5)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel('EAPC, % per year (95% CI)')
    ax.legend(handles=[Line2D([], [], color=BLUE, marker='o', ms=2.8, lw=0.8,
                              label='1990\u20132023'),
                       Line2D([], [], color=SKY, marker='o', ms=2.8, lw=0.8,
                              label='2013\u20132023')],
              loc='lower right', bbox_to_anchor=(1.0, 1.005), ncol=2,
              handlelength=1.4, labelspacing=0.25, borderaxespad=0)
    panel(ax, 'C', x=-0.40)

    # D -------------------------------------------------------------
    ax = axs[1, 1]
    rows = []
    for m in ['Incidence', 'Deaths']:
        for sx, c in [('Male', BLUE), ('Female', VERM)]:
            r = strd[(strd.measure == m) & (strd.sex == sx)].iloc[0]
            rows.append((f'{m}, {sx.lower()}', r.last_apc, r.last_apc_low,
                         r.last_apc_high, r.last_segment, c))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v, lo, hi, per, c) in zip(ys, rows):
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt='o', ms=2.8,
                    color=c, ecolor=c, elinewidth=0.7, capsize=1.8)
        ax.text(hi + 0.12, y, per, va='center', fontsize=5.8, color=GREY)
    ax.axvline(0, color=BLACK, lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlim(0, 5.6)
    ax.set_xlabel('APC of the final joinpoint segment, % per year (95% CI)')
    ax.legend(handles=[Line2D([], [], color=BLUE, marker='o', ms=2.8, lw=0.8,
                              label='Male'),
                       Line2D([], [], color=VERM, marker='o', ms=2.8, lw=0.8,
                              label='Female')],
              loc='lower right', handlelength=1.4, labelspacing=0.25)
    panel(ax, 'D', x=-0.30)
    save(fig, 'fig2_acceleration.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    d = read('apc_ie_effects.csv')
    d = d[d.location == 'Global']
    fig, axs = plt.subplots(1, 3, figsize=mm(180, 122))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.20,
                        wspace=0.30)
    specs = [('age', 'Age group', 45, 'A'),
             ('period', 'Period', 45, 'B'),
             ('cohort', 'Birth cohort', 45, 'C')]
    for ax, (et, xlab, rot, lab) in zip(axs, specs):
        for m, c, mk, ml in [('Incidence', BLUE, 'o', 'Incidence'),
                             ('Deaths', VERM, 's', 'Deaths')]:
            sub = d[(d.effect_type == et) & (d.measure == m)].copy()
            if et == 'age':
                sub = sub.sort_values('level')
            else:
                sub['key'] = sub.level.astype(int)
                sub = sub.sort_values('key')
            x = np.arange(len(sub))
            rr = sub.rate_ratio.values
            lo = np.exp(sub.ci_low.values)
            hi = np.exp(sub.ci_high.values)
            ax.errorbar(x, rr, yerr=[rr - lo, hi - rr], fmt=mk, ms=2.6,
                        color=c, ecolor=c, elinewidth=0.6, capsize=1.5,
                        label=ml, zorder=3)
        ax.set_yscale('log')
        ax.axhline(1, color=GREY, lw=0.6)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub.level.values, rotation=rot,
                           ha='right' if rot else 'center', fontsize=6.2)
        ax.minorticks_off()
        ax.set_yticks([0.5, 1, 2, 5])
        ax.set_yticklabels(['0.5', '1', '2', '5'])
        ax.set_ylim(0.35, 9)
        ax.set_xlabel(xlab)
        ax.set_ylabel('Rate ratio (95% CI)')
        ax.margins(x=0.06)
        panel(ax, lab, x=-0.22)
    axs[0].legend(loc='upper left', handlelength=1.4, labelspacing=0.25)
    save(fig, 'figS2_apc_ie.png')


# ---------------------------------------------------------------- figure 4
REG_ABBR = {
    'Western Europe': 'WEU', 'Eastern Sub-Saharan Africa': 'ESSA',
    'Australasia': 'AUS', 'High-income North America': 'HNA',
    'Central Sub-Saharan Africa': 'CSSA', 'Eastern Europe': 'EEU',
    'High-income Asia Pacific': 'HIAP', 'Andean Latin America': 'ALA',
    'Central Europe': 'CEU', 'Southern Latin America': 'SLA',
    'East Asia': 'EA', 'Caribbean': 'CAR', 'Southeast Asia': 'SEA',
    'Tropical Latin America': 'TLA', 'Southern Sub-Saharan Africa': 'SSSA',
    'Oceania': 'OCE', 'Central Latin America': 'CLA',
    'Western Sub-Saharan Africa': 'WSSA',
    'North Africa and Middle East': 'NAME', 'South Asia': 'SA',
    'Central Asia': 'CA',
}
REG_SHORT = {
    'Western Europe': 'W. Europe', 'Eastern Sub-Saharan Africa': 'E. sub-Sah. Africa',
    'Australasia': 'Australasia', 'High-income North America': 'High-inc. N. America',
    'Central Sub-Saharan Africa': 'C. sub-Sah. Africa', 'Eastern Europe': 'E. Europe',
    'High-income Asia Pacific': 'High-inc. Asia Pacific', 'Andean Latin America': 'Andean Latin Am.',
    'Central Europe': 'C. Europe', 'Southern Latin America': 'S. Latin America',
    'East Asia': 'E. Asia', 'Caribbean': 'Caribbean', 'Southeast Asia': 'SE Asia',
    'Tropical Latin America': 'Trop. Latin America', 'Southern Sub-Saharan Africa': 'S. sub-Sah. Africa',
    'Oceania': 'Oceania', 'Central Latin America': 'C. Latin America',
    'Western Sub-Saharan Africa': 'W. sub-Sah. Africa',
    'North Africa and Middle East': 'N. Africa & Middle East', 'South Asia': 'S. Asia',
    'Central Asia': 'C. Asia',
}


def smart_labels(ax, xs, ys, labels, sizes, fontsize=5.8,
                 vlines=(), hlines=()):
    """Place bubble labels around their markers, avoiding bubbles, other
    labels, the axes frame and any reference lines; a thin leader is drawn
    when a label has to move away from its marker."""
    fig = ax.figure
    fig.canvas.draw()
    tr = ax.transData.transform
    centres = [tr((x, y)) for x, y in zip(xs, ys)]
    radii = [np.sqrt(v / np.pi) * fig.dpi / 72 for v in sizes]
    wpx = [len(t) * fontsize * 0.62 * fig.dpi / 72 for t in labels]
    hpx = fontsize * 1.15 * fig.dpi / 72
    axbb = ax.get_window_extent()
    vpx = [tr((v, ys[0]))[0] for v in vlines]
    hpx_ref = [tr((xs[0], h))[1] for h in hlines]
    placed = []
    order = np.argsort(-np.array(ys))
    angles = np.deg2rad(np.arange(0, 360, 22.5))
    for i in order:
        cx, cy = centres[i]
        w, r = wpx[i], radii[i]
        best, best_score, best_disp = None, 1e18, 0.0
        for step, extra in enumerate((2, 7, 13, 20, 28, 38)):
            for a in angles:
                dx, dy = np.cos(a), np.sin(a)
                d = r + extra
                x0 = cx + dx * d - (w if dx < -0.25 else (w / 2 if abs(dx) <= 0.25 else 0))
                y0 = cy + dy * d - (hpx if dy < -0.25 else (hpx / 2 if abs(dy) <= 0.25 else 0))
                score = 0.8 * step
                for j, (ox, oy) in enumerate(centres):
                    if j == i:
                        continue
                    rr = radii[j] * 0.95
                    if (x0 < ox + rr and x0 + w > ox - rr and
                            y0 < oy + rr and y0 + hpx > oy - rr):
                        score += 500
                for b in placed:
                    if (x0 < b[0] + b[2] + 2.5 and x0 + w + 2.5 > b[0] and
                            y0 < b[1] + b[3] + 2.5 and y0 + hpx + 2.5 > b[1]):
                        score += 1200
                if (x0 < axbb.x0 + 2 or x0 + w > axbb.x1 - 2 or
                        y0 < axbb.y0 + 2 or y0 + hpx > axbb.y1 - 2):
                    score += 2200
                for vx in vpx:
                    if x0 - 1 < vx < x0 + w + 1 and abs(cx - vx) > 4:
                        score += 70
                for hy in hpx_ref:
                    if y0 - 1 < hy < y0 + hpx + 1 and abs(cy - hy) > 4:
                        score += 70
                if score < best_score:
                    best_score, best = score, (x0, y0, w, hpx)
                    best_disp = d - r
        placed.append(best)
        x0, y0 = best[0], best[1]
        far = best_disp > 4
        ax.annotate(labels[i], xy=(xs[i], ys[i]), xycoords='data',
                    xytext=((x0 - centres[i][0]) / fig.dpi * 72,
                            (y0 - centres[i][1]) / fig.dpi * 72),
                    textcoords='offset points', fontsize=fontsize,
                    ha='left', va='bottom', annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=GREY, lw=0.35,
                                    shrinkA=1.2, shrinkB=2.2,
                                    relpos=(0.5, 0.5)) if far else None)


def fig4():
    r = read('regions_2023.csv')
    reg = r[r.level == 'GBD region'].copy()
    glb = r[r.level == 'Global'].iloc[0]
    ra = read('regions_asr_2023.csv')
    ai = ra[ra.measure == 'Incidence'].set_index('region')
    ad = ra[ra.measure == 'Deaths'].set_index('region')
    gs = read('global_asr_series.csv')
    gi = gs[(gs.year == 2023) & (gs.measure == 'Incidence')].asr_15_49.iloc[0]
    gd = gs[(gs.year == 2023) & (gs.measure == 'Deaths')].asr_15_49.iloc[0]
    fig, axs = plt.subplots(2, 2, figsize=mm(180, 152))
    fig.subplots_adjust(left=0.205, right=0.98, top=0.955, bottom=0.085,
                        wspace=0.55, hspace=0.38)

    # A -------------------------------------------------------------
    # directly age-standardised rates (2023 global 15-49 standard population)
    ax = axs[0, 0]
    regs = ai.index.tolist()                    # CSV order
    xi = ai.asr.values
    yi = ad.loc[regs, 'asr'].values
    s = 8 + 0.20 * ai.population.values / 1e6    # 2023 population aged 15-49
    ax.scatter(xi, yi, s=s, color=BLUE, alpha=0.55, edgecolor=BLUE,
               linewidth=0.5, zorder=3)
    ax.axvline(gi, color=GREY, lw=0.6, ls=(0, (3, 2)))
    ax.axhline(gd, color=GREY, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(45, 168)
    ax.set_ylim(11, 80)
    ax.set_xlabel('Incidence ASR, 2023 (per 100,000)')
    ax.set_ylabel('Deaths ASR, 2023 (per 100,000)')
    smart_labels(ax, xi, yi, [REG_ABBR[x] for x in regs], s, fontsize=5.2)
    ax.text(0.02, 0.99, 'Bubble area \u221d 2023 population',
            transform=ax.transAxes, fontsize=5.4, color=GREY, va='top',
            bbox=dict(fc='white', ec='none', pad=1.0))
    ax.text(167, gd + 0.9, 'global', fontsize=5.4, color=GREY, ha='right',
            va='bottom')
    ax.text(gi + 2.6, 11.6, 'global', fontsize=5.4, color=GREY, va='bottom')
    panel(ax, 'A', x=-0.26)

    # B -------------------------------------------------------------
    ax = axs[0, 1]
    d = reg.sort_values('paf_deaths_2023')
    ax.barh(np.arange(len(d)), d.paf_deaths_2023, height=0.72, color=BLUE)
    ax.axvline(glb.paf_deaths_2023, color=VERM, lw=0.7, ls=(0, (4, 2)))
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([REG_SHORT[x] for x in d.location], fontsize=6)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel('PAF of deaths, 2023 (%)')
    ax.text(glb.paf_deaths_2023 + 0.08, -0.62, f'global {num(glb.paf_deaths_2023, 2)}%',
            fontsize=5.6, color=VERM, va='bottom')
    panel(ax, 'B', x=-0.52)

    # C -------------------------------------------------------------
    ax = axs[1, 0]
    d = reg.sort_values('attributable_deaths_rate_2023')
    ax.barh(np.arange(len(d)), d.attributable_deaths_rate_2023, height=0.72,
            color=GREEN)
    ax.axvline(glb.attributable_deaths_rate_2023, color=VERM, lw=0.7,
               ls=(0, (4, 2)))
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([REG_SHORT[x] for x in d.location], fontsize=6)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel('Metabolic-attributable deaths, 2023 (per 100,000)')
    ax.text(glb.attributable_deaths_rate_2023 + 0.03, -0.62,
            f'global {num(glb.attributable_deaths_rate_2023, 2)}',
            fontsize=5.6, color=VERM, va='bottom')
    panel(ax, 'C', x=-0.52)

    # D -------------------------------------------------------------
    ax = axs[1, 1]
    d = reg.sort_values('attributable_deaths_number_2023', ascending=False).head(12)
    d = d.sort_values('attributable_deaths_number_2023')
    ax.barh(np.arange(len(d)), d.attributable_deaths_number_2023 / 1e3,
            height=0.72, color=ORANGE)
    for i, v in enumerate(d.attributable_deaths_number_2023 / 1e3):
        ax.text(v + 0.15, i, f'{v:.1f}k', va='center', fontsize=5.6)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([REG_SHORT[x] for x in d.location], fontsize=6)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlim(0, 11.8)
    ax.set_xlabel('Attributable deaths, 2023 (thousands)')
    panel(ax, 'D', x=-0.52)
    save(fig, 'fig3_regions.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    g = read('attributable_global.csv')
    g = g[(g.sex == 'both') & (g.age == '15-49')]
    ac = read('attributable_country_2023.csv')
    fig, axs = plt.subplots(2, 2, figsize=mm(218, 152))
    fig.subplots_adjust(left=0.165, right=0.955, top=0.955, bottom=0.09,
                        wspace=0.44, hspace=0.42)

    met_d = g[(g.risk == 'Metabolic risks') & (g.measure == 'Deaths') &
              (g.metric == 'Number')].sort_values('year')
    met_y = g[(g.risk == 'Metabolic risks') & (g.measure == 'DALYs') &
              (g.metric == 'Number')].sort_values('year')

    # A -------------------------------------------------------------
    ax = axs[0, 0]
    ax2 = ax.twinx()
    ax.fill_between(met_d.year, met_d.lower / 1e3, met_d.upper / 1e3,
                    color=VERM, alpha=0.18, lw=0)
    ax.plot(met_d.year, met_d.val / 1e3, color=VERM, lw=0.8)
    ax2.fill_between(met_y.year, met_y.lower / 1e6, met_y.upper / 1e6,
                     color=GREEN, alpha=0.18, lw=0)
    ax2.plot(met_y.year, met_y.val / 1e6, color=GREEN, lw=0.8)
    ax.set_xlim(1990, 2023)
    ax.set_ylim(0, 75)
    ax2.set_ylim(0, 5.0)
    ax.set_xlabel('Year')
    ax.set_ylabel('Attributable deaths (thousands)')
    ax2.set_ylabel('Attributable DALYs (millions)')
    ax2.spines['right'].set_linewidth(0.6)
    ax.legend(handles=[Line2D([], [], color=VERM, lw=0.8, label='Deaths (left)'),
                       Line2D([], [], color=GREEN, lw=0.8, label='DALYs (right)')],
              loc='upper left', handlelength=1.4, labelspacing=0.25)
    panel(ax, 'A')

    # B -------------------------------------------------------------
    ax = axs[0, 1]
    for meas, c, lab in [('Deaths', VERM, 'Deaths'), ('DALYs', GREEN, 'DALYs')]:
        sub = g[(g.risk == 'Metabolic risks') & (g.measure == meas) &
                (g.metric == 'Percent')].sort_values('year')
        ax.fill_between(sub.year, sub.lower * 100, sub.upper * 100, color=c,
                        alpha=0.18, lw=0)
        ax.plot(sub.year, sub.val * 100, color=c, lw=0.8, label=lab)
    ax.set_xlim(1990, 2023)
    ax.set_xlabel('Year')
    ax.set_ylabel('PAF of burden (%)')
    ax.legend(loc='upper left', handlelength=1.4, labelspacing=0.25)
    panel(ax, 'B')

    # C -------------------------------------------------------------
    ax = axs[1, 0]
    for risk, c, lab in [('High body-mass index', PINK, 'High BMI'),
                         ('High fasting plasma glucose', ORANGE, 'High FPG')]:
        sub = g[(g.risk == risk) & (g.measure == 'Deaths') &
                (g.metric == 'Number')].sort_values('year')
        ax.fill_between(sub.year, sub.lower / 1e3, sub.upper / 1e3, color=c,
                        alpha=0.20, lw=0)
        ax.plot(sub.year, sub.val / 1e3, color=c, lw=0.8, label=lab)
    ax.set_xlim(1990, 2023)
    ax.set_xlabel('Year')
    ax.set_ylabel('Attributable deaths (thousands)')
    ax.legend(loc='upper left', handlelength=1.4, labelspacing=0.25)
    panel(ax, 'C')

    # D -------------------------------------------------------------
    ax = axs[1, 1]
    d = ac[(ac.risk == 'Metabolic risks') & (ac.measure == 'Deaths') &
           (ac.metric == 'Percent')].copy()
    for c in ['val', 'lower', 'upper']:
        d[c] = d[c] * 100
    d = d.sort_values('val')
    top = d.tail(10)
    bot = d.head(10)
    dd = pd.concat([top, bot]).sort_values('val')
    y = np.arange(len(dd))
    xerr = np.vstack([dd.val - dd.lower, dd.upper - dd.val])
    ax.barh(y, dd.val, height=0.7, color=BLUE)
    ax.errorbar(dd.val, y, xerr=xerr, fmt='none', ecolor=BLACK,
                elinewidth=0.6, capsize=1.3)
    ax.axvline(3.956100, color=VERM, lw=0.7, ls=(0, (4, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(dd.location, fontsize=5.7)
    ax.set_ylim(-0.7, len(dd) - 0.3)
    ax.set_xlim(-1.5, 22)
    ax.set_xlabel('Metabolic PAF of deaths, 2023 (%)')
    ax.text(4.15, 0.15, 'global 3.96%', fontsize=5.6, color=VERM, va='bottom')
    ax.axhline(9.5, color=GREY, lw=0.5, ls=(0, (2, 2)))
    ax.text(21.6, 14.5, 'top 10', fontsize=5.6, color=GREY, va='center',
            ha='center', rotation=90)
    ax.text(21.6, 4.5, 'bottom 10', fontsize=5.6, color=GREY, va='center',
            ha='center', rotation=90)
    panel(ax, 'D', x=-0.36)
    save(fig, 'fig4_attribution.png')


# ---------------------------------------------------------------- figure 6
SITE_SHORT = {
    'Tracheal, bronchus, and lung cancer': 'Trachea/bronchus/lung',
    'Brain and central nervous system cancer': 'Brain/CNS',
    'Colon and rectum cancer': 'Colon/rectum',
    'Gallbladder and biliary tract cancer': 'Gallbladder/biliary',
    'Lip and oral cavity cancer': 'Lip/oral cavity',
    'Non-melanoma skin cancer': 'Non-melanoma skin',
    'Other malignant neoplasms': 'Other malignant',
    'Non-Hodgkin lymphoma': 'Non-Hodgkin lymphoma',
    'Malignant skin melanoma': 'Melanoma',
    'Multiple myeloma': 'Multiple myeloma',
}


def sname(s):
    return SITE_SHORT.get(s, s.replace(' cancer', ''))


def fig6():
    sg = read('sites_global_2023.csv')
    se = read('sites_eapc.csv')
    sp = read('sites_paf_2023.csv')
    fig, axs = plt.subplots(2, 2, figsize=mm(180, 162))
    fig.subplots_adjust(left=0.205, right=0.975, top=0.955, bottom=0.085,
                        wspace=0.62, hspace=0.42)

    # A -------------------------------------------------------------
    ax = axs[0, 0]
    inc = sg[(sg.measure == 'Incidence') & (sg.metric == 'Number')].set_index('cause')
    dea = sg[(sg.measure == 'Deaths') & (sg.metric == 'Number')].set_index('cause')
    d = inc.sort_values('val', ascending=False).head(15)
    d = d.assign(deaths=dea.loc[d.index, 'val'])
    d = d.sort_values('val')
    y = np.arange(len(d))
    ax.hlines(y, d.deaths, d.val, color=GREY, lw=0.6, zorder=1)
    ax.scatter(d.deaths, y, marker='s', s=9, color=VERM, zorder=3,
               label='Deaths')
    ax.scatter(d.val, y, marker='o', s=11, color=BLUE, zorder=3,
               label='Cases')
    for i, (v, dv) in enumerate(zip(d.val, d.deaths)):
        ax.text(v * 1.12, i, f'{v / 1e3:.0f}k', va='center', fontsize=5.4)
        ax.text(dv * 0.88, i, f'{dv / 1e3:.0f}k', va='center', ha='right',
                fontsize=5.4, color=VERM)
    ax.set_xscale('log')
    ax.set_xlim(1.5e3, 2.6e6)
    ax.set_yticks(y)
    ax.set_yticklabels([sname(c) for c in d.index], fontsize=6)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel('Annual cases or deaths, 2023 (log scale)')
    ax.legend(loc='lower right', handletextpad=0.1, borderaxespad=0.2,
              labelspacing=0.2, fontsize=6)
    panel(ax, 'A', x=-0.55, y=1.06)

    # B -------------------------------------------------------------
    ax = axs[0, 1]
    d = se[(se.measure == 'Incidence')].sort_values('eapc')
    y = np.arange(len(d))
    ax.errorbar(d.eapc, y, xerr=[d.eapc - d.ci_low, d.ci_high - d.eapc],
                fmt='o', ms=2.2, color=BLUE, ecolor=BLUE, elinewidth=0.6,
                capsize=1.2, lw=0)
    ax.axvline(0, color=BLACK, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([sname(c) for c in d.cause], fontsize=4.7)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlabel('Incidence EAPC, 1990\u20132023 (% per year, 95% CI)')
    panel(ax, 'B', x=-0.55)

    # C -------------------------------------------------------------
    ax = axs[1, 0]
    d = sp[sp.measure == 'Deaths'].sort_values('paf_percent')
    y = np.arange(len(d))
    ax.barh(y, d.paf_percent, height=0.7, color=ORANGE)
    for i, v in enumerate(d.paf_percent):
        ax.text(v + 0.4, i, num(v, 1), va='center', fontsize=5.4)
    ax.set_yticks(y)
    ax.set_yticklabels([sname(c) for c in d.site], fontsize=6)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlim(0, 37)
    ax.set_xlabel('Metabolic PAF of deaths, 2023 (%)')
    panel(ax, 'C', x=-0.55)

    # D -------------------------------------------------------------
    ax = axs[1, 1]
    d = sp[sp.measure == 'Deaths'].sort_values('attributable_number')
    y = np.arange(len(d))
    v = d.attributable_number / 1e3
    lo = d.attributable_low / 1e3
    hi = d.attributable_high / 1e3
    xerr = np.vstack([v - lo, hi - v])
    ax.barh(y, v, height=0.7, color=PINK)
    ax.errorbar(v, y, xerr=xerr, fmt='none', ecolor=BLACK, elinewidth=0.6,
                capsize=1.3)
    ax.axvline(0, color=BLACK, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([sname(c) for c in d.site], fontsize=6)
    ax.set_ylim(-0.7, len(d) - 0.3)
    # full 95% UI shown for every site (breast spans -9.8k to +10.3k)
    ax.set_xlim(-11.2, 12.4)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xlabel('Attributable deaths, 2023 (thousands)')
    ib = [k for k, c in enumerate(d.site) if sname(c) == 'Breast'][0]
    ax.text(-10.9, 1.25,
            f'Breast {num(v.iloc[ib], 1)}k\n'
            f'95% UI {num(lo.iloc[ib], 1)} to {num(hi.iloc[ib], 1)}',
            fontsize=5.2, color=GREY, ha='left', va='bottom', linespacing=1.4)
    panel(ax, 'D', x=-0.55)
    save(fig, 'figS1_sites.png')


# ---------------------------------------------------------------- figure 7
def fig7():
    ii = read('inequality_indices.csv')
    it = read('inequality_trends.csv')
    cd = read('inequality_country_data.csv')
    fig, axs = plt.subplots(2, 2, figsize=mm(198, 152))
    fig.subplots_adjust(left=0.215, right=0.975, top=0.955, bottom=0.09,
                        wspace=0.55, hspace=0.42)

    # A -------------------------------------------------------------
    ax = axs[0, 0]
    rows = [('Total Incidence (15-49 ASR)', 'Incidence (ASR)'),
            ('Total Deaths (15-49 ASR)', 'Deaths (ASR)'),
            ('Total DALYs (15-49 ASR)', 'DALYs (ASR)'),
            ('Metabolic PAF (deaths, %)', 'Metabolic PAF, deaths'),
            ('Metabolic-attributable death rate', 'Metabolic death rate')]
    ys = np.arange(len(rows))[::-1]
    for y, (o, lab) in zip(ys, rows):
        r = ii[(ii.outcome == o) & (ii.sex == 'both')].iloc[0]
        ax.errorbar(r.rii, y, xerr=[[r.rii - r.rii_ci_low], [r.rii_ci_high - r.rii]],
                    fmt='o', ms=2.8, color=BLUE, ecolor=BLUE, elinewidth=0.7,
                    capsize=1.8)
    ax.axvline(1, color=BLACK, lw=0.6)
    ax.set_xscale('log')
    ax.set_xticks([0.2, 0.5, 1, 2, 5])
    ax.set_xticklabels(['0.2', '0.5', '1', '2', '5'])
    ax.minorticks_off()
    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in rows], fontsize=6.3)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0.15, 6)
    ax.set_xlabel('RII, 2023 (95% CI, log scale)')
    panel(ax, 'A', x=-0.43)

    # B -------------------------------------------------------------
    ax = axs[0, 1]
    for o, c, lab in [('Total Incidence (15-49 ASR)', BLUE, 'Incidence (ASR)'),
                      ('Total Deaths (15-49 ASR)', VERM, 'Deaths (ASR)'),
                      ('Metabolic PAF (deaths, %)', ORANGE, 'Metabolic PAF')]:
        d = it[it.outcome == o].sort_values('year')
        ax.plot(d.year, d.rii, color=c, lw=0.8, label=lab)
    ax.axhline(1, color=BLACK, lw=0.6)
    ax.set_yscale('log')
    ax.set_yticks([0.2, 0.5, 1, 2, 5])
    ax.set_yticklabels(['0.2', '0.5', '1', '2', '5'])
    ax.minorticks_off()
    ax.set_ylim(0.15, 6)
    ax.set_xlim(1990, 2023)
    ax.set_xlabel('Year')
    ax.set_ylabel('RII (log scale)')
    ax.legend(loc='lower left', handlelength=1.4, labelspacing=0.25,
              fontsize=6)
    panel(ax, 'B')

    # C -------------------------------------------------------------
    ax = axs[1, 0]
    for o, c, lab in [('Total Deaths (15-49 ASR)', VERM, 'Deaths (ASR)'),
                      ('Total Incidence (15-49 ASR)', BLUE, 'Incidence (ASR)')]:
        d = cd[(cd.outcome == o) & (cd.sex == 'both') & (cd.year == 2023)]
        d = d.sort_values('sdi')
        pop = d['pop'].values
        yv = d['y'].values * pop
        cx = np.concatenate([[0], np.cumsum(pop) / pop.sum()])
        cy = np.concatenate([[0], np.cumsum(yv) / yv.sum()])
        ax.plot(cx, cy, color=c, lw=0.8, label=lab)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=(0, (3, 2)), label='Equality')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Cumulative share of population (ranked by SDI)')
    ax.set_ylabel('Cumulative share of outcome')
    ax.legend(loc='upper left', handlelength=1.4, labelspacing=0.25,
              fontsize=6.2)
    panel(ax, 'C')

    # D -------------------------------------------------------------
    ax = axs[1, 1]
    rows = []
    for o, lab in [('Total Incidence (15-49 rate)', 'Incidence'),
                   ('Total Deaths (15-49 rate)', 'Deaths'),
                   ('Total DALYs (15-49 rate)', 'DALYs')]:
        for sx, c in [('male', BLUE), ('female', VERM)]:
            r = ii[(ii.outcome == o) & (ii.sex == sx)].iloc[0]
            rows.append((f'{lab}, {sx}', r.rii, r.rii_ci_low, r.rii_ci_high,
                         r.sex, c))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v, lo, hi, sx, c) in zip(ys, rows):
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt='o', ms=2.8,
                    color=c, ecolor=c, elinewidth=0.7, capsize=1.8)
    ax.axvline(1, color=BLACK, lw=0.6)
    ax.set_xscale('log')
    ax.set_xticks([0.2, 0.5, 1, 2, 5])
    ax.set_xticklabels(['0.2', '0.5', '1', '2', '5'])
    ax.minorticks_off()
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0.15, 6)
    ax.set_xlabel('RII, 2023 (95% CI, log scale)')
    ax.legend(handles=[Line2D([], [], color=BLUE, marker='o', ms=2.8, lw=0.8,
                              label='Male'),
                       Line2D([], [], color=VERM, marker='o', ms=2.8, lw=0.8,
                              label='Female')],
              loc='lower right', handlelength=1.4, labelspacing=0.25,
              fontsize=6.2)
    panel(ax, 'D', x=-0.30)
    save(fig, 'fig5_inequality.png')


# ---------------------------------------------------------------- figure 8
def waterfall(ax, labels, deltas, total_label, color_pos=BLUE,
              color_neg=VERM, ylab='Deaths (thousands)', nd=1):
    """labels: [start, comp1, ..., end]; deltas: component changes."""
    n = len(labels)
    cum = 0.0
    for i in range(n):
        if i == 0:
            v = deltas[0]
            ax.bar(i, v, width=0.62, color='#b0b0b0')
            ax.text(i, v, num(v, nd), ha='center', va='bottom', fontsize=5.8)
            cum = v
        elif i == n - 1:
            ax.bar(i, deltas[-1], width=0.62, color='#b0b0b0')
            ax.text(i, deltas[-1], num(deltas[-1], nd), ha='center',
                    va='bottom', fontsize=5.8)
            ax.plot([i - 1 + 0.31, i - 0.31], [cum, cum], color=GREY, lw=0.5,
                    ls=(0, (2, 2)))
            cum = deltas[-1]
        else:
            d = deltas[i]
            bottom = cum if d >= 0 else cum + d
            ax.bar(i, abs(d), bottom=bottom, width=0.62,
                   color=color_pos if d >= 0 else color_neg)
            if d >= 0:
                ax.text(i, cum + d, num(d, nd), ha='center', va='bottom',
                        fontsize=5.8)
            else:
                ax.text(i, cum + d, num(d, nd), ha='center', va='top',
                        fontsize=5.8)
            ax.plot([i - 0.31 - 0.38, i - 0.31], [cum, cum], color=GREY,
                    lw=0.5, ls=(0, (2, 2)))
            cum += d
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=6.4)
    ax.set_ylabel(ylab)
    ax.axhline(0, color=BLACK, lw=0.6)


def fig8():
    dg = read('decomposition_global.csv')
    kt = read('age_composition_decomposition.csv')
    fig, axs = plt.subplots(2, 2, figsize=mm(180, 142))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.95, bottom=0.095,
                        wspace=0.42, hspace=0.45)

    tot = dg[(dg.outcome == 'total_deaths') & (dg.period == '1990-2023')].iloc[0]
    att = dg[(dg.outcome == 'attributable_deaths') &
             (dg.period == '1990-2023')].iloc[0]

    # A -------------------------------------------------------------
    ax = axs[0, 0]
    waterfall(ax, ['1990', 'Population', 'Rate', '2023'],
              [tot.value_0 / 1e3, tot.population_effect / 1e3,
               tot.rate_effect / 1e3, tot.value_1 / 1e3],
              'total')
    ax.set_ylim(0, 1450)
    panel(ax, 'A')

    # B -------------------------------------------------------------
    ax = axs[0, 1]
    waterfall(ax, ['1990', 'Population', 'Rate', 'PAF', '2023'],
              [att.value_0 / 1e3, att.population_effect / 1e3,
               att.rate_effect / 1e3, att.paf_effect / 1e3,
               att.value_1 / 1e3], 'attr')
    ax.set_ylim(0, 60)
    panel(ax, 'B')

    # C -------------------------------------------------------------
    ax = axs[1, 0]
    periods = ['1990-2023', '2000-2023', '2010-2023']
    comps = [('population_effect', 'Population', BLUE),
             ('rate_effect', 'Rate', VERM),
             ('paf_effect', 'PAF', ORANGE)]
    x = np.arange(len(periods))
    w = 0.24
    for k, (col, lab, c) in enumerate(comps):
        vals = [dg[(dg.outcome == 'attributable_deaths') & (dg.period == p)][col].iloc[0] / 1e3
                for p in periods]
        ax.bar(x + (k - 1) * w, vals, width=w, color=c, label=lab)
    for i, p in enumerate(periods):
        r = dg[(dg.outcome == 'attributable_deaths') & (dg.period == p)].iloc[0]
        net = (r.value_1 - r.value_0) / 1e3
        sg = '+' if net >= 0 else '\u2212'
        ax.text(i, 15.4, f'net {sg}{abs(net):.1f}k', ha='center', fontsize=6)
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(['1990\u20132023', '2000\u20132023', '2010\u20132023'],
                       fontsize=6.6)
    ax.set_ylim(-7, 25)
    ax.set_ylabel('Change in attributable deaths\n(thousands)')
    ax.legend(loc='upper right', ncol=1, handlelength=1.1, labelspacing=0.2,
              fontsize=6)
    panel(ax, 'C', x=-0.30)

    # D -------------------------------------------------------------
    ax = axs[1, 1]
    measures = ['Incidence', 'Deaths', 'DALYs']
    comps = [('rate_effect', 'Rate effect', BLUE),
             ('composition_effect', 'Composition effect', ORANGE),
             ('interaction', 'Interaction', GREY)]
    x = np.arange(len(measures))
    for i, m in enumerate(measures):
        r = kt[kt.measure == m].iloc[0]
        cum = 0.0
        for col, lab, c in comps:
            v = r[col]
            bottom = cum if v >= 0 else cum + v
            ax.bar(i, abs(v), bottom=bottom, width=0.6, color=c,
                   label=lab if i == 0 else None)
            cum += v
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m}\n(total {num(kt[kt.measure == m].total_change.iloc[0], 1)})'
                        for m in measures], fontsize=6.4)
    ax.set_ylabel('Change in crude rate, 1990\u20132023\n(per 100,000)')
    ax.set_ylim(-400, 300)
    ax.legend(loc='lower left', handlelength=1.1, labelspacing=0.2, fontsize=6)
    share = kt[kt.measure == 'Incidence'].composition_share_percent.iloc[0]
    # zoom on the two small measures (same component colours)
    axi = ax.inset_axes([0.135, 0.585, 0.42, 0.35])
    axi.set_facecolor('white')
    axi.patch.set_alpha(1.0)
    axi.set_zorder(5)
    for i, m in enumerate(['Deaths', 'Incidence']):
        r = kt[kt.measure == m].iloc[0]
        cum = 0.0
        for col, lab, c in comps:
            v = r[col]
            left = cum if v >= 0 else cum + v
            axi.barh(i, abs(v), left=left, height=0.55, color=c)
            cum += v
    axi.axvline(0, color=BLACK, lw=0.5)
    axi.set_yticks([])
    axi.set_xlim(-16, 17)
    axi.set_ylim(-0.55, 1.62)
    axi.set_xticks([-10, 0, 10])
    axi.tick_params(axis='x', labelsize=5, width=0.5, length=1.8)
    for i, m in enumerate(['Deaths', 'Incidence']):
        axi.text(-15.4, i, m, fontsize=5.2, va='center', ha='left')
    for sp in ['top', 'right']:
        axi.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        axi.spines[sp].set_linewidth(0.5)
    axi.text(17, 1.55, 'not to scale', fontsize=4.8, color=GREY,
             va='center', ha='right')
    ax.text(0.575, 0.95,
            'composition effect =\n' + f'{num(share, 1)}% of incidence change',
            transform=ax.transAxes, fontsize=5.2, color=GREY, va='top', ha='left')
    panel(ax, 'D')
    save(fig, 'fig6_decomposition.png')


# ---------------------------------------------------------------- figure 9
def fig9():
    g = read('global_asr_series.csv')
    pasr = read('projection_asr_series.csv')
    sca = read('projections_asr_scenarios_2050.csv')
    sc = read('projections_scenarios_2050.csv')
    rt = read('projection_recent_trend.csv')
    pa = read('projections_attributable.csv')
    ag = read('attributable_global.csv')
    fig, axs = plt.subplots(2, 2, figsize=mm(180, 172))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.955, bottom=0.115,
                        wspace=0.38, hspace=0.48)

    # A -------------------------------------------------------------
    ax = axs[0, 0]
    for m in ['Incidence', 'Deaths', 'DALYs']:
        c = MEAS_C[m]
        obs = g[g.measure == m].sort_values('year')
        ax.plot(obs.year, obs.asr_15_49, color=c, lw=0.8, label=m)
        p = pasr[pasr.measure == m].sort_values('year')
        ax.plot(pd.concat([obs.tail(1), p]).year,
                pd.concat([obs.tail(1), p]).asr, color=c, lw=0.8,
                ls=(0, (3, 1.6)))
        # ribbon anchored on the observed 2023 ASR
        xs = np.concatenate([[obs.year.iloc[-1]], p.year.values])
        lo = np.concatenate([[obs.asr_15_49.iloc[-1]], p.asr_low.values])
        hi = np.concatenate([[obs.asr_15_49.iloc[-1]], p.asr_high.values])
        ax.fill_between(xs, lo, hi, color=c, alpha=0.13, lw=0)
    ax.axvline(2023, color=GREY, lw=0.6, ls=(0, (1.5, 1.5)))
    ax.set_yscale('log')
    ax.set_yticks([20, 50, 100, 200, 500, 1000, 2000, 4000])
    ax.set_yticklabels(['20', '50', '100', '200', '500', '1,000', '2,000', '4,000'])
    ax.minorticks_off()
    ax.set_xlim(1990, 2050)
    ax.set_ylim(15, 4500)
    ax.set_xlabel('Year')
    ax.set_ylabel('ASR (per 100,000, log scale)')
    ax.legend(loc='center left', handlelength=1.6, labelspacing=0.25)
    ax.text(2024, 18, 'projection', fontsize=5.6, color=GREY, va='bottom')
    panel(ax, 'A')

    # B -------------------------------------------------------------
    ax = axs[0, 1]
    obs23 = g[(g.measure == 'Incidence') & (g.year == 2023)]['count'].iloc[0]
    srow = sca[sca.measure == 'Incidence'].set_index('scenario').number_2050
    vals = {'Observed\n2023': obs23}
    vals['Constant\nASR'] = srow['constant_2023_asr_damped_pop']
    vals['ARIMA ASR\ndamped'] = srow['arima_asr_damped_pop']
    vals['ARIMA ASR\nundamped'] = srow['arima_asr_undamped_pop']
    vals['Recent\ntrend'] = rt[(rt.location == 'Global') & (rt.measure == 'Incidence') &
                               (rt.year == 2050)].number.iloc[0]
    x = np.arange(len(vals))
    cols = ['#b0b0b0', BLUE, BLUE, SKY, ORANGE]
    ax.bar(x, [v / 1e6 for v in vals.values()], width=0.62, color=cols)
    for i, v in enumerate(vals.values()):
        ax.text(i, v / 1e6 + 0.06, f'{v / 1e6:.2f}', ha='center', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(list(vals.keys()), fontsize=5.8)
    ax.set_ylim(0, 6.0)
    ax.set_ylabel('Incident cases (millions per year)')
    panel(ax, 'B')

    # C -------------------------------------------------------------
    ax = axs[1, 0]
    obs = ag[(ag.risk == 'Metabolic risks') & (ag.measure == 'Deaths') &
             (ag.metric == 'Percent') & (ag.sex == 'both') &
             (ag.age == '15-49')].sort_values('year')
    pj = pa[(pa.location == 'Global') & (pa.measure == 'Deaths') &
            (pa.variable == 'paf_percent')].sort_values('year')
    ax.fill_between(obs.year, obs.lower * 100, obs.upper * 100, color=PINK,
                    alpha=0.15, lw=0)
    ax.plot(obs.year, obs.val * 100, color=PINK, lw=0.8)
    ax.fill_between(pj.year, pj.pi_low, pj.pi_high, color=PINK, alpha=0.15,
                    lw=0)
    ax.plot(pj.year, pj.projected, color=PINK, lw=0.8, ls=(0, (3, 1.6)))
    ax.axvline(2023, color=GREY, lw=0.6, ls=(0, (1.5, 1.5)))
    last = pj.iloc[-1]
    ax.annotate(f'{num(last.projected, 2)}% in 2050', (last.year, last.projected),
                xytext=(-6, 6), textcoords='offset points', fontsize=6,
                ha='right')
    ax.set_xlim(1990, 2053)
    ax.set_ylim(0, 8.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('PAF of early-onset cancer deaths (%)')
    panel(ax, 'C')

    # D -------------------------------------------------------------
    ax = axs[1, 1]
    obs23 = ag[(ag.risk == 'Metabolic risks') & (ag.measure == 'Deaths') &
               (ag.metric == 'Number') & (ag.sex == 'both') &
               (ag.age == '15-49') & (ag.year == 2023)].val.iloc[0]
    a = sc[(sc.location == 'Global') & (sc.outcome == 'Attributable deaths')]
    vals = {'Observed\n2023': obs23}
    names = [('constant_2023_rate_damped_pop', 'Constant\nrate'),
             ('risk_control_paf_frozen_2023', 'PAF frozen\nat 2023'),
             ('arima_damped_pop', 'ARIMA\ndamped'),
             ('arima_undamped_pop', 'ARIMA\nundamped')]
    for s_, lab in names:
        vals[lab] = a[a.scenario == s_].number_2050.iloc[0]
    vals['Recent\ntrend'] = rt[(rt.location == 'Global') & (rt.measure == 'Deaths') &
                               (rt.outcome_type == 'metabolic_attributable') &
                               (rt.year == 2050)].number.iloc[0]
    x = np.arange(len(vals))
    cols = ['#b0b0b0', BLUE, GREEN, BLUE, SKY, ORANGE]
    ax.bar(x, [v / 1e3 for v in vals.values()], width=0.62, color=cols)
    for i, v in enumerate(vals.values()):
        ax.text(i, v / 1e3 + 1.2, f'{v / 1e3:.1f}', ha='center', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(list(vals.keys()), fontsize=5.8, rotation=30, ha='right')
    ax.set_ylim(0, 92)
    ax.set_ylabel('Attributable deaths (thousands per year)')
    panel(ax, 'D')
    save(fig, 'figS3_projections.png')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
