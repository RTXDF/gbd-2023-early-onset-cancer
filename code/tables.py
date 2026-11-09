#!/usr/bin/env python3
"""Generate publication-ready tables (Markdown + XLSX) from results CSVs."""
import os, json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
TBL = os.path.join(RES, 'tables')
os.makedirs(TBL, exist_ok=True)

def rd(n):
    p = os.path.join(RES, n)
    if not os.path.exists(p) or os.path.getsize(p) == 0: return None
    try: return pd.read_csv(p)
    except pd.errors.EmptyDataError: return None

def fmt_n(x):
    if pd.isna(x): return '—'
    if abs(x) >= 1e6: return f'{x/1e6:.2f} million'
    if abs(x) >= 1e3: return f'{x/1e3:.1f} thousand'
    return f'{x:.0f}'

def fmt_ui(v, lo, hi, pct=False, scale=1):
    if pd.isna(v): return '—'
    m = 100 if pct else 1
    def f(x):
        x = x * scale * m
        return f'{x:,.0f}' if abs(x) >= 1000 else f'{x:.2f}'
    return f'{f(v)} ({f(lo)}–{f(hi)})'

def md(df, name, title):
    lines = [f'### {title}', '', df.to_markdown(index=False), '']
    open(os.path.join(TBL, name + '.md'), 'w').write('\n'.join(lines))
    return df

tables = {}

# ---- Table 1: global burden 1990 vs 2023 by sex ----
p = os.path.join(ROOT, 'data', 'panels', 'A_global_trends.pkl')
if os.path.exists(p):
    a = pd.read_pickle(p)
    a = a[(a.location == 1) & (a.cause == 1059) & (a.age == 24)]
    rows = []
    for sex, sl in [(3, 'Both'), (1, 'Male'), (2, 'Female')]:
        for measure, ml in [(6, 'Incidence'), (1, 'Deaths'), (2, 'DALYs')]:
            r90 = a[(a.year == 1990) & (a.sex == sex) & (a.measure == measure) & (a.metric == 1)]
            r23 = a[(a.year == 2023) & (a.sex == sex) & (a.measure == measure) & (a.metric == 1)]
            c90 = a[(a.year == 1990) & (a.sex == sex) & (a.measure == measure) & (a.metric == 3)]
            c23 = a[(a.year == 2023) & (a.sex == sex) & (a.measure == measure) & (a.metric == 3)]
            if len(r23) == 0: continue
            n90, n23 = r90.val.iloc[0], r23.val.iloc[0]
            rows.append({'Sex': sl, 'Outcome': ml,
                         '1990 cases/deaths/DALYs': fmt_ui(n90, r90.lower.iloc[0], r90.upper.iloc[0]),
                         '2023 cases/deaths/DALYs': fmt_ui(n23, r23.lower.iloc[0], r23.upper.iloc[0]),
                         'Change 1990–2023 (%)': f'{(n23/n90-1)*100:+.1f}',
                         '1990 rate per 100 000': f'{c90.val.iloc[0]:.1f}' if len(c90) else '—',
                         '2023 rate per 100 000': f'{c23.val.iloc[0]:.1f}' if len(c23) else '—'})
    if rows: tables['table1_global_burden'] = md(pd.DataFrame(rows), 'table1_global_burden', 'Table 1. Global early-onset cancer burden, 1990 and 2023')

# ---- Table 2: EAPC + AAPC ----
e = rd('trends_global_eapc.csv'); j = rd('trends_global_joinpoint.csv')
if e is not None:
    e = e[(e.age == '15-49') & (e.sex == 'Both')]
    rows = []
    for r in e.itertuples():
        jp = j[(j.age == '15-49') & (j.sex == 'Both') & (j.measure == r.measure)] if j is not None else None
        aapc = '—'; jpy = '—'
        if jp is not None and len(jp) and pd.notna(jp.aapc.iloc[0]):
            aapc = f'{jp.aapc.iloc[0]:.2f} ({jp.aapc_ci_low.iloc[0]:.2f}–{jp.aapc_ci_high.iloc[0]:.2f})'
            jpy = str(jp.joinpoint_years.iloc[0])
        rows.append({'Outcome': r.measure, 'Period': r.period,
                     'EAPC (95% CI)': f'{r.eapc:.2f} ({r.ci_low:.2f}–{r.ci_high:.2f})',
                     'AAPC (95% CI)': aapc, 'Joinpoints': jpy})
    tables['table2_trends'] = md(pd.DataFrame(rows), 'table2_trends', 'Table 2. Estimated annual percentage change, global early-onset cancer')

# ---- Table 3: metabolic attribution 2023 + EAPC ----
d = rd('attributable_global.csv'); t = rd('attributable_trends_eapc.csv')
if d is not None:
    d = d[(d.year == 2023) & (d.sex == 'both') & (d.age == '15-49')]
    rows = []
    for risk in ['Metabolic risks', 'High fasting plasma glucose', 'High body-mass index']:
        dd = d[d.risk == risk]
        if not len(dd): continue
        for measure in ['Deaths', 'DALYs']:
            mm = dd[dd.measure == measure]
            num = mm[mm.metric == 'Number']; paf = mm[mm.metric == 'Percent']; rate = mm[mm.metric == 'Rate']
            if not len(num): continue
            ea = t[(t.risk == risk) & (t.measure == measure) & (t.metric == 'Number')] if t is not None else None
            eapc_s = f'{ea.eapc.iloc[0]:.2f} ({ea.ci_low.iloc[0]:.2f}–{ea.ci_high.iloc[0]:.2f})' if ea is not None and len(ea) else '—'
            rows.append({'Risk': risk, 'Outcome': measure,
                         'Attributable number (95% UI)': fmt_ui(num.val.iloc[0], num.lower.iloc[0], num.upper.iloc[0]),
                         'PAF % (95% UI)': fmt_ui(paf.val.iloc[0], paf.lower.iloc[0], paf.upper.iloc[0], pct=True) if len(paf) else '—',
                         'Attributable rate per 100 000 (95% UI)': fmt_ui(rate.val.iloc[0], rate.lower.iloc[0], rate.upper.iloc[0]) if len(rate) else '—',
                         'EAPC of number 1990–2023 (95% CI)': eapc_s})
    if rows: tables['table3_attribution'] = md(pd.DataFrame(rows), 'table3_attribution', 'Table 3. Metabolic-risk-attributable early-onset cancer burden, 2023')

# ---- Table 4: top sites 2023 ----
s = rd('sites_global_2023.csv'); se = rd('sites_eapc.csv')
if s is not None:
    rows = []
    inc = s[(s.measure == 'Incidence') & (s.metric == 'Number')].sort_values('val', ascending=False).head(10)
    for r in inc.itertuples():
        de = s[(s.cause == r.cause) & (s.measure == 'Deaths') & (s.metric == 'Number')]
        da = s[(s.cause == r.cause) & (s.measure == 'DALYs') & (s.metric == 'Number')]
        ea = se[(se.cause == r.cause) & (se.measure == 'Incidence')] if se is not None else None
        rows.append({'Cancer site': r.cause,
                     'Incident cases (95% UI)': fmt_ui(r.val, r.lower, r.upper),
                     'Deaths (95% UI)': fmt_ui(de.val.iloc[0], de.lower.iloc[0], de.upper.iloc[0]) if len(de) else '—',
                     'DALYs (95% UI)': fmt_ui(da.val.iloc[0], da.lower.iloc[0], da.upper.iloc[0]) if len(da) else '—',
                     'Incidence EAPC (95% CI)': f'{ea.eapc.iloc[0]:.2f} ({ea.ci_low.iloc[0]:.2f}–{ea.ci_high.iloc[0]:.2f})' if ea is not None and len(ea) else '—'})
    tables['table4_sites'] = md(pd.DataFrame(rows), 'table4_sites', 'Table 4. Ten leading early-onset cancer sites, 2023')

# ---- Table 5: inequality ----
d = rd('inequality_indices.csv')
if d is not None and len(d):
    rows = []
    for r in d.itertuples():
        def g(name): return getattr(r, name, float('nan'))
        rows.append({'Outcome': r.outcome.replace(' (15-49 rate)', '').replace('Total ', ''),
                     'Sex': r.sex.capitalize(),
                     'Concentration index': f'{r.concentration_index:.3f}',
                     'SII (95% CI)': f'{r.sii:.1f} ({g("sii_ci_low"):.1f}–{g("sii_ci_high"):.1f})',
                     'RII (95% CI)': f'{r.rii:.2f} ({g("rii_ci_low"):.2f}–{g("rii_ci_high"):.2f})'})
    tables['table5_inequality'] = md(pd.DataFrame(rows), 'table5_inequality', 'Table 5. Socioeconomic inequality in early-onset cancer burden, 2023')

# ---- Table 6: decomposition ----
d = rd('decomposition_global.csv')
if d is not None and len(d):
    dd = d[d.period == '1990-2023'].copy()
    rows = []
    for r in dd.itertuples():
        rows.append({'Outcome': r.outcome, '1990 value': fmt_n(r.value_0), '2023 value': fmt_n(r.value_1),
                     'Population effect': fmt_n(r.population_effect), 'Rate effect': fmt_n(r.rate_effect),
                     'PAF effect': fmt_n(r.paf_effect) if pd.notna(r.paf_effect) else '—'})
    tables['table6_decomposition'] = md(pd.DataFrame(rows), 'table6_decomposition', 'Table 6. Das Gupta decomposition of change in global early-onset cancer deaths, 1990–2023')

# ---- Table 7: projections ----
d = rd('projections_rate.csv'); n = rd('projections_number.csv')
if d is not None:
    dd = d[(d.location == 'Global') & (d.year.isin([2030, 2040, 2050]))]
    rows = []
    for r in dd.sort_values(['measure', 'year']).itertuples():
        nn = n[(n.location == 'Global') & (n.measure == r.measure) & (n.year == r.year)] if n is not None else None
        rows.append({'Outcome': r.measure, 'Year': r.year,
                     'Projected rate per 100 000 (95% PI)': f'{r.projected_rate:.1f} ({r.pi_low:.1f}–{r.pi_high:.1f})',
                     'Projected number (95% PI)': fmt_ui(nn.projected_number.iloc[0], nn.pi_low.iloc[0], nn.pi_high.iloc[0]) if nn is not None and len(nn) else '—',
                     'Model': r.model})
    if rows: tables['table7_projections'] = md(pd.DataFrame(rows), 'table7_projections', 'Table 7. Projected global early-onset cancer burden to 2050')

# ---- XLSX workbook with all tables + supplementary country data ----
try:
    with pd.ExcelWriter(os.path.join(ROOT, 'manuscript', 'tables.xlsx'), engine='openpyxl') as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
        for supp in ['trends_country_eapc.csv', 'attributable_country_2023.csv', 'inequality_country_data.csv', 'apc_ie_effects.csv', 'sites_eapc.csv']:
            df = rd(supp)
            if df is not None and len(df):
                df.to_excel(xw, sheet_name=('S_' + supp.replace('.csv',''))[:31], index=False)
    print('tables.xlsx written')
except Exception as ex:
    print('xlsx skipped:', ex)
print(f'{len(tables)} tables generated -> {TBL}')