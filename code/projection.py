#!/usr/bin/env python3
"""Projections of (metabolic-attributable) early-onset cancer burden to 2050.

Rate projection (population-independent, primary): ARIMA on log rate, AIC-selected among
  (0,1,0)+drift, (1,1,0), (0,2,0)+drift, (2,1,0); plus constant-rate scenario. 95% PI.
Number projection (scenario): projected crude 15-49 rate x projected 15-49 population.
  Population scenario = damped log-linear extrapolation of GBD 2008-2023 trend (damp=0.9);
  undamped (damp=1.0) sensitivity. Clearly labelled as scenario (no external WPP available).
Attributable projection (novelty): project metabolic PAF & attributable rate; number = rate x pop.
Targets: Global + 5 SDI quintiles; measures Incidence/Deaths/DALYs; both sexes.
Outputs (results/): projections_rate.csv, projections_number.csv, projections_attributable.csv
"""
import os, json, math, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results'); os.makedirs(RES, exist_ok=True)
idmap = json.load(open(os.path.join(ROOT, 'data', 'id_map.json')))
FUTURE = list(range(2024, 2051))
MEASURES = {1: 'Deaths', 2: 'DALYs', 6: 'Incidence'}
TARGETS = [('Global', 1)] + [(v, int(k)) for k, v in idmap['sdi_quintiles'].items()]
CAND = [((0,1,0), True), ((1,1,0), False), ((0,2,0), True), ((2,1,0), False)]

def fit_project(series_years, series_vals, horizon=27):
    y = np.asarray(series_vals, float); yrs = np.asarray(series_years, int)
    m = np.isfinite(y) & (y > 0); y, yrs = y[m], yrs[m]
    if len(y) < 10: return None
    ly = np.log(y)
    best = None
    for order, drift in CAND:
        try:
            trend = 't' if drift else 'n'
            mod = ARIMA(ly, order=order, trend=trend).fit()
            if best is None or mod.aic < best[0]:
                best = (mod.aic, order, drift, mod)
        except Exception:
            continue
    if best is None: return None
    aic, order, drift, mod = best
    fc = mod.get_forecast(steps=horizon)
    mean = np.exp(np.asarray(fc.predicted_mean))
    ci = np.exp(np.asarray(fc.conf_int(alpha=0.05)))
    # constant-rate scenario
    const = np.full(horizon, y[-1])
    return {'order': str(order), 'drift': drift, 'aic': aic, 'mean': mean,
            'low': ci[:, 0], 'high': ci[:, 1], 'const': const,
            'last_year': int(yrs[-1]), 'last_val': float(y[-1])}

def project_population(loc, damp=0.9):
    p = os.path.join(PANELS, 'L_pop_sex3.pkl')
    if not os.path.exists(p): return None
    d = pd.read_pickle(p)
    d = d[(d.location == loc) & (d.age == 24) & (d.measure == 44) & (d.metric == 1) & (d.sex == 3)]
    d = d[(d.year >= 2008) & (d.year <= 2023)].sort_values('year')
    if len(d) < 8: return None
    ly = np.log(d.val.values); t = d.year.values - d.year.values.mean()
    slope = np.polyfit(t, ly, 1)[0]  # annual log growth
    last = d.val.values[-1]; base_year = int(d.year.values[-1])
    out = {}
    for yr in FUTURE:
        h = yr - base_year
        cum = sum(damp ** s for s in range(1, h + 1))
        out[yr] = last * math.exp(slope * cum)
    return out

def get_series(panel, loc, measure, metric, age):
    p = os.path.join(PANELS, panel + '.pkl')
    if not os.path.exists(p): return None
    d = pd.read_pickle(p)
    d = d[(d.location == loc) & (d.measure == measure) & (d.metric == metric) & (d.age == age) & (d.sex == 3)].sort_values('year')
    if len(d) < 10: return None
    return d.year.values, d.val.values

def run():
    rate_rows, num_rows, attr_rows = [], [], []
    for locname, loc in TARGETS:
        pop_proj = project_population(loc, 0.9)
        pop_und = project_population(loc, 1.0)
        for measure, mlabel in MEASURES.items():
            # ---- rate projection: ASR (age27) and 15-49 crude rate (age24) ----
            for age, agelab in [(24, '15-49')]:
                s = get_series('A_global_trends' if loc == 1 else 'BCD_country_eo_both', loc, measure, 3, age)
                if s is None and loc != 1:
                    s = get_series('BCD_country_eo_both', loc, measure, 3, age)
                if s is None: continue
                yrs, vals = s
                fp = fit_project(yrs, vals)
                if fp is None: continue
                for i, yr in enumerate(FUTURE):
                    rate_rows.append({'location': locname, 'measure': mlabel, 'age': agelab, 'year': yr,
                                      'model': f"ARIMA{fp['order']}{'drift' if fp['drift'] else ''}",
                                      'projected_rate': fp['mean'][i], 'pi_low': fp['low'][i], 'pi_high': fp['high'][i],
                                      'constant_rate_scenario': fp['const'][i]})
                # number projection using crude 15-49 rate x population (only for age24)
                if age == 24 and pop_proj:
                    for i, yr in enumerate(FUTURE):
                        num = fp['mean'][i] / 1e5 * pop_proj[yr]
                        num_lo = fp['low'][i] / 1e5 * pop_proj[yr]
                        num_hi = fp['high'][i] / 1e5 * pop_proj[yr]
                        num_und = fp['mean'][i] / 1e5 * pop_und[yr] if pop_und else np.nan
                        num_rows.append({'location': locname, 'measure': mlabel, 'year': yr,
                                         'projected_number': num, 'pi_low': num_lo, 'pi_high': num_hi,
                                         'undamped_pop_scenario': num_und, 'pop_scenario': pop_proj[yr],
                                         'rate_model': f"ARIMA{fp['order']}"})
            # ---- attributable projection (metabolic risk104) ----
            if loc == 1:
                rp = 'GHI_risk104_country_both'
            else:
                rp = 'GHI_risk104_country_both'
            for metric, mlab in [(3, 'attributable_rate'), (2, 'paf_percent')]:
                s = get_series(rp, loc, measure, metric, 24)
                if s is None: continue
                yrs, vals = s
                if metric == 2: vals = vals * 100  # API PAF is a fraction -> percent
                fp = fit_project(yrs, vals)
                if fp is None: continue
                for i, yr in enumerate(FUTURE):
                    attr_rows.append({'location': locname, 'measure': mlabel, 'variable': mlab, 'year': yr,
                                      'projected': fp['mean'][i], 'pi_low': fp['low'][i], 'pi_high': fp['high'][i],
                                      'model': f"ARIMA{fp['order']}"})
    pd.DataFrame(rate_rows).to_csv(os.path.join(RES, 'projections_rate.csv'), index=False)
    pd.DataFrame(num_rows).to_csv(os.path.join(RES, 'projections_number.csv'), index=False)
    pd.DataFrame(attr_rows).to_csv(os.path.join(RES, 'projections_attributable.csv'), index=False)
    print(f"projections: rate={len(rate_rows)}, number={len(num_rows)}, attributable={len(attr_rows)}")

if __name__ == '__main__':
    run()