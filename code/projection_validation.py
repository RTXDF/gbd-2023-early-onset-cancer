#!/usr/bin/env python3
"""Validation and scenario comparison for the 2024-2050 projections.

(1) Backtest (Global, 15-49, both sexes): the ARIMA candidate set used by
    scripts/projection.py is fitted on 1990-2012 log rates and forecast to
    2013-2023 (11 steps). Metrics: MAE, MAPE, mean bias, 95% PI coverage.
    Benchmarks: constant 2012 rate and log-linear trend extrapolation (with a
    proper OLS prediction interval on the log scale).
(2) Diagnostics for the full-period models actually used in
    results/projections_rate.csv: Ljung-Box p (lags 5, 10), residual SD, AIC,
    selected order, and a 2050 cross-check against the published projection.
(3) Scenario comparison to 2050 for Global + 5 SDI quintiles:
      (a) constant 2023 rate + damped population
      (b) ARIMA rate + damped population (damping 0.9)
      (c) ARIMA rate + undamped population
      (d) risk-control for metabolic-attributable deaths/DALYs: freeze the PAF at
          2023 and project the baseline (cause-1059) rate with ARIMA.
(4) IHME GBD 2023 cancer forecasts (data/aux/cancer_forecasts_2024_2050.csv):
    Global total-cancer forecasts are all-age (or <20 years) and are therefore
    NOT directly comparable with the 15-49 estimates; they are reported side by
    side with an explicit comparability flag, never forced onto the 15-49 scale.
Outputs (results/):
  projection_backtest.csv, projection_diagnostics.csv,
  projections_scenarios.csv, projections_scenarios_2050.csv,
  ihme_comparison.csv, ihme_comparison_2050.csv
"""
import os
import re
import sys
import json
import math
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from projection import fit_project, project_population, CAND  # noqa: E402

ROOT = os.path.dirname(HERE)
PANELS = os.path.join(ROOT, 'data', 'panels')
RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)

IDMAP = json.load(open(os.path.join(ROOT, 'data', 'id_map.json')))
MEASURES = {6: 'Incidence', 1: 'Deaths', 2: 'DALYs'}
ATTR_MEASURES = {1: 'Deaths', 2: 'DALYs'}
FUTURE = list(range(2024, 2051))
TARGETS = [('Global', 1)] + [(v, int(k)) for k, v in IDMAP['sdi_quintiles'].items()]
TARGETS = [(name, loc) for name, loc in TARGETS]
Z = 1.959963985


def load(name):
    return pd.read_pickle(os.path.join(PANELS, name + '.pkl'))


def series(panel, loc, measure, metric, age=24):
    d = panel[(panel.location == loc) & (panel.measure == measure) &
              (panel.metric == metric) & (panel.age == age) & (panel.sex == 3)].sort_values('year')
    return d.year.values.astype(int), d.val.values.astype(float)


# --------------------------------------------------------------------------
# 1. backtest
# --------------------------------------------------------------------------
def loglinear_forecast(years, vals, horizon):
    x = years.astype(float)
    y = np.log(vals)
    xm = x.mean()
    X = sm.add_constant(x - xm)
    fit = sm.OLS(y, X).fit()
    s2 = float(np.sum(fit.resid ** 2) / max(len(y) - 2, 1))
    sxx = float(np.sum((x - xm) ** 2))
    future = np.arange(years[-1] + 1, years[-1] + horizon + 1, dtype=float)
    Xf = sm.add_constant(future - xm)
    mean = fit.predict(Xf)
    se = np.sqrt(s2 * (1 + 1 / len(y) + (future - xm) ** 2 / sxx))
    return np.exp(mean), np.exp(mean - Z * se), np.exp(mean + Z * se)


def metrics(actual, pred, low, high):
    err = pred - actual
    return {
        'n_test': int(len(actual)),
        'mae': float(np.mean(np.abs(err))),
        'mape': float(np.mean(np.abs(err / actual)) * 100),
        'mean_bias': float(np.mean(err)),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'coverage_95': float(np.mean((actual >= low) & (actual <= high))),
    }


def population_projection(loc, damp, bcd):
    """Damped log-linear population projection.

    Uses projection.project_population (L_pop_sex3) where available. L_pop_sex3
    covers the 204 countries + Global only, so SDI-quintile populations are not in
    that panel; for those locations the GBD-implied 15-49 population
    (incidence number / crude incidence rate) is used with the same estimator.
    """
    p = project_population(loc, damp)
    if p is not None:
        return p, 'L_pop_sex3'
    num = bcd[(bcd.location == loc) & (bcd.measure == 6) & (bcd.metric == 1) &
              (bcd.age == 24) & (bcd.sex == 3) & (bcd.year >= 2008) &
              (bcd.year <= 2023)].sort_values('year')
    rate = bcd[(bcd.location == loc) & (bcd.measure == 6) & (bcd.metric == 3) &
               (bcd.age == 24) & (bcd.sex == 3) & (bcd.year >= 2008) &
               (bcd.year <= 2023)].sort_values('year')
    if len(num) < 8 or len(rate) != len(num):
        return None, 'unavailable'
    pop = num.val.values / rate.val.values * 1e5
    years = num.year.values
    ly = np.log(pop)
    t = years - years.mean()
    slope = float(np.polyfit(t, ly, 1)[0])
    last, base_year = float(pop[-1]), int(years[-1])
    out = {}
    for yr in FUTURE:
        h = yr - base_year
        cum = sum(damp ** s for s in range(1, h + 1))
        out[yr] = last * math.exp(slope * cum)
    return out, 'gbd_implied_count_over_rate'


def run_backtest(bcd):
    rows = []
    for measure, mlab in MEASURES.items():
        yrs, vals = series(bcd, 1, measure, 3)
        train = yrs <= 2012
        test = (yrs >= 2013) & (yrs <= 2023)
        ytr, vtr = yrs[train], vals[train]
        actual = vals[test]
        tyr = yrs[test]
        # ARIMA candidate set (same as projection.py)
        fp = fit_project(ytr, vtr, horizon=len(actual))
        if fp is not None:
            rows.append({'location': 'Global', 'measure': mlab, 'model': 'arima_aic_selected',
                         'model_detail': f"ARIMA{fp['order']}{'drift' if fp['drift'] else ''}",
                         'train_start': int(ytr.min()), 'train_end': int(ytr.max()),
                         'test_start': int(tyr.min()), 'test_end': int(tyr.max()),
                         **metrics(actual, fp['mean'], fp['low'], fp['high'])})
        # constant 2012 rate
        const = np.full(len(actual), vtr[-1])
        rows.append({'location': 'Global', 'measure': mlab, 'model': 'constant_2012_rate',
                     'model_detail': 'last observed train value repeated',
                     'train_start': int(ytr.min()), 'train_end': int(ytr.max()),
                     'test_start': int(tyr.min()), 'test_end': int(tyr.max()),
                     **metrics(actual, const, const, const)})
        # log-linear trend
        mean, low, high = loglinear_forecast(ytr, vtr, len(actual))
        rows.append({'location': 'Global', 'measure': mlab, 'model': 'loglinear_trend',
                     'model_detail': 'OLS on log rate + 95% prediction interval',
                     'train_start': int(ytr.min()), 'train_end': int(ytr.max()),
                     'test_start': int(tyr.min()), 'test_end': int(tyr.max()),
                     **metrics(actual, mean, low, high)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'projection_backtest.csv'), index=False)
    print(f'projection_backtest.csv: {len(out)} rows')
    print(out[['measure', 'model', 'mae', 'mape', 'mean_bias', 'coverage_95']].round(3).to_string(index=False))
    return out


# --------------------------------------------------------------------------
# 2. diagnostics of the published full-period models
# --------------------------------------------------------------------------
def parse_model(model):
    m = re.search(r'ARIMA\((\d+),\s*(\d+),\s*(\d+)\)', model)
    order = tuple(int(x) for x in m.groups()) if m else None
    return order, model.endswith('drift')


def run_diagnostics(bcd):
    proj = pd.read_csv(os.path.join(RES, 'projections_rate.csv'))
    diag = []
    for (locname, mlab), g in proj.groupby(['location', 'measure']):
        order, drift = parse_model(g.model.iloc[0])
        if order is None:
            continue
        loc = dict(zip([n for n, _ in TARGETS], [i for _, i in TARGETS]))[locname]
        measure = {v: k for k, v in MEASURES.items()}[mlab]
        yrs, vals = series(bcd, loc, measure, 3)
        ly = np.log(vals)
        try:
            fit = ARIMA(ly, order=order, trend='t' if drift else 'n').fit()
        except Exception as e:  # pragma: no cover
            print(f'diagnostics fit failed for {locname} {mlab}: {e}')
            continue
        # discard diffuse-initialisation burn-in (d + AR order) before diagnostics
        resid = np.asarray(fit.resid)
        burn = min(2, max(len(resid) - 12, 0))
        resid_use = resid[burn:]
        lb = acorr_ljungbox(resid_use, lags=[5, 10], return_df=True)
        diag.append({
            'location': locname, 'measure': mlab, 'model': g.model.iloc[0],
            'order': str(order), 'drift': bool(drift),
            'n_obs': int(len(ly)), 'n_used_diagnostics': int(len(resid_use)),
            'burn_in_discarded': int(burn), 'aic': float(fit.aic),
            'residual_sd_log': float(np.std(resid_use, ddof=1)),
            'residual_sd_log_incl_burnin': float(np.std(resid, ddof=1)),
            'ljung_box_p_lag5': float(lb.loc[5, 'lb_pvalue']),
            'ljung_box_p_lag10': float(lb.loc[10, 'lb_pvalue']),
            'forecast_2050_rate': float(np.exp(fit.get_forecast(steps=27).predicted_mean[-1])),
            'published_2050_rate': float(g[g.year == 2050].projected_rate.iloc[0]),
        })
    out = pd.DataFrame(diag)
    out['forecast_2050_matches_published'] = np.isclose(
        out.forecast_2050_rate, out.published_2050_rate, rtol=1e-6)
    out.to_csv(os.path.join(RES, 'projection_diagnostics.csv'), index=False)
    print(f'projection_diagnostics.csv: {len(out)} rows')
    print(out[['location', 'measure', 'model', 'aic', 'ljung_box_p_lag5',
               'ljung_box_p_lag10', 'forecast_2050_matches_published']].round(4).to_string(index=False))
    return out


# --------------------------------------------------------------------------
# 3. scenarios to 2050
# --------------------------------------------------------------------------
def run_scenarios(bcd, risk):
    rows = []
    for name, loc in TARGETS:
        pop_d, pop_src = population_projection(loc, 0.9, bcd)
        pop_u, _ = population_projection(loc, 1.0, bcd)
        if pop_d is None:
            print(f'no population projection for {name}')
            continue
        pop_basis = {name: pop_src}
        # ---- total burden ----
        for measure, mlab in MEASURES.items():
            yrs, vals = series(bcd, loc, measure, 3)
            fp = fit_project(yrs, vals, horizon=len(FUTURE))
            if fp is None:
                continue
            const = float(vals[-1])
            obs_num = float(bcd[(bcd.location == loc) & (bcd.measure == measure) &
                                (bcd.metric == 1) & (bcd.age == 24) & (bcd.sex == 3) &
                                (bcd.year == 2023)].val.iloc[0])
            for i, yr in enumerate(FUTURE):
                for scen, rate, pop in [
                    ('constant_2023_rate_damped_pop', const, pop_d[yr]),
                    ('arima_damped_pop', float(fp['mean'][i]), pop_d[yr]),
                    ('arima_undamped_pop', float(fp['mean'][i]), pop_u[yr]),
                ]:
                    rows.append({'location': name, 'outcome': mlab, 'measure_type': 'total',
                                 'scenario': scen, 'year': yr, 'rate': rate,
                                 'number': rate / 1e5 * pop, 'paf_percent': np.nan,
                                 'population': pop, 'population_basis': pop_src})
            # observed baseline row for change calculations
            rows.append({'location': name, 'outcome': mlab, 'measure_type': 'total',
                         'scenario': 'observed_2023', 'year': 2023,
                         'rate': const, 'number': obs_num, 'paf_percent': np.nan,
                         'population': float(bcd[(bcd.location == loc) & (bcd.measure == measure) &
                                                 (bcd.metric == 1) & (bcd.age == 24) &
                                                 (bcd.sex == 3) & (bcd.year == 2023)].val.iloc[0]) /
                         const * 1e5})
        # ---- attributable burden ----
        for measure, mlab in ATTR_MEASURES.items():
            yrs, vals = series(risk, loc, measure, 3)          # attributable rate
            yrs_p, vals_p = series(risk, loc, measure, 2)      # PAF fraction
            fa = fit_project(yrs, vals, horizon=len(FUTURE))
            fpaf = fit_project(yrs_p, vals_p * 100, horizon=len(FUTURE))
            if fa is None:
                continue
            const_rate = float(vals[-1])
            paf_2023 = float(vals_p[-1] * 100)
            attr_obs = float(risk[(risk.location == loc) & (risk.measure == measure) &
                                  (risk.metric == 1) & (risk.age == 24) & (risk.sex == 3) &
                                  (risk.year == 2023)].val.iloc[0])
            # baseline total rate for the risk-control scenario
            byrs, bvals = series(bcd, loc, measure, 3)
            fbase = fit_project(byrs, bvals, horizon=len(FUTURE))
            outcome = f'Attributable {mlab.lower()}'
            for i, yr in enumerate(FUTURE):
                for scen, rate, pop, paf in [
                    ('constant_2023_rate_damped_pop', const_rate, pop_d[yr], paf_2023),
                    ('arima_damped_pop', float(fa['mean'][i]), pop_d[yr],
                     float(fpaf['mean'][i]) if fpaf is not None else np.nan),
                    ('arima_undamped_pop', float(fa['mean'][i]), pop_u[yr],
                     float(fpaf['mean'][i]) if fpaf is not None else np.nan),
                ]:
                    rows.append({'location': name, 'outcome': outcome,
                                 'measure_type': 'metabolic_attributable', 'scenario': scen,
                                 'year': yr, 'rate': rate, 'number': rate / 1e5 * pop,
                                 'paf_percent': paf, 'population': pop, 'population_basis': pop_src})
                # (d) risk control: freeze PAF at 2023, project baseline rate
                if fbase is not None:
                    base_rate = float(fbase['mean'][i])
                    attr_rate = base_rate * paf_2023 / 100
                    rows.append({'location': name, 'outcome': outcome,
                                 'measure_type': 'metabolic_attributable',
                                 'scenario': 'risk_control_paf_frozen_2023', 'year': yr,
                                 'rate': attr_rate, 'number': attr_rate / 1e5 * pop_d[yr],
                                 'paf_percent': paf_2023, 'population': pop_d[yr], 'population_basis': pop_src})
            rows.append({'location': name, 'outcome': outcome,
                         'measure_type': 'metabolic_attributable', 'scenario': 'observed_2023',
                         'year': 2023, 'rate': const_rate, 'number': attr_obs,
                         'paf_percent': paf_2023,
                         'population': attr_obs / const_rate * 1e5 if const_rate else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'projections_scenarios.csv'), index=False)
    print(f'projections_scenarios.csv: {len(out)} rows')

    # ---- 2050 summary ----
    obs = out[out.year == 2023].set_index(['location', 'outcome', 'scenario'])
    summ = []
    for (loc, outcome, scen), g in out[(out.year == 2050)].groupby(['location', 'outcome', 'scenario']):
        r = g.iloc[0]
        key = (loc, outcome, 'observed_2023')
        o_rate = float(obs.loc[key, 'rate']) if key in obs.index else np.nan
        o_num = float(obs.loc[key, 'number']) if key in obs.index else np.nan
        summ.append({'location': loc, 'outcome': outcome, 'measure_type': r.measure_type,
                     'scenario': scen, 'year': 2050, 'rate_2050': r.rate,
                     'number_2050': r.number, 'paf_2050': r.paf_percent,
                     'observed_2023_rate': o_rate, 'observed_2023_number': o_num,
                     'rate_change_vs_2023_percent': (r.rate / o_rate - 1) * 100 if o_rate else np.nan,
                     'number_change_vs_2023_percent': (r.number / o_num - 1) * 100 if o_num else np.nan})
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(RES, 'projections_scenarios_2050.csv'), index=False)
    print(f'projections_scenarios_2050.csv: {len(s)} rows')
    print(s[s.location == 'Global'].round(2).to_string(index=False))
    return out, s


# --------------------------------------------------------------------------
# 4. IHME forecast comparison
# --------------------------------------------------------------------------
def run_ihme():
    f = pd.read_csv(os.path.join(ROOT, 'data', 'aux', 'cancer_forecasts_2024_2050.csv'))
    g = f[(f.location_id == 1) & (f.cause_id == 1059) & (f.sex_id == 3)].copy()
    mapping = {
        'All ages': ('All ages', 'Number'),
        'Age-standardized': ('All ages', 'Rate'),
        '<20 years': ('<20 years', 'Number'),
        '<20 years Age-standardized': ('<20 years', 'Rate'),
    }
    g = g[g.age_group_name.isin(mapping)]
    g['age'] = g.age_group_name.map(lambda x: mapping[x][0])
    g['metric'] = g.age_group_name.map(lambda x: mapping[x][1])
    note = {
        ('All ages', 'Number'): 'IHME all-age total-cancer count; not 15-49-specific.',
        ('All ages', 'Rate'): ('IHME age-standardized rate over all ages; not a 15-49 '
                               'crude rate, do not compare with the 15-49 projections.'),
        ('<20 years', 'Number'): 'IHME <20-year count; not 15-49-specific.',
        ('<20 years', 'Rate'): ('IHME <20-year age-standardized rate; not a 15-49 crude rate.'),
    }
    g['comparability_note'] = [note[(a, m)] for a, m in zip(g.age, g.metric)]
    out = g[['year_id', 'measure_name', 'metric', 'age', 'val', 'lower', 'upper',
             'age_group_name', 'comparability_note']].rename(
        columns={'year_id': 'year', 'measure_name': 'measure', 'age_group_name': 'source_age_group'})
    # the IHME Rate metric is stored as a proportion (per person); add per-100,000
    is_rate = out.metric == 'Rate'
    out['val_per_100000'] = np.where(is_rate, out.val * 1e5, np.nan)
    out['lower_per_100000'] = np.where(is_rate, out.lower * 1e5, np.nan)
    out['upper_per_100000'] = np.where(is_rate, out.upper * 1e5, np.nan)
    out = out.sort_values(['measure', 'metric', 'age', 'year']).reset_index(drop=True)
    out.to_csv(os.path.join(RES, 'ihme_comparison.csv'), index=False)
    print(f'ihme_comparison.csv: {len(out)} rows')

    rows = []
    for (measure, metric, age), gg in out.groupby(['measure', 'metric', 'age']):
        gg = gg.sort_values('year')
        a = gg[gg.year == 2024]
        b = gg[gg.year == 2050]
        if not len(a) or not len(b):
            continue
        rows.append({'measure': measure, 'metric': metric, 'age': age,
                     'val_2024': float(a.val.iloc[0]),
                     'val_2050': float(b.val.iloc[0]),
                     'val_2050_low': float(b.lower.iloc[0]),
                     'val_2050_high': float(b.upper.iloc[0]),
                     'val_2024_per_100000': (float(a.val.iloc[0]) * 1e5
                                             if metric == 'Rate' else np.nan),
                     'val_2050_per_100000': (float(b.val.iloc[0]) * 1e5
                                             if metric == 'Rate' else np.nan),
                     'percent_change_2024_2050': (b.val.iloc[0] / a.val.iloc[0] - 1) * 100,
                     'comparability_note': b.comparability_note.iloc[0]})
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(RES, 'ihme_comparison_2050.csv'), index=False)
    print(f'ihme_comparison_2050.csv: {len(s)} rows')
    print(s.round(2).to_string(index=False))
    return out, s


def main():
    bcd = load('BCD_country_eo_both')
    risk = load('GHI_risk104_country_both')
    run_backtest(bcd)
    run_diagnostics(bcd)
    run_scenarios(bcd, risk)
    run_ihme()
    print('projection validation complete')


if __name__ == '__main__':
    main()
