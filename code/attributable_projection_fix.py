"""Make the global metabolic-attributable projections internally coherent.

The legacy projection file projected the attributable rate with its own ARIMA
model, independent of the total-burden ARIMA and the PAF projection, so the
attributable numbers did not equal total burden x PAF.  This script rebuilds the
global attributable series as ASR_pathway x PAF and the 2050 attributable
scenarios as total number x PAF, and rewrites the recent-trend rows the same
way.  Originals are copied to results/superseded/ first.

Run with .venv/bin/python from the project root.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SUP = RES / "superseded"

ATTR = RES / "projections_attributable.csv"
SCEN = RES / "projections_scenarios_2050.csv"
RECENT = RES / "projection_recent_trend.csv"

for f in (ATTR, SCEN, RECENT):
    SUP.mkdir(exist_ok=True)
    backup = SUP / (f.name + ".pre_attributable_fix")
    if not backup.exists():
        shutil.copy2(f, backup)

attr = pd.read_csv(ATTR)
scen = pd.read_csv(SCEN)
recent = pd.read_csv(RECENT)
asr = pd.read_csv(RES / "projection_asr_series.csv")
asr_scen = pd.read_csv(RES / "projections_asr_scenarios_2050.csv")
obs = pd.read_csv(RES / "attributable_global.csv")

MEAS = ["Deaths", "DALYs"]

# observed 2023 PAF (percent), both sexes, ages 15-49
paf23 = {}
for m in MEAS:
    row = obs[(obs.risk == "Metabolic risks") & (obs.measure == m) &
              (obs.metric == "Percent") & (obs.sex == "both") &
              (obs.age == "15-49") & (obs.year == 2023)]
    v = float(row.val.iloc[0])
    paf23[m] = v * 100 if v < 1 else v  # stored as a fraction for metric Percent

# projected 2050 PAF (percent)
paf50 = {}
for m in MEAS:
    row = attr[(attr.location == "Global") & (attr.measure == m) &
               (attr.variable == "paf_percent") & (attr.year == 2050)]
    paf50[m] = float(row.projected.iloc[0])

print("PAF 2023 (%):", {k: round(v, 4) for k, v in paf23.items()})
print("PAF 2050 (%):", {k: round(v, 4) for k, v in paf50.items()})

# ---- 1. attributable-rate series = ASR pathway x PAF pathway --------------
for m in MEAS:
    a = attr[(attr.location == "Global") & (attr.measure == m) &
             (attr.variable == "attributable_rate")].index
    s = asr[asr.measure == m].set_index("year")
    p = attr[(attr.location == "Global") & (attr.measure == m) &
             (attr.variable == "paf_percent")].set_index("year")
    for i in a:
        yr = int(attr.at[i, "year"])
        rate = float(s.at[yr, "asr"]) * float(p.at[yr, "projected"]) / 100
        lo = float(s.at[yr, "asr_low"]) * float(p.at[yr, "projected"]) / 100
        hi = float(s.at[yr, "asr_high"]) * float(p.at[yr, "projected"]) / 100
        attr.at[i, "projected"] = rate
        attr.at[i, "pi_low"] = lo
        attr.at[i, "pi_high"] = hi
        attr.at[i, "model"] = "ASR x PAF (derived)"

# ---- 2. 2050 attributable scenarios = total number x PAF ------------------
# total numbers from the ASR refit (same source as Table 7 Panel A)
tot = {}
for m, lab in (("Deaths", "Attributable deaths"), ("DALYs", "Attributable dalys")):
    g = asr_scen[asr_scen.measure == m].set_index("scenario")
    tot[m] = {
        "damped": float(g.at["arima_asr_damped_pop", "number_2050"]),
        "undamped": float(g.at["arima_asr_undamped_pop", "number_2050"]),
        "constant": float(g.at["constant_2023_asr_damped_pop", "number_2050"]),
    }

mapping = {
    "arima_damped_pop": ("damped", paf50),
    "arima_undamped_pop": ("undamped", paf50),
    "constant_2023_rate_damped_pop": ("constant", None),   # PAF held at 2023
    "risk_control_paf_frozen_2023": ("damped", None),      # PAF held at 2023
}

for m, lab in (("Deaths", "Attributable deaths"), ("DALYs", "Attributable dalys")):
    idx = scen[(scen.location == "Global") & (scen.outcome == lab)].index
    for i in idx:
        sc = scen.at[i, "scenario"]
        base, pafd = mapping[sc]
        paf = paf23[m] if pafd is None else paf50[m]
        num = tot[m][base] * paf / 100
        scen.at[i, "number_2050"] = num
        # rate per 100,000 = ASR pathway x PAF/100
        asr_base = {"Deaths": 33.584522721227835, "DALYs": 1782.4752030126301}[m]
        if sc == "constant_2023_rate_damped_pop":
            asr_base = {"Deaths": 29.86614157428461, "DALYs": 1548.664990498622}[m]
        scen.at[i, "rate_2050"] = asr_base * paf / 100
        scen.at[i, "paf_2050"] = paf
        o_num = float(scen.at[i, "observed_2023_number"])
        o_rate = float(scen.at[i, "observed_2023_rate"])
        scen.at[i, "number_change_vs_2023_percent"] = (num / o_num - 1) * 100
        scen.at[i, "rate_change_vs_2023_percent"] = (
            scen.at[i, "rate_2050"] / o_rate - 1) * 100

# ---- 2b. global total rows = ASR-based totals (same refit as Table 7A) ----
SMAP = {
    "arima_damped_pop": "arima_asr_damped_pop",
    "arima_undamped_pop": "arima_asr_undamped_pop",
    "constant_2023_rate_damped_pop": "constant_2023_asr_damped_pop",
}
for m in ["Incidence", "Deaths", "DALYs"]:
    idx = scen[(scen.location == "Global") & (scen.measure_type == "total") &
               (scen.outcome == m)].index
    for i in idx:
        old_scen = scen.at[i, "scenario"]
        row = asr_scen[(asr_scen.measure == m) & (asr_scen.scenario == SMAP[old_scen])].iloc[0]
        scen.at[i, "rate_2050"] = float(row.asr_2050)
        scen.at[i, "number_2050"] = float(row.number_2050)
        o_rate = float(scen.at[i, "observed_2023_rate"])
        o_num = float(scen.at[i, "observed_2023_number"])
        scen.at[i, "rate_change_vs_2023_percent"] = (float(row.asr_2050) / o_rate - 1) * 100
        scen.at[i, "number_change_vs_2023_percent"] = (float(row.number_2050) / o_num - 1) * 100

# ---- 3. recent-trend attributable rows = total row x PAF 2050 -------------
for m in MEAS:
    tots = recent[(recent.location == "Global") & (recent.measure == m) &
                  (recent.outcome_type.isna()) & (recent.year == 2050)]
    if tots.empty:
        continue
    tot_rate = float(tots.rate.iloc[0])
    tot_num = float(tots.number.iloc[0])
    idx = recent[(recent.location == "Global") & (recent.measure == m) &
                 (recent.outcome_type == "metabolic_attributable") &
                 (recent.year == 2050)].index
    for i in idx:
        recent.at[i, "rate"] = tot_rate * paf50[m] / 100
        recent.at[i, "number"] = tot_num * paf50[m] / 100

attr.to_csv(ATTR, index=False)
scen.to_csv(SCEN, index=False)
recent.to_csv(RECENT, index=False)

print("\nGlobal attributable scenarios after fix:")
print(scen[(scen.location == "Global") &
           (scen.measure_type == "metabolic_attributable")]
      [["outcome", "scenario", "rate_2050", "number_2050", "paf_2050",
        "number_change_vs_2023_percent"]].round(4).to_string(index=False))
print("\nGlobal attributable rate 2050:")
print(attr[(attr.location == "Global") & (attr.year == 2050) &
           (attr.variable == "attributable_rate")]
      [["measure", "projected", "pi_low", "pi_high"]].round(4).to_string(index=False))
print("\nRecent-trend attributable 2050:")
print(recent[(recent.location == "Global") & (recent.year == 2050) &
             (recent.outcome_type == "metabolic_attributable")]
      [["measure", "rate", "number"]].round(4).to_string(index=False))
