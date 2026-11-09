"""Build manuscript/tables.xlsx from the results/ CSVs (GBD 2023, ages 15-49).

All values are read from results/; nothing is hard-coded except table titles,
footnotes and the README text.  Run with .venv/bin/python.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "manuscript" / "tables.xlsx"

DARK = PatternFill("solid", fgColor="1F3864")
WHITE_BOLD = Font(bold=True, color="FFFFFF", size=10)
TITLE_FONT = Font(bold=True, size=12)
PANEL_FONT = Font(bold=True, size=10, color="1F3864")
FOOT_FONT = Font(italic=True, size=8)

LEFT = Alignment(horizontal="left", vertical="top", wrap_text=True)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)

NUMFMT = {
    "int": "#,##0",
    "int_signed": "+#,##0;-#,##0",
    "rate1": "0.0",
    "rate2": "0.00",
    "eapc": "+0.00;-0.00",
    "ci3": "0.000",
    "pct1": '+0.0"%";-0.0"%"',
    "pct2": "0.00",
    "sci": "0.00E+00",
    "num2": "0.00",
}


# --------------------------------------------------------------------------- #
# value formatters (composite "estimate (interval)" strings)
# --------------------------------------------------------------------------- #
def n0(x: float) -> str:
    return f"{x:,.0f}"


def _sep(lo) -> str:
    """En dash between bounds, or 'to' when the lower bound is negative."""
    return " to " if lo < 0 else "\u2013"


def ui_int(v, lo, hi) -> str:
    return f"{n0(v)} ({n0(lo)}{_sep(lo)}{n0(hi)})"


def ui_rate(v, lo, hi, dp: int = 1) -> str:
    return f"{v:,.{dp}f} ({lo:,.{dp}f}{_sep(lo)}{hi:,.{dp}f})"


def ci(v, lo, hi, dp: int = 2) -> str:
    return f"{v:+.{dp}f} ({lo:.{dp}f} to {hi:.{dp}f})"


def ci_nosign(v, lo, hi, dp: int = 2) -> str:
    return f"{v:.{dp}f} ({lo:.{dp}f} to {hi:.{dp}f})"


def ci_ratio(v, lo, hi, dp: int = 3) -> str:
    return f"{v:.{dp}f} ({lo:.{dp}f} to {hi:.{dp}f})"


def paf(v, lo, hi) -> str:
    return f"{v:.2f} ({lo:.2f}{_sep(lo)}{hi:.2f})"


def yrs(s: str) -> str:
    return str(s).replace(";", ", ").replace("-", "\u2013")


def dash(x):
    if x is None:
        return "\u2014"
    if isinstance(x, float) and np.isnan(x):
        return "\u2014"
    return x


# --------------------------------------------------------------------------- #
# sheet writer
# --------------------------------------------------------------------------- #
def build_sheet(wb, name, title, widths, blocks, footnote):
    ws = wb.create_sheet(name)
    ncols = len(widths)
    last_col = get_column_letter(ncols)

    c = ws.cell(1, 1, title)
    c.font = TITLE_FONT
    c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=False)
    ws.merge_cells(f"A1:{last_col}1")
    ws.row_dimensions[1].height = 22

    r = 2
    filter_ref = None
    for bi, blk in enumerate(blocks):
        if blk.get("label"):
            if bi == 0:
                # Row 2 must carry the column headers, so the first panel label is
                # appended to the title instead of occupying its own row.
                c = ws.cell(1, 1, f"{title} \u2014 {blk['label'].rstrip('.')}")
                c.font = TITLE_FONT
                c.alignment = Alignment(horizontal="left", vertical="center")
            else:
                c = ws.cell(r, 1, blk["label"])
                c.font = PANEL_FONT
                c.alignment = Alignment(horizontal="left", vertical="center")
                ws.merge_cells(f"A{r}:{last_col}{r}")
                ws.row_dimensions[r].height = 18
                r += 1
        for j, h in enumerate(blk["headers"], 1):
            c = ws.cell(r, j, h)
            c.font = WHITE_BOLD
            c.fill = DARK
            c.alignment = CENTER
        ws.row_dimensions[r].height = 46
        if bi == 0:
            hdr_row = r
        r += 1
        start = r
        for row in blk["rows"]:
            for j, (val, fmt) in enumerate(zip(row, blk["fmts"]), 1):
                cell = ws.cell(r, j)
                val = dash(val)
                cell.value = val
                align = blk.get("align", ["left"] + ["center"] * (ncols - 1))[j - 1]
                cell.alignment = LEFT if align == "left" else CENTER
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if fmt == "p":
                        cell.number_format = "0.00E+00" if abs(val) < 0.001 else "0.000"
                    elif fmt in NUMFMT:
                        cell.number_format = NUMFMT[fmt]
            r += 1
        if bi == 0:
            filter_ref = f"A{hdr_row}:{last_col}{r - 1}"
        r += 1  # spacer row between blocks

    foot_row = r
    c = ws.cell(foot_row, 1, footnote)
    c.font = FOOT_FONT
    c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    ws.merge_cells(f"A{foot_row}:{last_col}{foot_row}")
    ws.row_dimensions[foot_row].height = 46

    for j, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(j)].width = w

    ws.freeze_panes = "A3"
    if filter_ref:
        ws.auto_filter.ref = filter_ref
    return ws


def block(headers, rows, fmts, label=None, align=None):
    b = {"headers": headers, "rows": rows, "fmts": fmts}
    if label:
        b["label"] = label
    if align:
        b["align"] = align
    return b


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
head = json.loads((RES / "headline.json").read_text())
gasr = pd.read_csv(RES / "global_asr_series.csv")
sasr = pd.read_csv(RES / "sex_asr_series.csv")
strend = pd.read_csv(RES / "sex_asr_trends.csv")
atrend = pd.read_csv(RES / "asr_trends.csv")
segs = pd.read_csv(RES / "asr_joinpoint_segments.csv")
jsens = pd.read_csv(RES / "joinpoint_sensitivity.csv")
attr = pd.read_csv(RES / "attributable_global.csv")
attr_sex = pd.read_csv(RES / "attributable_sex_2023.csv")
attr_tr = pd.read_csv(RES / "attributable_trends_eapc.csv")
sites = pd.read_csv(RES / "sites_global_2023.csv")
site_eapc = pd.read_csv(RES / "sites_eapc.csv")
site_paf = pd.read_csv(RES / "sites_paf_2023.csv")
site_att = pd.read_csv(RES / "sites_attributable_2023.csv")
ineq = pd.read_csv(RES / "inequality_indices.csv")
ineq_tr = pd.read_csv(RES / "inequality_trends.csv")
decomp = pd.read_csv(RES / "decomposition_global.csv")
kitagawa = pd.read_csv(RES / "age_composition_decomposition.csv")
proj = pd.read_csv(RES / "projection_asr_2050.csv")
scen = pd.read_csv(RES / "projections_scenarios_2050.csv")
asr_scen = pd.read_csv(RES / "projections_asr_scenarios_2050.csv")
proj_attr = pd.read_csv(RES / "projections_attributable.csv")
recent = pd.read_csv(RES / "projection_recent_trend.csv")
backtest = pd.read_csv(RES / "projection_backtest.csv")
diagn = pd.read_csv(RES / "projection_diagnostics.csv")
country_asr = pd.read_csv(RES / "country_asr_trends.csv")
country_attr = pd.read_csv(RES / "attributable_country_2023.csv")
ineq_country = pd.read_csv(RES / "inequality_country_data.csv")
apcie = pd.read_csv(RES / "apc_ie_effects.csv")
ageb = pd.read_csv(RES / "age_bands_global.csv")
sens_std = pd.read_csv(RES / "sensitivity_standard_pop.csv")
sens_rank = pd.read_csv(RES / "sensitivity_country_rank.csv")
sens_age = pd.read_csv(RES / "sensitivity_age_definition.csv")
regions = pd.read_csv(RES / "regions_2023.csv")
region_asr23 = pd.read_csv(RES / "regions_asr_2023.csv")
region_asr_tr = pd.read_csv(RES / "regions_asr_trends.csv")
sdiq = pd.read_csv(RES / "sdi_quintile_asr_trends.csv")
hac = pd.read_csv(RES / "hac_comparison.csv")
hac_lag = pd.read_csv(RES / "hac_lag_sensitivity.csv")
hac_sum = json.loads((RES / "hac_summary.json").read_text())
det = pd.read_csv(RES / "detection_excluded_summary.csv")
det_site = pd.read_csv(RES / "detection_site_contrib.csv")
gco_global = pd.read_csv(RES / "globocan_global.csv")
gco_sites = pd.read_csv(RES / "globocan_sites.csv")
gco_region = pd.read_csv(RES / "globocan_region.csv")
gco_gap = pd.read_csv(RES / "globocan_gap_by_site.csv")
gco_sum = json.loads((RES / "globocan_summary.json").read_text())
ssd = pd.read_csv(RES / "site_sex_decomposition.csv")

MEAS = ["Incidence", "Deaths", "DALYs"]
SEXES = [("both", "Both sexes"), ("male", "Male"), ("female", "Female")]
SEXMAP = {"Both": "Both sexes", "Male": "Male", "Female": "Female"}

wb = Workbook()
wb.remove(wb.active)

# --------------------------------------------------------------------------- #
# Table 1
# --------------------------------------------------------------------------- #
rows = []
for skey, slab in SEXES:
    for meas in MEAS:
        k90 = head[f"global_1990_15_49_{skey}"]
        k23 = head[f"global_2023_15_49_{skey}"]
        m = meas.lower()
        if m == "dalys":
            m = "dalys"
        v90, l90, h90 = (k90[f"{m}_number"][z] for z in ("val", "low", "high"))
        v23, l23, h23 = (k23[f"{m}_number"][z] for z in ("val", "low", "high"))
        r90, rl90, rh90 = (k90[f"{m}_rate"][z] for z in ("val", "low", "high"))
        r23, rl23, rh23 = (k23[f"{m}_rate"][z] for z in ("val", "low", "high"))
        if skey == "both":
            g90 = gasr[(gasr.year == 1990) & (gasr.measure == meas)].iloc[0]
            g23 = gasr[(gasr.year == 2023) & (gasr.measure == meas)].iloc[0]
            a90 = ui_rate(g90.asr_15_49, g90.asr_low, g90.asr_high)
            a23 = ui_rate(g23.asr_15_49, g23.asr_low, g23.asr_high)
            asr90, asr23 = g90.asr_15_49, g23.asr_15_49
        else:
            s90 = sasr[(sasr.year == 1990) & (sasr.measure == meas) & (sasr.sex == slab.split()[0])].iloc[0]
            s23 = sasr[(sasr.year == 2023) & (sasr.measure == meas) & (sasr.sex == slab.split()[0])].iloc[0]
            a90 = f"{s90.asr_15_49:,.1f}"
            a23 = f"{s23.asr_15_49:,.1f}"
            asr90, asr23 = s90.asr_15_49, s23.asr_15_49
        rows.append([
            slab, meas,
            ui_int(v90, l90, h90), ui_int(v23, l23, h23),
            (v23 / v90 - 1) * 100,
            ui_rate(r90, rl90, rh90), ui_rate(r23, rl23, rh23),
            a90, a23,
            (asr23 / asr90 - 1) * 100,
        ])
build_sheet(
    wb, "Table1",
    "Table 1. Global burden of early-onset cancer (ages 15\u201349), 1990 and 2023, by sex and outcome",
    [12, 12, 30, 30, 12, 24, 24, 22, 22, 12],
    [block(
        ["Sex", "Outcome", "1990 count (95% UI)", "2023 count (95% UI)",
         "Change in count (%)", "1990 crude rate (95% UI)", "2023 crude rate (95% UI)",
         "1990 ASR (95% UI)", "2023 ASR (95% UI)", "Change in ASR (%)"],
        rows, [None, None, None, None, "pct1", None, None, None, None, "pct1"])],
    "ASR = age-standardised rate, directly standardised within ages 15\u201349 to the 2023 global 15\u201349 "
    "population, so the 2023 ASR equals the 2023 crude rate; UI = GBD 95% uncertainty interval; crude and "
    "standardised rates per 100,000 population aged 15\u201349; change = relative change 1990\u20132023. "
    "Sex-specific ASR point estimates are shown without UI (no sex-specific ASR UI is produced by the "
    "pipeline; 1990/2023 values also appear in sex_asr_trends.csv). Source: GBD 2023 (release 8352), "
    "cause 1059, ages 15\u201349, results/headline.json and results/global_asr_series.csv.",
)

# --------------------------------------------------------------------------- #
# Table 2
# --------------------------------------------------------------------------- #
t2rows = []
for meas in MEAS:
    a = atrend[atrend.measure == meas].iloc[0]
    t2rows.append([
        meas, "Both sexes",
        ci(a.eapc_1990_2023, a.eapc_1990_2023_low, a.eapc_1990_2023_high),
        ci(a.eapc_2013_2023, a.eapc_2013_2023_low, a.eapc_2013_2023_high),
        yrs(a.joinpoints),
        ci(a.aapc, a.aapc_low, a.aapc_high),
        yrs(a.last_segment),
        ci(a.last_apc, a.last_apc_low, a.last_apc_high),
    ])
for meas in MEAS:
    for sex in ("Male", "Female"):
        s = strend[(strend.measure == meas) & (strend.sex == sex)].iloc[0]
        t2rows.append([
            meas, sex,
            ci(s.eapc_1990_2023, s.eapc_1990_2023_low, s.eapc_1990_2023_high),
            ci(s.eapc_2013_2023, s.eapc_2013_2023_low, s.eapc_2013_2023_high),
            yrs(s.joinpoints),
            ci(s.aapc, s.aapc_low, s.aapc_high),
            yrs(s.last_segment),
            ci(s.last_apc, s.last_apc_low, s.last_apc_high),
        ])
build_sheet(
    wb, "Table2",
    "Table 2. Trends in age-standardised incidence, mortality and DALY rates, 1990\u20132023 and 2013\u20132023",
    [12, 12, 24, 24, 18, 24, 14, 24],
    [block(
        ["Outcome", "Sex", "EAPC 1990\u20132023 (95% CI)", "EAPC 2013\u20132023 (95% CI)",
         "Joinpoint years", "AAPC 1990\u20132023 (95% CI)", "Last segment",
         "Last-segment APC (95% CI)"],
        t2rows, [None, None, None, None, None, None, None, None])],
    "EAPC = estimated annual percentage change from a log-linear regression of the annual ASR on calendar year; "
    "joinpoints selected by BIC with a maximum of 3; AAPC = average annual percentage change over 1990\u20132023 "
    "from the joinpoint model; APC = annual percentage change within the final joinpoint segment; CI = 95% "
    "confidence interval. Joinpoint counts and exact joinpoint years are unstable (BIC selected the maximum "
    "allowed number of joinpoints in every series; see Table S6), so the years shown should be read as "
    "approximate. AAPC and last-segment APC for sex-specific series come from results/sex_asr_trends.csv. "
    "Source: results/asr_trends.csv, results/sex_asr_trends.csv, results/asr_joinpoint_segments.csv.",
)

# --------------------------------------------------------------------------- #
# Table 3
# --------------------------------------------------------------------------- #
RISK_LABEL = {
    "Metabolic risks": "Metabolic risks (combined)",
    "High body-mass index": "High BMI",
    "High fasting plasma glucose": "High fasting plasma glucose",
}


def aval(risk, meas, metric, year, field="val"):
    m = attr[(attr.risk == risk) & (attr.measure == meas) & (attr.metric == metric)
             & (attr.year == year) & (attr.sex == "both") & (attr.age == "15-49")]
    return float(m.iloc[0][field])


t3a = []
for risk in ["Metabolic risks", "High body-mass index", "High fasting plasma glucose"]:
    for meas in ["Deaths", "DALYs"]:
        n90 = aval(risk, meas, "Number", 1990)
        p90 = aval(risk, meas, "Percent", 1990) * 100
        r90 = aval(risk, meas, "Rate", 1990)
        n23 = aval(risk, meas, "Number", 2023)
        p23 = aval(risk, meas, "Percent", 2023) * 100
        r23 = aval(risk, meas, "Rate", 2023)
        e_n = attr_tr[(attr_tr.risk == risk) & (attr_tr.measure == meas)
                      & (attr_tr.metric == "Number") & (attr_tr.period == "1990-2023")].iloc[0]
        e_r = attr_tr[(attr_tr.risk == risk) & (attr_tr.measure == meas)
                      & (attr_tr.metric == "Rate") & (attr_tr.period == "2013-2023")].iloc[0]
        t3a.append([
            RISK_LABEL[risk], meas,
            ui_int(n90, aval(risk, meas, "Number", 1990, "lower"), aval(risk, meas, "Number", 1990, "upper")),
            ui_int(n23, aval(risk, meas, "Number", 2023, "lower"), aval(risk, meas, "Number", 2023, "upper")),
            (n23 / n90 - 1) * 100,
            paf(p90, aval(risk, meas, "Percent", 1990, "lower") * 100, aval(risk, meas, "Percent", 1990, "upper") * 100),
            paf(p23, aval(risk, meas, "Percent", 2023, "lower") * 100, aval(risk, meas, "Percent", 2023, "upper") * 100),
            ui_rate(r90, aval(risk, meas, "Rate", 1990, "lower"), aval(risk, meas, "Rate", 1990, "upper"), 2),
            ui_rate(r23, aval(risk, meas, "Rate", 2023, "lower"), aval(risk, meas, "Rate", 2023, "upper"), 2),
            ci(e_n.eapc, e_n.ci_low, e_n.ci_high),
            ci(e_r.eapc, e_r.ci_low, e_r.ci_high),
        ])

t3b = []
for sex, slab in (("male", "Male"), ("female", "Female")):
    for meas in ["Deaths", "DALYs"]:
        s = attr_sex[(attr_sex.sex == sex) & (attr_sex.measure == meas)].iloc[0]
        t3b.append([
            slab, meas,
            ui_int(s.attributable_number_2023, s.attributable_number_2023_low, s.attributable_number_2023_high),
            paf(s.paf_2023_percent, s.paf_2023_low, s.paf_2023_high),
            ui_rate(s.attributable_rate_2023, s.attributable_rate_2023_low, s.attributable_rate_2023_high, 2),
            ci(s.eapc_rate_2013_2023, s.eapc_rate_2013_2023_ci_low, s.eapc_rate_2013_2023_ci_high),
        ])
t3_headers = ["Risk factor", "Outcome", "Attributable number 1990 (95% UI)", "Attributable number 2023 (95% UI)",
              "Change 1990\u20132023 (%)", "PAF 1990 % (95% UI)", "PAF 2023 % (95% UI)",
              "Attributable rate 1990 (95% UI)", "Attributable rate 2023 (95% UI)",
              "EAPC number 1990\u20132023 (95% CI)", "EAPC rate 2013\u20132023 (95% CI)"]
build_sheet(
    wb, "Table3",
    "Table 3. Metabolic-risk-attributable burden of early-onset cancer, 1990 and 2023",
    [26, 10, 28, 28, 13, 20, 20, 22, 22, 24, 24],
    [
        block(t3_headers, t3a, [None, None, None, None, "pct1", None, None, None, None, None, None],
              label="Panel A. Global attributable burden, both sexes"),
        block(["Sex", "Outcome", "Attributable number 2023 (95% UI)", "PAF 2023 % (95% UI)",
               "Attributable rate 2023 (95% UI)", "EAPC rate 2013\u20132023 (95% CI)"],
              t3b, [None, None, None, None, None, None], label="Panel B. Attributable burden by sex, 2023"),
    ],
    "PAF = population-attributable fraction, defined as attributable burden divided by total cancer burden "
    "excluding non-melanoma skin cancer; estimates come from the GBD comparative risk assessment for the "
    "metabolic risks high body-mass index and high fasting plasma glucose. Attributable rates per 100,000 "
    "population aged 15\u201349; PAF in percent; change = relative change in the attributable number "
    "1990\u20132023; EAPC = estimated annual percentage change. Rows for DALYs use DALY counts and rates. "
    "Source: GBD 2023 (release 8352), results/attributable_global.csv, results/attributable_sex_2023.csv, "
    "results/attributable_trends_eapc.csv.",
)

# --------------------------------------------------------------------------- #
# Table S11 (site-specific table, moved to the supplement)
# --------------------------------------------------------------------------- #
inc = sites[(sites.measure == "Incidence") & (sites.metric == "Number")].sort_values("val", ascending=False)
deaths = sites[(sites.measure == "Deaths") & (sites.metric == "Number")].set_index("cause")
dalys = sites[(sites.measure == "DALYs") & (sites.metric == "Number")].set_index("cause")
eapc_i = site_eapc[site_eapc.measure == "Incidence"].set_index("cause")
pafd = site_paf[(site_paf.measure == "Deaths")].set_index("site")
attd = site_att[(site_att.rei == 104) & (site_att.metric == "Number")].set_index("cause")
t4 = []
for _, row in inc.iterrows():
    c = row.cause
    d = deaths.loc[c]
    y = dalys.loc[c]
    e = eapc_i.loc[c]
    if c in pafd.index:
        pp = pafd.loc[c]
        paf_s = paf(pp.paf_percent, pp.paf_low, pp.paf_high)
    else:
        paf_s = "\u2014"
    if c in attd.index:
        aa = attd.loc[c]
        att_s = ui_int(aa.val, aa.lower, aa.upper)
    else:
        att_s = "\u2014"
    t4.append([c, ui_int(row.val, row.lower, row.upper), ui_int(d.val, d.lower, d.upper),
               ui_int(y.val, y.lower, y.upper), ci(e.eapc, e.ci_low, e.ci_high), paf_s, att_s])
build_sheet(
    wb, "TableS11",
    "Table S11. Site-specific burden of early-onset cancer, 2023, and metabolic attribution",
    [40, 28, 28, 30, 24, 20, 26],
    [block(["Cancer site", "Incident cases 2023 (95% UI)", "Deaths 2023 (95% UI)", "DALYs 2023 (95% UI)",
            "Incidence EAPC 1990\u20132023 (95% CI)", "Metabolic PAF deaths % (95% UI)",
            "Metabolic-attributable deaths (95% UI)"],
           t4, [None, None, None, None, None, None, None])],
    "The 29 GBD cancer causes (cause 1059 and its children) are listed by descending incident cases in 2023. "
    "Non-melanoma skin cancer is included as a site but is excluded from the all-cancer total, so the 29 sites "
    "do not sum to the global total (they cover ~97% of it); no total row is shown and site sums should not be "
    "read as the overall burden. Only the 14 sites with metabolic attribution carry a PAF and attributable "
    "death count (\u2014 = no metabolic attribution in GBD); those 14 sites sum exactly to the global "
    "metabolic-attributable total of 47,303 deaths. PAF = population-attributable fraction; EAPC = estimated "
    "annual percentage change; rates and UI as in Table 1. Source: results/sites_global_2023.csv, "
    "results/sites_eapc.csv, results/sites_paf_2023.csv, results/sites_attributable_2023.csv.",
)

# --------------------------------------------------------------------------- #
# Table 4 (was Table 5 before the display-item reduction)
# --------------------------------------------------------------------------- #
t5a = []
for _, r in ineq.iterrows():
    basis = {"asr": "ASR", "crude": "Crude rate", "paf": "PAF (%)"}[r.rate_type]
    t5a.append([
        r.outcome.replace(" (15-49", " (15\u201349").replace("rate)", "crude rate)"),
        "Both sexes" if r.sex == "both" else r.sex.capitalize(),
        basis,
        ci_ratio(r.concentration_index, r.concentration_index_ci_low, r.concentration_index_ci_high, 3),
        f"{r.sii:,.2f} ({r.sii_ci_low:,.2f} to {r.sii_ci_high:,.2f})",
        ci_ratio(r.rii, r.rii_ci_low, r.rii_ci_high, 3),
        float(r.rii_pvalue),
    ])

t5b = []
out_order = ["Total Incidence (15-49 ASR)", "Total Deaths (15-49 ASR)", "Metabolic PAF (deaths, %)"]
for oc in out_order:
    row = [oc.replace(" (15-49", " (15\u201349")]
    for yr in (1990, 2010, 2023):
        m = ineq_tr[(ineq_tr.outcome == oc) & (ineq_tr.year == yr)].iloc[0]
        row.append(f"{m.concentration_index:+.3f}")
    for yr in (1990, 2010, 2023):
        m = ineq_tr[(ineq_tr.outcome == oc) & (ineq_tr.year == yr)].iloc[0]
        row.append(ci_ratio(m.rii, m.rii_ci_low, m.rii_ci_high, 3))
    t5b.append(row)
build_sheet(
    wb, "Table4",
    "Table 4. Socioeconomic inequality in the early-onset cancer burden, 2023, and trends in inequality",
    [44, 12, 16, 26, 28, 26, 24],
    [
        block(["Outcome", "Sex", "Rate basis", "Concentration index (95% CI)", "SII (95% CI)",
               "RII (95% CI)", "p value"],
              t5a, [None, None, None, None, None, None, "p"],
              label="Panel A. Inequality indices across 204 countries, 2023"),
        block(["Outcome", "CI 1990", "CI 2010", "CI 2023", "RII 1990 (95% CI)", "RII 2010 (95% CI)",
               "RII 2023 (95% CI)"],
              t5b, [None, None, None, None, None, None, None],
              label="Panel B. Trends in concentration index and relative index of inequality, 1990\u20132023"),
    ],
    "Indices are population-weighted across the 204 GBD countries and territories ranked by the Socio-demographic "
    "Index (SDI). CI = concentration index, computed from fractional SDI rank; a positive CI indicates "
    "concentration of burden in higher-SDI (pro-rich) settings. SII = slope index of inequality (absolute "
    "difference across the SDI range); RII = relative index of inequality from a log-link model with the "
    "population-weighted fractional rank. p values refer to the RII test (two-sided). ASR-based indices are "
    "age-standardised within 15\u201349; sex-specific indices use crude rates; PAF indices use the metabolic "
    "cause-specific PAF. Panel A RII intervals come from the 2023 cross-sectional model and Panel B RII "
    "intervals from the fitted year-by-year trend model, so the two panels report different intervals for the "
    "same 2023 point estimate. Source: results/inequality_indices.csv, results/inequality_trends.csv.",
)

# --------------------------------------------------------------------------- #
# Table 5 (was Table 6 before the display-item reduction)
# --------------------------------------------------------------------------- #
t6a = []
for _, r in decomp.iterrows():
    y0, y1 = str(r.period).split("-")
    total_change = r.value_1 - r.value_0
    t6a.append([
        "Total deaths" if r.outcome == "total_deaths" else "Metabolic-attributable deaths",
        int(y0), int(y1), float(r.value_0), float(r.value_1), total_change,
        float(r.population_effect), float(r.rate_effect),
        dash(None if pd.isna(r.paf_effect) else float(r.paf_effect)),
    ])

t6b = []
for _, r in kitagawa[kitagawa.period == "1990-2023"].iterrows():
    t6b.append([r.measure, float(r.crude_1990), float(r.crude_2023), float(r.total_change),
                float(r.rate_effect), float(r.composition_effect), float(r.interaction),
                float(r.composition_share_percent)])
build_sheet(
    wb, "Table5",
    "Table 5. Decomposition of changes in the early-onset cancer burden, 1990\u20132023",
    [34, 9, 9, 16, 16, 16, 18, 16, 16],
    [
        block(["Outcome", "Start year", "End year", "Start value", "End value", "Total change",
               "Population effect", "Rate effect", "PAF effect"],
              t6a, [None, "int", "int", "int", "int", "int_signed", "int_signed", "int_signed", "int_signed"],
              label="Panel A. Das Gupta symmetric decomposition of changes in death counts"),
        block(["Measure", "Crude rate 1990", "Crude rate 2023", "Total change", "Rate effect",
               "Composition effect", "Interaction", "Composition share (%)"],
              t6b, [None, "rate1", "rate1", "num2", "num2", "num2", "num2", "rate1"],
              label="Panel B. Kitagawa decomposition of the change in the crude rate, 1990\u20132023"),
    ],
    "Panel A: Das Gupta symmetric decomposition of the change in death counts into population growth, "
    "age-specific rate change and (for metabolic-attributable deaths only) change in the PAF, so population + "
    "rate + PAF effects sum to the total change. Panel B: Kitagawa decomposition of the change in the crude "
    "rate per 100,000 aged 15\u201349 into rate and age-composition effects plus an interaction term; the "
    "composition share is the composition effect as a percentage of the total change. Source: "
    "results/decomposition_global.csv, results/age_composition_decomposition.csv.",
)

# --------------------------------------------------------------------------- #
# Table S12 (projections, moved to the supplement)
# --------------------------------------------------------------------------- #
asc = asr_scen.set_index(["measure", "scenario"])
rt2050 = recent[(recent.location == "Global") & (recent.year == 2050)
                & (recent.outcome_type.isna())].set_index("measure")
t7a = []
for _, r in proj.iterrows():
    prim = asc.loc[(r.measure, "arima_asr_damped_pop")]
    t7a.append([
        r.measure,
        f"{r.asr_2023:,.1f}",
        ui_rate(prim.asr_2050, prim.asr_2050_low, prim.asr_2050_high),
        float(prim.rate_change_percent),
        ui_int(prim.number_2050, prim.number_2050_low, prim.number_2050_high),
        float(asc.loc[(r.measure, "constant_2023_asr_damped_pop"), "number_2050"]),
        float(asc.loc[(r.measure, "arima_asr_undamped_pop"), "number_2050"]),
        float(rt2050.loc[r.measure, "number"]),
    ])

pa_deaths = proj_attr[(proj_attr.location == "Global") & (proj_attr.year == 2050)
                      & (proj_attr.variable == "paf_percent")].set_index("measure")
pa_rate = proj_attr[(proj_attr.location == "Global") & (proj_attr.year == 2050)
                    & (proj_attr.variable == "attributable_rate")].set_index("measure")
att_scen = scen[(scen.location == "Global") & (scen.measure_type == "metabolic_attributable")].set_index(
    ["outcome", "scenario"])
rt_att = recent[(recent.location == "Global") & (recent.year == 2050)
                & (recent.outcome_type == "metabolic_attributable")].set_index("measure")
t7b = []
for meas, scen_name in (("Deaths", "Attributable deaths"), ("DALYs", "Attributable dalys")):
    paf_2023 = float(attr_sex[(attr_sex.sex == "both") & (attr_sex.measure == meas)].iloc[0].paf_2023_percent)
    t7b.append([
        meas,
        f"{paf_2023:.2f}",
        paf(pa_deaths.loc[meas, "projected"], pa_deaths.loc[meas, "pi_low"], pa_deaths.loc[meas, "pi_high"]),
        ui_rate(pa_rate.loc[meas, "projected"], pa_rate.loc[meas, "pi_low"], pa_rate.loc[meas, "pi_high"], 3),
        float(att_scen.loc[(scen_name, "constant_2023_rate_damped_pop")].number_2050),
        float(att_scen.loc[(scen_name, "risk_control_paf_frozen_2023")].number_2050),
        float(att_scen.loc[(scen_name, "arima_damped_pop")].number_2050),
        float(att_scen.loc[(scen_name, "arima_undamped_pop")].number_2050),
        float(rt_att.loc[meas, "number"]),
    ])
build_sheet(
    wb, "TableS12",
    "Table S12. Projections of the early-onset cancer burden to 2050 under alternative scenarios",
    [12, 12, 24, 24, 30, 24, 26, 26, 26],
    [
        block(["Outcome", "ASR 2023", "ASR 2050 (95% PI)", "Change in ASR 2023\u20132050 (%)",
               "Number 2050 (95% PI)", "Constant 2023-ASR number 2050",
               "ARIMA on ASR, undamped population, number 2050", "Recent-trend number 2050"],
              t7a, [None, None, None, "pct1", None, "int", "int", "int"],
              label="Panel A. Total burden in 2050"),
        block(["Outcome", "PAF 2023 %", "PAF 2050 % (95% PI)",
               "Attributable rate 2050 per 100,000 (95% PI)",
               "Attributable number 2050: constant rate", "Attributable number 2050: PAF frozen at 2023",
               "Attributable number 2050: ARIMA, damped population",
               "Attributable number 2050: ARIMA, undamped population",
               "Attributable number 2050: recent trend"],
              t7b, [None, None, None, None, "int", "int", "int", "int", "int"],
              label="Panel B. Metabolic-attributable burden in 2050"),
    ],
    "Panel A applies ARIMA models to the log directly age-standardised rate (ASR, ages 15\u201349, 2023 "
    "global weights) and multiplies by the projected population aged 15\u201349 (4.25 billion in 2050 under "
    "the damped scenario). The primary, constant-ASR and undamped-population counts all come from the same "
    "ASR refit (results/projections_asr_scenarios_2050.csv), so there is no cross-refit discrepancy. 95% "
    "PI = 95% prediction interval. 'Recent trend' extends the 2013\u20132023 log-linear crude-rate trend and "
    "is an upper bound. Panel B derives every attributable number as the corresponding total number in Panel A "
    "multiplied by the projected PAF (4.81% for deaths, 4.28% for DALYs in 2050), or by the 2023 PAF in the "
    "risk-control scenario, so Panels A and B reconcile exactly; the same derivation is used for the "
    "attributable rate (ASR x PAF). PAF = population-attributable fraction; projections are illustrative "
    "scenarios, not forecasts, and ARIMA did not outperform simple benchmarks in backtesting (Table S9). Source: "
    "results/projections_asr_scenarios_2050.csv, results/projection_asr_2050.csv, "
    "results/projections_attributable.csv, results/projections_scenarios_2050.csv, "
    "results/projection_recent_trend.csv.",
)

# --------------------------------------------------------------------------- #
# Table S13 - sensitivity analyses: 1990 standard population and ages 15-39
# --------------------------------------------------------------------------- #
STD_NAME = {"2023 global 15-49": "2023 standard (manuscript)",
            "1990 global 15-49": "1990 standard"}
s13a = []
for std in ("2023 global 15-49", "1990 global 15-49"):
    for meas in MEAS:
        d = sens_std[(sens_std.scope == "Global") & (sens_std.standard == std) &
                     (sens_std.measure == meas)].set_index("year")
        r0, r23 = d.loc[1990], d.loc[2023]
        s13a.append([
            STD_NAME[std], meas, float(r0["asr"]), float(r23["asr"]),
            (float(r23["asr"]) / float(r0["asr"]) - 1) * 100,
            ci(float(r23.eapc_1990_2023), float(r23.eapc_1990_2023_low),
               float(r23.eapc_1990_2023_high)),
            ci(float(r23.eapc_2013_2023), float(r23.eapc_2013_2023_low),
               float(r23.eapc_2013_2023_high)),
            f"{float(r23.last_apc):+.2f} ({yrs(r23.last_segment)})",
        ])
assert len(s13a) == 6

REGION_SCOPES = [s for s in sens_std.scope.unique()
                 if s != "Global" and not str(s).startswith("SDI ")]
assert len(REGION_SCOPES) == 21, REGION_SCOPES  # all 21 GBD regions
s13b = []
for meas in MEAS:
    d = sens_rank[sens_rank.measure == meas]
    rho = d[["rank_2023", "rank_1990"]].corr(method="spearman").iloc[0, 1]
    top20 = len(set(d.nsmallest(20, "rank_2023").country) &
                set(d.nsmallest(20, "rank_1990").country))
    shift = int((d.quartile_2023 - d.quartile_1990).abs().ge(2).sum())
    wide = (sens_std[sens_std.scope.isin(REGION_SCOPES) & (sens_std.year == 2023) &
                     (sens_std.measure == meas)]
            .pivot_table(index="scope", columns="standard", values="asr"))
    rrho = wide.corr(method="spearman").iloc[0, 1]
    top5 = len(set(wide["2023 global 15-49"].nlargest(5).index) &
               set(wide["1990 global 15-49"].nlargest(5).index))
    s13b.append([meas, float(rho), f"{top20}/20", f"{shift}/204", float(rrho), f"{top5}/5"])
assert len(s13b) == 3

s13c = []
for q in ("SDI Q1 lowest", "SDI Q5 highest"):
    d = sens_std[(sens_std.scope == q) & (sens_std.measure == "Incidence") &
                 (sens_std.standard == "1990 global 15-49")].set_index("year")
    s13c.append([q.replace("SDI ", ""), "Incidence", float(d.loc[1990, "asr"]),
                 float(d.loc[2023, "asr"]),
                 (float(d.loc[2023, "asr"]) / float(d.loc[1990, "asr"]) - 1) * 100])

s13d = []
for scope, label in (("Global", "Both sexes"), ("Global Male", "Male"),
                     ("Global Female", "Female")):
    for meas in MEAS:
        d = sens_age[(sens_age.analysis == "B_age_definition") &
                     (sens_age.scope == scope) & (sens_age.measure == meas)].set_index("year")
        r0, r19, r23 = d.loc[1990], d.loc[2019], d.loc[2023]
        s13d.append([
            label, meas, float(r0["count"]), float(r23["count"]),
            (float(r23["count"]) / float(r0["count"]) - 1) * 100,
            float(r0["asr"]), float(r19["asr"]), float(r23["asr"]),
            (float(r23["asr"]) / float(r19["asr"]) - 1) * 100,
            (float(r23["asr"]) / float(r0["asr"]) - 1) * 100,
            ci(float(r23.eapc_1990_2023), float(r23.eapc_1990_2023_low),
               float(r23.eapc_1990_2023_high)),
            ci(float(r23.eapc_2013_2023), float(r23.eapc_2013_2023_low),
               float(r23.eapc_2013_2023_high)),
            f"{float(r23.last_apc):+.2f} ({yrs(r23.last_segment)})",
        ])
assert len(s13d) == 9

s13e = []
for scope, label in (("Global 20-39", "Ages 20\u201339 (defensible)"),
                     ("Global 15-39 (approximate)", "Ages 15\u201339 (approximate, upper bound)")):
    for meas in ("Deaths", "DALYs"):
        d = sens_age[(sens_age.scope == scope) & (sens_age.measure == meas)].set_index("year")
        r0, r23 = d.loc[1990], d.loc[2023]
        s13e.append([label, meas, float(r0["paf"]), float(r0["attributable_count"]),
                     float(r23["paf"]), float(r23["attributable_count"])])
assert len(s13e) == 4

build_sheet(
    wb, "TableS13",
    "Table S13. Sensitivity of the main results to the standard population and the upper age bound",
    [26, 12, 11, 11, 13, 24, 24, 20, 11, 13, 24, 24, 20],
    [
        block(["Standard population", "Measure", "ASR 1990", "ASR 2023",
               "Change 1990\u20132023 (%)", "EAPC 1990\u20132023 (95% CI)",
               "EAPC 2013\u20132023 (95% CI)", "Last-segment APC (years)"],
              s13a, [None, None, "rate1", "rate1", "pct1", None, None, None],
              label="Panel A. Global directly age-standardised rates under the 1990 and 2023 standard populations"),
        block(["Measure", "Country rank Spearman \u03c1", "Top-20 overlap",
               "Countries moving >1 quartile", "Region rank Spearman \u03c1", "Top-5 region overlap"],
              s13b, [None, "num2", None, None, "num2", None],
              label="Panel B. Concordance of the 2023 country and regional rankings between the two standards"),
        block(["SDI quintile", "Measure", "ASR 1990", "ASR 2023", "Change 1990\u20132023 (%)"],
              s13c, [None, None, "rate1", "rate1", "pct1"],
              label="Panel C. Incidence at the SDI extremes under the 1990 standard"),
        block(["Sex", "Measure", "Count 1990", "Count 2023", "Count change (%)",
               "ASR 1990", "ASR 2019", "ASR 2023", "ASR change 2019\u20132023 (%)",
               "ASR change 1990\u20132023 (%)", "EAPC 1990\u20132023 (95% CI)",
               "EAPC 2013\u20132023 (95% CI)", "Last-segment APC (years)"],
              s13d, [None, None, "int", "int", "pct1", "rate1", "rate1", "rate1",
                     "pct1", "pct1", None, None, None],
              label="Panel D. Ages 15\u201339 instead of 15\u201349: counts, ASRs and trends"),
        block(["PAF definition", "Measure", "PAF 1990 (%)", "Attributable 1990",
               "PAF 2023 (%)", "Attributable 2023"],
              s13e, [None, None, "pct2", "int", "pct2", "int"],
              label="Panel E. Metabolic-attributable fraction under the narrower age definition"),
    ],
    "Sensitivity analyses requested by the review process. Panel A: the 1990 standard re-weights the same "
    "age-specific rates to the 1990 global 15\u201349 population; the 2023 standard is the manuscript standard, "
    "and by construction its 2023 ASR equals the 2023 crude rate. Panel B: country ranks are compared across "
    "all 204 countries (1 = highest ASR); region ranks across all 21 GBD regions. Panel C: "
    "equal fifths of the 2023 SDI index. Panel D: the upper age bound is lowered from 49 to 39 years (ages "
    "15\u201319 to 35\u201339). The 1990\u20132023 incidence EAPC at 15\u201339 is not significant "
    "(-0.10, 95% CI -0.26 to 0.06, p=0.22), whereas at 15\u201349 it is -0.29 (p<0.001), so the long-run "
    "incidence decline is concentrated at ages 40\u201349. Panel E: the risk panel has no 15\u201319 age group, "
    "so the defensible PAF is for ages 20\u201339; the 15\u201339 row applies the 20\u201324 PAF to the 15\u201319 "
    "total and is an assumption-based upper bound (metabolic PAF rises steeply with age), not a 15\u201339 "
    "estimate. All ASRs here are point estimates: GBD publishes marginal age-specific uncertainty intervals but "
    "not the age-specific covariance needed to re-weight them, so no GBD uncertainty interval is propagated; "
    "EAPC and APC intervals are analytic 95% CIs. Joinpoint models selected the maximum number of joinpoints, "
    "so the exact years are unstable and only the direction of the last segment should be interpreted. "
    "Region- and SDI-level values are population-weighted aggregations of country ASRs. Source: "
    "results/sensitivity_standard_pop.csv, results/sensitivity_country_rank.csv, "
    "results/sensitivity_age_definition.csv; computed by scripts/sensitivity_asr.py.",
)

# --------------------------------------------------------------------------- #
# Table S1
# --------------------------------------------------------------------------- #
# Country EAPCs are for the DIRECTLY AGE-STANDARDISED series (2023 global 15-49
# weights), not the GBD crude rates; trends_country_eapc_fdr.csv is superseded.
LOC_IDS = country_attr[["location_id", "location"]].drop_duplicates()
assert len(LOC_IDS) == 204 and LOC_IDS.location_id.is_unique
ct = country_asr.merge(LOC_IDS, on="location", how="left", validate="many_to_one")
assert ct.location_id.notna().all() and len(ct) == 204 * 3
PERIODS = [("1990-2023", "1990\u20132023"), ("2013-2023", "2013\u20132023")]
s1 = []
for meas in MEAS:
    for pkey, plabel in PERIODS:
        stem = "eapc_" + pkey.replace("-", "_")
        sub = ct[ct.measure == meas].sort_values(stem, ascending=False)
        for _, r in sub.iterrows():
            q = float(r[f"{stem}_q"])
            s1.append([
                int(r.location_id), r.location, meas, plabel,
                float(r.asr_1990), float(r.asr_2023), float(r.asr_change_percent),
                float(r[stem]), float(r[f"{stem}_low"]), float(r[f"{stem}_high"]),
                float(r[f"{stem}_p"]), q, "Yes" if q < 0.05 else "No",
            ])
assert len(s1) == 204 * 3 * 2
build_sheet(
    wb, "TableS1",
    "Table S1. Country-level trends in directly age-standardised incidence, mortality and DALY rates, "
    "1990\u20132023 and 2013\u20132023, both sexes",
    [11, 40, 12, 11, 10, 10, 14, 9, 9, 9, 11, 11, 13],
    [block(["location_id", "Location", "Measure", "Period", "ASR 1990", "ASR 2023",
            "Change in ASR 1990\u20132023 (%)", "EAPC", "95% CI lower", "95% CI upper",
            "p value", "q value", "FDR-significant"],
           s1, [None, None, None, None, "rate1", "rate1", "pct1", "eapc", "eapc", "eapc",
                "p", "p", None],
           align=["center", "left", "center", "center", "center", "center", "center",
                  "center", "center", "center", "center", "center", "center"])],
    "ASR = directly age-standardised rate within ages 15\u201349 to the 2023 global 15\u201349 population "
    "(per 100,000), computed from the national age-specific series rather than taken from the GBD crude "
    "rates; by construction the 2023 ASR equals the 2023 crude rate. Each country appears six times (three "
    "measures \u00d7 two trend periods), and the two ASR columns and the 1990\u20132023 change are identical "
    "on the 1990\u20132023 and 2013\u20132023 rows of each country\u2013measure pair. EAPC = estimated annual percentage change in the ASR from a log-linear "
    "regression on calendar year; p values are from that model and q values are Benjamini\u2013Hochberg "
    "false-discovery-rate adjusted p values within each measure\u2013period family (204 countries). "
    "FDR-significant = q < 0.05. Rows are sorted by measure, period and descending EAPC. "
    "Source: results/country_asr_trends.csv.",
)

# --------------------------------------------------------------------------- #
# Table S14
# --------------------------------------------------------------------------- #
hac = hac.copy()
hac["analysis"] = pd.Categorical(
    hac.analysis, ["Global ASR", "Sex-specific ASR", "RII trend", "Site ASR"], ordered=True)
hac = hac.sort_values(["analysis", "series", "measure", "period"])
s14a = []
for _, r in hac.iterrows():
    changed = bool(r.sig_conv) != bool(r.sig_hac)
    s14a.append([
        str(r.analysis), r.series, r.measure, r.period,
        float(r.estimate_conv),
        ci_nosign(float(r.estimate_conv), float(r.ci_low_conv), float(r.ci_high_conv)),
        float(r.p_conv),
        ci_nosign(float(r.estimate_hac), float(r.ci_low_hac), float(r.ci_high_hac)),
        float(r.p_hac), float(r.se_ratio), "Yes" if changed else "No",
    ])
hac_lag = hac_lag.sort_values(["analysis", "series", "measure", "period", "maxlags"])
s14b = []
for _, r in hac_lag.iterrows():
    s14b.append([
        r.analysis, r.series, r.measure, r.period, int(r.maxlags),
        float(r.estimate), ci_nosign(float(r.estimate), float(r.ci_low), float(r.ci_high)),
        float(r.p), "Yes" if bool(r.sig) else "No",
    ])
changed_txt = "; ".join(
    f"{c['series']} ({c['measure']}, {c['period']})"
    for c in hac_sum["changed_series"])
s14c = [
    ["Trend series tested", str(hac_sum["n_series"])],
    ["Series with a change in significance at p<0.05", str(hac_sum["n_significance_changed"])],
    ["Series with a change in significance", changed_txt],
    ["Median HAC/conventional SE ratio", f"{hac_sum['se_ratio_median']:.2f}"],
    ["95th percentile of the SE ratio", f"{hac_sum['se_ratio_p95']:.2f}"],
    ["Maximum SE ratio", f"{hac_sum['se_ratio_max']:.2f}"],
    ["Largest SE ratio", f"{hac_sum['max_se_ratio_series']['series']} "
                         f"({hac_sum['max_se_ratio_series']['measure']}, "
                         f"{hac_sum['max_se_ratio_series']['period']})"],
    ["Series with a change in sign of the estimate", str(hac_sum["direction_disagreements"])],
    ["HAC lag rule", "floor(4 x (T/100)^(2/9)): 3 for 34 years, 2 for 11 years"],
]
build_sheet(
    wb, "TableS14",
    "Table S14. Serial-correlation-robust inference: conventional versus Newey\u2013West HAC standard errors",
    [18, 34, 12, 11, 10, 24, 11, 24, 11, 9, 13],
    [
        block(["Analysis", "Series", "Measure", "Period", "Estimate",
               "95% CI conventional", "p conventional", "95% CI HAC", "p HAC",
               "SE ratio (HAC/conventional)", "Significance changed"],
              s14a, [None, None, None, None, "num2", None, "p", None, "p", "num2", None],
              label="Panel A. All trend series, conventional versus HAC inference"),
        block(["Analysis", "Series", "Measure", "Period", "Maximum lag",
               "Estimate", "95% CI", "p value", "Significant"],
              s14b, [None, None, None, None, None, "num2", None, "p", None],
              label="Panel B. Sensitivity of HAC inference to the maximum lag length"),
        block(["Metric", "Value"], s14c, [None, None],
              label="Panel C. Summary of the comparison"),
    ],
    "The estimate is the EAPC (% per year) for the ASR analyses and the annual slope of the log RII for the "
    "RII trend analyses, so units differ across analysis rows. Conventional standard errors assume independent "
    "residuals; HAC standard errors are Newey\u2013West with a Bartlett kernel and the lag rule in Panel C. "
    "A HAC standard error larger than the conventional one indicates positive serial correlation that the "
    "conventional interval understates. Panel B refits each headline series with 1\u20135 lags. The headline "
    "direction and significance of every global, sex-specific and RII trend is unchanged; four of 58 "
    "site-specific incidence or mortality series lose significance, none changes direction. Full series are "
    "in results/hac_comparison.csv and results/hac_lag_sensitivity.csv.",
)

# --------------------------------------------------------------------------- #
# Table S15
# --------------------------------------------------------------------------- #
sex_order = {"both": 0, "Male": 1, "Female": 2}
scen_order = {"Thyroid only": 0, "Core detection-sensitive": 1,
              "Core plus breast": 2, "All cancers (reference)": 3}
det = det.copy()
det["_s"] = det.sex.map(sex_order)
det["_c"] = det.scenario.map(scen_order)
det = det.sort_values(["_s", "_c"])
s15a = []
for _, r in det.iterrows():
    s15a.append([
        {"both": "Both sexes"}.get(r.sex, r.sex), r.scenario,
        float(r.asr_1990), float(r.asr_2013), float(r.asr_2023),
        float(r.pct_change_1990_2023), float(r.pct_change_2013_2023),
        ci(float(r.eapc_2013_2023), float(r.eapc_2013_2023_low), float(r.eapc_2013_2023_high)),
        ci(float(r.last_apc), float(r.last_apc_low), float(r.last_apc_high)),
        float(r.share_of_2013_2023_rise_removed), float(r.coverage_of_aggregate_2023 * 100),
        "None (reference)" if not isinstance(r.excluded_sites, str) else
        r.excluded_sites.replace(",", ", "),
    ])
det_site = det_site.copy()
det_site["_s"] = det_site.sex.map(sex_order)
det_site = det_site.sort_values(["_s", "change_2013_2023"], ascending=[True, False])
s15b = []
for _, r in det_site.iterrows():
    s15b.append([
        {"both": "Both sexes"}.get(r.sex, r.sex), r.site,
        float(r.asr_2013), float(r.asr_2023), float(r.change_2013_2023),
        float(r.share_of_total_change_pct), "Yes" if bool(r.detection_sensitive) else "No",
    ])
build_sheet(
    wb, "TableS15",
    "Table S15. Detection-effect sensitivity: incidence after removing diagnostic-intensity-sensitive sites",
    [12, 26, 10, 10, 10, 13, 13, 20, 20, 15, 15, 52],
    [
        block(["Sex", "Scenario", "ASR 1990", "ASR 2013", "ASR 2023",
               "Change 1990\u20132023 (%)", "Change 2013\u20132023 (%)",
               "EAPC 2013\u20132023 (95% CI)", "Final-segment APC (95% CI)",
               "Share of 2013\u20132023 rise removed (%)", "Coverage of aggregate 2023 (%)",
               "Sites removed"],
              s15a, [None, None, "rate1", "rate1", "rate1", "pct1", "pct1", None, None,
                     "pct1", "pct1", None],
              label="Panel A. Pre-specified exclusion scenarios, by sex"),
        block(["Sex", "Site", "ASR 2013", "ASR 2023", "Change 2013\u20132023",
               "Share of total 2013\u20132023 change (%)", "Detection-sensitive"],
              s15b, [None, None, "rate1", "rate1", "num2", "pct1", None],
              label="Panel B. Site contributions to the 2013\u20132023 change in incidence"),
    ],
    "Sites are subtracted in age-standardised space: the aggregate and every site are directly standardised "
    "to the same 2023 global 15\u201349 weights, so the residual is the ASR of the cancer set with the "
    "excluded sites removed. Scenarios are nested: thyroid alone; thyroid, prostate, kidney and melanoma "
    "(female: thyroid, kidney and melanoma); and the same core set plus breast cancer. The share of the "
    "2013\u20132023 rise removed is (ASR removed in 2023 \u2212 ASR removed in 2013) / (all-cancer ASR in "
    "2023 \u2212 all-cancer ASR in 2013) x 100. Coverage is the sum of the named site ASRs divided by the "
    "all-cancer ASR in 2023; the shortfall is the residual \u2018other malignant neoplasms\u2019 category. "
    "All cancers (reference) repeats the main analysis. Source: results/detection_excluded_summary.csv and "
    "results/detection_site_contrib.csv; computed by scripts/detection_excluded.py.",
)

# --------------------------------------------------------------------------- #
# Table S16
# --------------------------------------------------------------------------- #
gco_global = gco_global.copy()
gco_global["_s"] = gco_global.sex.map({"Both sexes": 0, "Male": 1, "Female": 2})
gco_global["_m"] = gco_global.measure.map({"Incidence": 0, "Deaths": 1})
gco_global = gco_global.sort_values(["_s", "_m"])
s16a = []
for _, r in gco_global.iterrows():
    s16a.append([
        r.sex, r.measure, float(r.globocan_count), float(r.globocan_crude),
        float(r.gbd_count), float(r.gbd_crude), float(r.count_ratio), float(r.rate_ratio),
    ])
gco_sites = gco_sites.copy()
gco_sites["_m"] = gco_sites.measure.map({"Incidence": 0, "Deaths": 1})
gco_sites["_c"] = (~gco_sites.comparable).astype(int)
gco_sites = gco_sites.sort_values(["_m", "_c", "gbd_2023"], ascending=[True, True, False])
s16b = []
for _, r in gco_sites.iterrows():
    s16b.append([
        r.measure, r.site, float(r.gbd_2023), float(r.globocan_2024), float(r.ratio),
        float(r.gbd_share), float(r.globocan_share), float(r.share_difference_pp),
        "Yes" if bool(r.comparable) else "No",
    ])
s16c = []
for meas in ("Incidence", "Deaths"):
    c = gco_sum["country"][meas]
    s16c.append(["Country", meas, int(c["n_countries"]), float(c["spearman_rho"]),
                 float(c["median_ratio"]),
                 f'{c["iqr_ratio"][0]:.2f}\u2013{c["iqr_ratio"][1]:.2f}',
                 int(c["top20_overlap"]), float(c["share_within_25pct"])])
    r = gco_region[gco_region.measure == meas]
    rr = gco_sum["region"][meas]
    top5 = len(set(r.nlargest(5, "gbd_crude").region) &
               set(r.nlargest(5, "globocan_crude").region))
    within = float(np.mean((r.ratio > 0.75) & (r.ratio < 1.25)) * 100)
    s16c.append(["GBD region", meas, int(rr["n_regions"]), float(rr["spearman_rho"]),
                 float(rr["median_ratio"]),
                 f'{rr["iqr_ratio"][0]:.2f}\u2013{rr["iqr_ratio"][1]:.2f}',
                 top5, within])
s16d = []
for _, r in gco_gap.nlargest(10, "absolute_difference").iterrows():
    s16d.append([r.site, float(r.gbd_2023), float(r.globocan_2024),
                 float(r.difference), float(r.ratio)])
build_sheet(
    wb, "TableS16",
    "Table S16. External benchmark against GLOBOCAN 2024, ages 15\u201349",
    [14, 40, 13, 13, 13, 13, 12, 12, 12, 16, 14],
    [
        block(["Sex", "Measure", "GLOBOCAN 2024 cases/deaths", "GLOBOCAN crude rate",
               "GBD 2023 cases/deaths", "GBD crude rate", "Count ratio (GCO/GBD)",
               "Rate ratio (GCO/GBD)"],
              s16a, [None, None, "int", "rate1", "int", "rate1", "num2", "num2"],
              label="Panel A. Global and sex-specific levels"),
        block(["Measure", "Site", "GBD 2023", "GLOBOCAN 2024", "Ratio (GCO/GBD)",
               "GBD share (%)", "GLOBOCAN share (%)", "Difference (percentage points)",
               "Directly comparable"],
              s16b, [None, None, "int", "int", "num2", "num2", "num2", "num2", None],
              label="Panel B. Site-specific comparison"),
        block(["Level", "Measure", "N", "Spearman \u03c1", "Median ratio",
               "Ratio IQR", "Top-20 overlap (country) / top-5 (region)",
               "Within \u00b125% (%)"],
              s16c, [None, None, None, "num2", "num2", None, None, "num2"],
              label="Panel C. Concordance of country and regional rankings"),
        block(["Site", "GBD 2023", "GLOBOCAN 2024", "Difference (GCO \u2212 GBD)",
               "Ratio (GCO/GBD)"],
              s16d, [None, "int", "int", "int_signed", "num2"],
              label="Panel D. Sites with the largest absolute difference, incidence"),
    ],
    "GLOBOCAN 2024 (IARC Global Cancer Observatory, Cancer Today v3 API, accessed 9 November 2026) "
    "provides estimates for 186 countries in 2024; the comparison uses all cancers excluding non-melanoma "
    "skin cancer and the seven GLOBOCAN five-year age groups from 15\u201319 to 45\u201349 (population "
    "4.04 billion). GBD values are for 2023 and ages 15\u201349 (population 4.00 billion). Rates are crude "
    "rates within 15\u201349, so the comparison is not affected by the choice of standard population; each "
    "source uses its own population denominator. Global incidence agrees closely (GLOBOCAN/GBD count ratio "
    "1.005; crude-rate ratio 0.996), whereas GBD deaths are about one-fifth higher (count ratio 0.824; "
    "rate ratio 0.816). Panel B excludes GBD \u2018other malignant neoplasms\u2019 (cause 489) from the "
    "concordance statistics because it bundles a different set of residual categories from GLOBOCAN "
    "\u2018other specified\u2019 and \u2018unspecified\u2019 (27 sites are directly comparable). Panel D "
    "ranks the 27 comparable sites by absolute difference; because the all-cancer totals agree closely, "
    "site-level differences largely offset (thyroid +0.36 million, cervical \u22120.26 million) and their "
    "net difference is \u22120.09 million. Country and region ratios are GLOBOCAN crude rate divided by GBD "
    "crude rate. Source: results/globocan_global.csv, results/globocan_sites.csv, "
    "results/globocan_region.csv, results/globocan_gap_by_site.csv and results/globocan_summary.json; "
    "computed by scripts/globocan_benchmark.py.",
)

# --------------------------------------------------------------------------- #
# Table S17
# --------------------------------------------------------------------------- #
def ssd_rows(sex, period, outcome, n):
    s = ssd[(ssd.sex == sex) & (ssd.period == period) & (ssd.outcome == outcome)]
    s = s.sort_values("change", ascending=False).head(n)
    return s


def ssd_row(r, with_paf=False):
    row = [r.site, float(r.value_0), float(r.value_1), float(r.change),
           float(r.population_effect), float(r.rate_effect)]
    if with_paf:
        row.append(float(r.paf_effect))
    row += [float(r.population_share), float(r.rate_share)]
    if with_paf:
        row.append(float(r.paf_share))
    return row


s17a = [ssd_row(r) for _, r in ssd_rows("Female", "2019-2023", "total_deaths", 12).iterrows()]
s17b = [ssd_row(r) for _, r in ssd_rows("Male", "2019-2023", "total_deaths", 12).iterrows()]
s17c = []
for sex in ("Female", "Male"):
    for _, r in ssd_rows(sex, "2019-2023", "attributable_deaths", 8).iterrows():
        s17c.append([sex] + ssd_row(r, with_paf=True))
s17d = []
for sex in ("Female", "Male"):
    for outcome in ("total_deaths", "attributable_deaths"):
        s = ssd[(ssd.sex == sex) & (ssd.period == "2019-2023") & (ssd.outcome == outcome)]
        pe = float(s.population_effect.sum())
        re_ = float(s.rate_effect.sum())
        pa = float(s.paf_effect.sum()) if outcome == "attributable_deaths" else np.nan
        s17d.append([sex, "All cancer deaths" if outcome == "total_deaths"
                     else "Metabolic-attributable deaths",
                     float(s.value_0.sum()), float(s.value_1.sum()), float(s.change.sum()),
                     pe, re_, pa])
build_sheet(
    wb, "TableS17",
    "Table S17. Site \u00d7 sex decomposition of the 2019\u20132023 change in early-onset cancer deaths",
    [10, 38, 13, 13, 13, 15, 13, 13, 14, 13, 13],
    [
        block(["Site", "Deaths 2019", "Deaths 2023", "Change",
               "Population effect", "Rate effect", "Population share (%)", "Rate share (%)"],
              s17a, [None, "int", "int", "int_signed", "int_signed", "int_signed",
                     "pct1", "pct1"],
              label="Panel A. Women: 12 sites with the largest increase"),
        block(["Site", "Deaths 2019", "Deaths 2023", "Change",
               "Population effect", "Rate effect", "Population share (%)", "Rate share (%)"],
              s17b, [None, "int", "int", "int_signed", "int_signed", "int_signed",
                     "pct1", "pct1"],
              label="Panel B. Men: 12 sites with the largest increase"),
        block(["Sex", "Site", "Attributable 2019", "Attributable 2023", "Change",
               "Population effect", "Rate effect", "PAF effect", "Population share (%)",
               "Rate share (%)", "PAF share (%)"],
              s17c, [None, None, "int", "int", "int_signed", "int_signed", "int_signed",
                     "int_signed", "pct1", "pct1", "pct1"],
              label="Panel C. Metabolic-attributable deaths, 2019\u20132023: eight sites with the largest increase"),
        block(["Sex", "Outcome", "2019", "2023", "Change", "Population effect",
               "Rate effect", "PAF effect"],
              s17d, [None, None, "int", "int", "int_signed", "int_signed", "int_signed",
                     "int_signed"],
              label="Panel D. Totals across the 29 GBD cancer sites"),
    ],
    "Das Gupta symmetric decomposition of N = population \u00d7 death rate, and of attributable deaths = "
    "population \u00d7 death rate \u00d7 PAF, for 2019\u20132023. Population is the sex-specific global "
    "15\u201349 population; rates are counts divided by that population. Effects are expressed in deaths and "
    "shares are effects divided by the total change. Panels A and B use total deaths; Panel C uses deaths "
    "attributable to the GBD metabolic-risk aggregate (rei 104). Sex-specific pancreatic attributable deaths "
    "were obtained by splitting the both-sex total by the sex-specific share of total pancreatic deaths "
    "because the sex-specific GBD slice was unavailable; the split values sum exactly to the published "
    "both-sex total. Site rows cover the 29 GBD cancer sites and account for about 95% of all-cancer deaths; "
    "the remainder is residual \u2018other malignant neoplasms\u2019, so Panel D is not identical to the "
    "all-cancer total in Table 1. Source: results/site_sex_decomposition.csv; computed by "
    "scripts/site_sex_decomposition.py.",
)

# --------------------------------------------------------------------------- #
# Table S2
# --------------------------------------------------------------------------- #
ca = country_attr[country_attr.risk == "Metabolic risks"].copy()
# Rank on the metabolic-risk AGGREGATE only; the component risks (high BMI,
# high fasting plasma glucose) return their own PAFs and must not enter the
# country ranking (their PAFs are neither additive nor interchangeable).
assert ca.risk.nunique() == 1 and ca.location.nunique() == 204
ca["key"] = ca.measure + "|" + ca.metric
wide = ca.pivot_table(index=["location_id", "location"], columns="key", values="val").reset_index()
lo = ca.pivot_table(index=["location_id", "location"], columns="key", values="lower").reset_index()
hi = ca.pivot_table(index=["location_id", "location"], columns="key", values="upper").reset_index()
w = wide.merge(lo, on=["location_id", "location"], suffixes=("", "_lo")).merge(
    hi, on=["location_id", "location"], suffixes=("", "_hi"))
w = w.sort_values("Deaths|Percent", ascending=False)
def cnt(v, lo, hi) -> str:
    """Count with thousands separators; one decimal for very small counts."""
    if max(abs(v), abs(hi)) < 1000:
        return ui_rate(v, lo, hi, 1)
    return ui_int(v, lo, hi)


s2 = []
for _, r in w.iterrows():
    s2.append([
        r.location, int(r.location_id),
        cnt(r["Deaths|Number"], r["Deaths|Number_lo"], r["Deaths|Number_hi"]),
        paf(r["Deaths|Percent"] * 100, r["Deaths|Percent_lo"] * 100, r["Deaths|Percent_hi"] * 100),
        ui_rate(r["Deaths|Rate"], r["Deaths|Rate_lo"], r["Deaths|Rate_hi"], 3),
        cnt(r["DALYs|Number"], r["DALYs|Number_lo"], r["DALYs|Number_hi"]),
        paf(r["DALYs|Percent"] * 100, r["DALYs|Percent_lo"] * 100, r["DALYs|Percent_hi"] * 100),
    ])
build_sheet(
    wb, "TableS2",
    "Table S2. Country-level metabolic-attributable burden of early-onset cancer, 2023",
    [40, 11, 26, 22, 22, 28, 22],
    [block(["Location", "location_id", "Attributable deaths (95% UI)", "PAF deaths % (95% UI)",
            "Attributable death rate (95% UI)", "Attributable DALYs (95% UI)", "PAF DALYs % (95% UI)"],
           s2, [None, None, None, None, None, None, None],
           align=["left", "center", "center", "center", "center", "center", "center"])],
    "Rows are the 204 countries and territories for the metabolic-risk aggregate, sorted by descending PAF "
    "for deaths; the component risks high body-mass index and high fasting plasma glucose are not used for "
    "this ranking because their PAFs are neither additive nor interchangeable with the aggregate. PAF = "
    "population-attributable fraction (attributable burden divided by total cancer burden excluding "
    "non-melanoma skin cancer), in percent; attributable rates per 100,000 population aged 15\u201349; "
    "UI = GBD 95% uncertainty interval. Lower bounds are slightly negative in a few very-low-PAF settings "
    "(e.g. Burkina Faso, Niger) and are retained as produced by the comparative risk assessment; they are "
    "not interpretable as protective effects. The 204 country values sum to 47,259 attributable deaths "
    "against the global estimate of 47,303 (rounding). Source: results/attributable_country_2023.csv "
    "(risk == 'Metabolic risks').",
)

# --------------------------------------------------------------------------- #
# Table S3
# --------------------------------------------------------------------------- #
ic = ineq_country[ineq_country.sex == "both"].copy()
piv = ic.pivot_table(index="location_id", columns="outcome", values="y").reset_index()
meta = ic.groupby("location_id").agg(sdi=("sdi", "first"), pop=("pop", "first"),
                                     rank=("rank", "first")).reset_index()
w3 = meta.merge(piv, on="location_id").sort_values("sdi", ascending=False)
s3 = []
for _, r in w3.iterrows():
    s3.append([
        r.location_id, float(r.sdi), int(r["rank"]), r["pop"],
        float(r["Total Incidence (15-49 ASR)"]), float(r["Total Deaths (15-49 ASR)"]),
        float(r["Total DALYs (15-49 ASR)"]), float(r["Metabolic PAF (deaths, %)"]),
        float(r["Metabolic-attributable death rate"]),
    ])
build_sheet(
    wb, "TableS3",
    "Table S3. Country-level input data for the socio-demographic inequality analysis, 2023, both sexes",
    [11, 10, 10, 18, 16, 16, 18, 16, 20],
    [block(["location_id", "SDI", "SDI rank", "Population 2023", "Incidence ASR (per 100,000)",
            "Death ASR (per 100,000)", "DALY ASR (per 100,000)", "Metabolic PAF deaths (%)",
            "Metabolic-attributable death rate (per 100,000)"],
           s3, [None, "ci3", "int", "int", "rate1", "rate1", "rate1", "pct2", "rate2"],
           align=["center"] * 9)],
    "Inputs to the concentration index, slope index of inequality and relative index of inequality: 204 "
    "countries and territories ranked by the Socio-demographic Index (SDI, 0\u20131) and weighted by population "
    "aged 15\u201349. ASR = age-standardised rate within ages 15\u201349; PAF = population-attributable fraction "
    "for metabolic risks; rates per 100,000. Source: results/inequality_country_data.csv.",
)

# --------------------------------------------------------------------------- #
# Table S4
# --------------------------------------------------------------------------- #
aie = apcie.copy()
aie["rr_low"] = np.exp(aie.ci_low)
aie["rr_high"] = np.exp(aie.ci_high)
et_order = {"age": 0, "period": 1, "cohort": 2}
aie["_m"] = aie.measure.map({"Incidence": 0, "Deaths": 1})
aie["_e"] = aie.effect_type.map(et_order)
aie["_l"] = aie.level.astype(str).str.split("-").str[0].astype(int)
aie = aie.sort_values(["_m", "_e", "_l"])
s4 = [[r.location, r.measure, r.effect_type.capitalize(), r.level, float(r.estimate), float(r.rate_ratio),
       float(r.rr_low), float(r.rr_high), int(r.n_boot)] for _, r in aie.iterrows()]
build_sheet(
    wb, "TableS4",
    "Table S4. Age\u2013period\u2013cohort effects on early-onset cancer incidence and mortality",
    [10, 12, 12, 12, 14, 14, 12, 12, 16],
    [block(["Location", "Measure", "Effect type", "Level", "Estimate (log scale)", "Rate ratio",
            "RR lower (exp CI)", "RR upper (exp CI)", "Bootstrap replicates"],
           s4, [None, None, None, None, "ci3", "ci3", "ci3", "ci3", "int"])],
    "Intrinsic estimator age\u2013period\u2013cohort model fitted to 5-year age bands (15\u201319 to 45\u201349), "
    "5-year periods (1990\u20131994 to 2020\u20132023) and 5-year birth cohorts; estimates are on the log "
    "scale with the reference level set by the estimator's parameterisation, and rate ratios are exp(estimate). "
    "The 95% intervals are bootstrap percentile intervals (500 replicates) on the log scale, exponentiated for "
    "the RR bounds. Age\u2013period\u2013cohort effects are not identifiable without constraints, so absolute "
    "levels depend on that choice; trends and monotonicity are the interpretable features. Source: "
    "results/apc_ie_effects.csv.",
)

# --------------------------------------------------------------------------- #
# Table S5
# --------------------------------------------------------------------------- #
se = site_eapc.copy()
se["_m"] = se.measure.map({"Incidence": 0, "Deaths": 1, "DALYs": 2})
se = se.sort_values(["_m", "eapc"], ascending=[True, False])
s5 = [[r.cause, r.measure, float(r.eapc), float(r.ci_low), float(r.ci_high), float(r.pvalue)]
      for _, r in se.iterrows()]
build_sheet(
    wb, "TableS5",
    "Table S5. Site-specific estimated annual percentage change in rates, 1990\u20132023",
    [42, 12, 10, 12, 12, 12],
    [block(["Cancer site", "Measure", "EAPC", "95% CI lower", "95% CI upper", "p value"],
           s5, [None, None, "eapc", "eapc", "eapc", "p"],
           align=["left", "center", "center", "center", "center", "center"])],
    "EAPC = estimated annual percentage change from a log-linear regression of the annual site-specific "
    "age-standardised rate on calendar year (29 sites \u00d7 3 measures); a negative EAPC indicates a "
    "declining rate. Rows are sorted by measure and descending EAPC. Source: results/sites_eapc.csv.",
)

# --------------------------------------------------------------------------- #
# Table S6
# --------------------------------------------------------------------------- #
sg = segs.copy()
sg["_m"] = sg.measure.map({"Incidence": 0, "Deaths": 1, "DALYs": 2})
sg["_s"] = sg.sex.map({"Both": 0, "Male": 1, "Female": 2})
sg = sg.sort_values(["_m", "_s", "segment"])
s6a = [[r.measure, SEXMAP[r.sex], int(r.n_joinpoints), yrs(r.joinpoint_years),
        ci(r.aapc, r.aapc_ci_low, r.aapc_ci_high), int(r.segment),
        f"{int(r.year_start)}\u2013{int(r.year_end)}", ci(r.apc, r.apc_ci_low, r.apc_ci_high)]
       for _, r in sg.iterrows()]

sj = jsens.copy()
sj["_m"] = sj.measure.map({"Incidence": 0, "Deaths": 1, "DALYs": 2})
sj["_s"] = sj.sex.map({"Both": 0, "Male": 1, "Female": 2})
sj = sj.sort_values(["_m", "_s", "max_k"])
s6b = [[r.measure, SEXMAP[r.sex], int(r.max_k), int(r.selected_n_joinpoints), yrs(r.joinpoint_years),
        float(r.bic), round(float(r.aapc), 3), round(float(r.last_segment_apc), 3),
        f"{int(r.last_segment_year_start)}\u2013{int(r.last_segment_year_end)}"]
       for _, r in sj.iterrows()]
build_sheet(
    wb, "TableS6",
    "Table S6. Joinpoint segments and sensitivity of the joinpoint selection for age-standardised rates",
    [12, 12, 12, 22, 24, 10, 14, 24, 14],
    [
        block(["Measure", "Sex", "No. of joinpoints", "Joinpoint years", "AAPC (95% CI)", "Segment",
               "Segment years", "APC (95% CI)"],
              s6a, [None, None, "int", None, None, "int", None, None],
              label="Panel A. Selected joinpoint model, 1990\u20132023"),
        block(["Measure", "Sex", "Maximum k allowed", "Selected k", "Selected joinpoint years", "BIC",
               "AAPC", "Last-segment APC", "Last segment"],
              s6b, [None, None, "int", "int", None, "num2", "eapc", "eapc", None],
              label="Panel B. Sensitivity of the BIC selection to the maximum number of joinpoints"),
    ],
    "AAPC = average annual percentage change; APC = annual percentage change within a segment; CI = 95% "
    "confidence interval; BIC = Bayesian information criterion. In Panel B, BIC selected the maximum permitted "
    "number of joinpoints (k = max_k) for every one of the 12 series at max_k 1\u20134, so the number and exact "
    "location of joinpoints are not robustly identified; the years should be interpreted as approximate "
    "(\u2018around 2013\u2019, \u2018around 2019\u2019) and not over-interpreted. Source: "
    "results/asr_joinpoint_segments.csv, results/joinpoint_sensitivity.csv.",
)

# --------------------------------------------------------------------------- #
# Table S7
# --------------------------------------------------------------------------- #
ab = ageb.copy()
ab["_m"] = ab.measure.map({"Incidence": 0, "Deaths": 1, "DALYs": 2})
ab["_b"] = ab.age_band.map({"15-29": 0, "30-39": 1, "40-49": 2, "15-49 total": 3})
ab = ab.sort_values(["_m", "_b"])
s7 = [[r.measure, r.age_band.replace("-", "\u2013") if r.age_band != "15-49 total" else "15\u201349 (total)",
       int(r.population_1990), int(r.population_2023), int(r.count_1990), int(r.count_2023),
       float(r.count_change_percent), float(r.rate_1990), float(r.rate_2023), float(r.rate_change_percent),
       ci(r.eapc_1990_2023, r.eapc_1990_2023_ci_low, r.eapc_1990_2023_ci_high),
       ci(r.eapc_2013_2023, r.eapc_2013_2023_ci_low, r.eapc_2013_2023_ci_high)]
      for _, r in ab.iterrows()]
build_sheet(
    wb, "TableS7",
    "Table S7. Burden of early-onset cancer by age band, 1990 and 2023",
    [12, 14, 18, 18, 15, 15, 14, 12, 12, 14, 24, 24],
    [block(["Measure", "Age band", "Population 1990", "Population 2023", "Count 1990", "Count 2023",
            "Change in count (%)", "Rate 1990 (per 100,000)", "Rate 2023 (per 100,000)",
            "Change in rate (%)", "EAPC 1990\u20132023 (95% CI)", "EAPC 2013\u20132023 (95% CI)"],
           s7, [None, None, "int", "int", "int", "int", "pct1", "rate1", "rate1", "pct1", None, None],
           align=["center"] * 12)],
    "Counts are numbers of cases, deaths or DALYs in the age band; rates are crude (not age-standardised) per "
    "100,000 population in the same age band; the 15\u201349 total row reproduces the all-ages-15\u201349 "
    "totals of Table 1. EAPC = estimated annual percentage change from a log-linear regression of the annual "
    "crude rate; CI = 95% confidence interval. No uncertainty intervals are produced for age-band counts. "
    "Source: results/age_bands_global.csv.",
)

# --------------------------------------------------------------------------- #
# Table S8
# --------------------------------------------------------------------------- #
# Regional rates are the directly age-standardised series produced by
# scripts/location_asr.py (2023 global 15-49 weights), not the GBD crude rates.
R23 = {m: region_asr23[region_asr23.measure == m].set_index("region") for m in MEAS}
RTR = {m: region_asr_tr[region_asr_tr.measure == m].set_index("region") for m in MEAS}
OLD = regions.set_index("location")
assert len(R23["Incidence"]) == 21

s8a, s8b = [], []
region_order = R23["Incidence"].sort_values("asr", ascending=False).index.tolist()
for reg_name in region_order + ["Global"]:
    if reg_name == "Global":
        g23 = {m: gasr[(gasr.year == 2023) & (gasr.measure == m)].iloc[0] for m in MEAS}
        g90 = {m: gasr[(gasr.year == 1990) & (gasr.measure == m)].iloc[0] for m in MEAS}
        at = {m: atrend[atrend.measure == m].iloc[0] for m in MEAS}
        pop = float(g23["Incidence"]["population"])
        counts = {m: float(g23[m]["count"]) for m in MEAS}
        row_a = ["Global"]
        for m in MEAS:
            row_a += [float(g90[m]["asr_15_49"]), float(g23[m]["asr_15_49"]),
                      (float(g23[m]["asr_15_49"]) / float(g90[m]["asr_15_49"]) - 1) * 100,
                      float(at[m].eapc_1990_2023), float(at[m].eapc_2013_2023)]
    else:
        pop = float(R23["Incidence"].loc[reg_name, "population"])
        counts = {m: float(R23[m].loc[reg_name, "count"]) for m in MEAS}
        row_a = [reg_name]
        for m in MEAS:
            t = RTR[m].loc[reg_name]
            row_a += [float(t.asr_1990), float(t.asr_2023), float(t.asr_change_percent),
                      float(t.eapc_1990_2023), float(t.eapc_2013_2023)]
    meta = OLD.loc[reg_name]
    s8a.append(row_a)
    s8b.append([row_a[0], pop, counts["Incidence"], counts["Deaths"], counts["DALYs"],
                float(meta.paf_deaths_2023), float(meta.attributable_deaths_number_2023),
                float(meta.attributable_deaths_rate_2023)])
build_sheet(
    wb, "TableS8",
    "Table S8. Regional burden of early-onset cancer, 1990 and 2023, directly age-standardised within 15\u201349",
    [34, 15, 14, 12, 12, 11, 14, 14, 11, 11, 11, 13, 13, 11, 11, 11],
    [
        block(["GBD region",
               "Incidence ASR 1990", "Incidence ASR 2023", "Incidence change (%)",
               "Incidence EAPC 1990\u20132023", "Incidence EAPC 2013\u20132023",
               "Death ASR 1990", "Death ASR 2023", "Death change (%)",
               "Death EAPC 1990\u20132023", "Death EAPC 2013\u20132023",
               "DALY ASR 1990", "DALY ASR 2023", "DALY change (%)",
               "DALY EAPC 1990\u20132023", "DALY EAPC 2013\u20132023"],
              s8a, [None] + ["rate1", "rate1", "pct1", "eapc", "eapc"] * 3,
              align=["left"] + ["center"] * 15,
              label="Panel A. Age-standardised rates and trends by region"),
        block(["GBD region", "Population 2023", "Incident cases 2023", "Deaths 2023", "DALYs 2023",
               "PAF deaths 2023 (%)", "Attributable deaths 2023", "Attributable death rate 2023"],
              s8b, [None, "int", "int", "int", "int", "pct2", "int", "rate2"],
              align=["left"] + ["center"] * 7,
              label="Panel B. Population, counts and metabolic attribution, 2023"),
    ],
    "The 21 GBD regions are sorted by descending 2023 incidence ASR, with the global total last, in both "
    "panels. Panel A: ASR = directly age-standardised rate within ages 15\u201349 to the 2023 global 15\u201349 "
    "population (per 100,000), computed from the national age-specific series and therefore NOT the GBD crude "
    "rate; change = relative change 1990\u20132023; EAPC = estimated annual percentage change in the ASR. "
    "Panel B: population aged 15\u201349 and counts in 2023; PAF = population-attributable fraction for "
    "metabolic risks (percent); attributable deaths are deaths attributable to high body-mass index and high "
    "fasting plasma glucose; the attributable death rate is per 100,000. Direct standardisation reverses the "
    "crude-rate ranking: the highest crude incidence rates are in Western Europe (137.3), Eastern Sub-Saharan "
    "Africa (123.8) and Australasia (123.4), but Eastern and Central Sub-Saharan Africa have the highest ASR, "
    "because their populations aged 15\u201349 are younger. Source: results/regions_asr_2023.csv, "
    "results/regions_asr_trends.csv, results/regions_2023.csv, results/global_asr_series.csv, "
    "results/asr_trends.csv.",
)

# --------------------------------------------------------------------------- #
# Table S9
# --------------------------------------------------------------------------- #
bt = backtest.sort_values(["measure", "mape"])
s9a = [[r.measure, r.model, r.model_detail, f"{int(r.train_start)}\u2013{int(r.train_end)}",
        f"{int(r.test_start)}\u2013{int(r.test_end)}", int(r.n_test), float(r.mae), float(r.mape),
        float(r.mean_bias), float(r.rmse), float(r.coverage_95) * 100] for _, r in bt.iterrows()]
dg = diagn.sort_values(["location", "measure"])
s9b = [[r.location, r.measure, r.model, str(r.order), int(r.n_obs), float(r.aic),
        float(r.ljung_box_p_lag5), float(r.ljung_box_p_lag10)] for _, r in dg.iterrows()]
SCEN_LABEL = {
    "arima_asr_damped_pop": "ARIMA on ASR, damped population",
    "arima_asr_undamped_pop": "ARIMA on ASR, undamped population",
    "constant_2023_asr_damped_pop": "Constant 2023 ASR, damped population",
}
g23 = gasr[gasr.year == 2023].set_index("measure")
sc = asr_scen.copy()
sc["_m"] = sc.measure.map({"Incidence": 0, "Deaths": 1, "DALYs": 2})
sc["_s"] = sc.scenario.map({"arima_asr_damped_pop": 0, "arima_asr_undamped_pop": 1,
                            "constant_2023_asr_damped_pop": 2})
sc = sc.sort_values(["_m", "_s"])
s9c = [["Global", r.measure, SCEN_LABEL[r.scenario],
        float(g23.loc[r.measure, "asr_15_49"]),
        float(r.asr_2050),
        dash(None if pd.isna(r.asr_2050_low) else float(r.asr_2050_low)),
        dash(None if pd.isna(r.asr_2050_high) else float(r.asr_2050_high)),
        float(r.number_2050),
        dash(None if pd.isna(r.number_2050_low) else float(r.number_2050_low)),
        dash(None if pd.isna(r.number_2050_high) else float(r.number_2050_high)),
        float(r.rate_change_percent)] for _, r in sc.iterrows()]
build_sheet(
    wb, "TableS9",
    "Table S9. Projection validation, model diagnostics and 2050 scenarios",
    [14, 14, 26, 22, 16, 12, 11, 11, 12, 11, 12],
    [
        block(["Measure", "Model", "Model detail", "Training period", "Test period", "n test years", "MAE",
               "MAPE (%)", "Mean bias", "RMSE", "95% PI coverage (%)"],
              s9a, [None, None, None, None, None, "int", "num2", "num2", "num2", "num2", "rate1"],
              label="Panel A. Backtest: models trained on 1990\u20132012 and evaluated on 2013\u20132023"),
        block(["Location", "Measure", "Model", "Order", "n observations", "AIC", "Ljung\u2013Box p (lag 5)",
               "Ljung\u2013Box p (lag 10)"],
              s9b, [None, None, None, None, "int", "num2", "p", "p"],
              label="Panel B. ARIMA diagnostics for the rate series used in the projection validation"),
        block(["Location", "Outcome", "Scenario", "ASR 2023", "ASR 2050", "ASR 2050 lower",
               "ASR 2050 upper", "Number 2050", "Number 2050 lower", "Number 2050 upper",
               "Change in ASR 2023\u20132050 (%)"],
              s9c, [None, None, None, "rate1", "rate1", "rate1", "rate1", "int", "int", "int", "pct1"],
              label="Panel C. 2050 scenarios from the ASR-based ARIMA projections (global)"),
    ],
    "Panel A: one-step backtest of the three projection models; MAPE = mean absolute percentage error, mean "
    "bias = mean of (observed \u2212 predicted), coverage = share of observations inside the 95% prediction "
    "interval. ARIMA did not outperform the constant-rate or log-linear benchmarks and underpredicted the "
    "2013\u20132023 rise, which is why projections are presented as scenarios rather than forecasts. Panel B: "
    "AIC = Akaike information criterion; Ljung\u2013Box p values test residual autocorrelation (p > 0.05 "
    "indicates no evidence of remaining autocorrelation) for the crude-rate series used in the validation. "
    "Panel C: the scenarios come from the directly age-standardised ARIMA refit and differ in the assumed "
    "population trajectory (damped vs undamped) and in whether the ASR is held at its 2023 value; the ASR "
    "models are ARIMA(1,1,0) for incidence and ARIMA(2,1,0) for deaths and DALYs. Source: "
    "results/projection_backtest.csv, results/projection_diagnostics.csv, "
    "results/projections_asr_scenarios_2050.csv.",
)

# --------------------------------------------------------------------------- #
# Table S10
# --------------------------------------------------------------------------- #
sq = sdiq.copy()
sq["_q"] = sq.sdi_quintile.map({"Q1 lowest": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Q5 highest": 4})
sq["_m"] = sq.measure.map({"Incidence": 0, "Deaths": 1, "DALYs": 2})
sq = sq.sort_values(["_q", "_m"])
s10 = [[r.sdi_quintile, r.measure, float(r.asr_1990), float(r.asr_2023),
        float(r.asr_change_percent),
        ci(r.eapc_1990_2023, r.eapc_1990_2023_low, r.eapc_1990_2023_high),
        float(r.eapc_1990_2023_p),
        ci(r.eapc_2013_2023, r.eapc_2013_2023_low, r.eapc_2013_2023_high),
        float(r.eapc_2013_2023_p)] for _, r in sq.iterrows()]
build_sheet(
    wb, "TableS10",
    "Table S10. Burden of early-onset cancer by Socio-demographic Index quintile, 1990 and 2023, "
    "directly age-standardised within 15\u201349",
    [14, 12, 14, 14, 14, 28, 12, 28, 12],
    [block(["SDI quintile", "Measure", "ASR 1990", "ASR 2023", "Change in ASR (%)",
            "EAPC 1990\u20132023 (95% CI)", "p", "EAPC 2013\u20132023 (95% CI)", "p"],
           s10, [None, None, "rate1", "rate1", "pct1", None, "p", None, "p"],
           align=["left", "center"] + ["center"] * 7)],
    "Quintiles use the fixed 2023 SDI assignment for all years (Q1 = lowest, Q5 = highest). Rates are directly "
    "age-standardised within ages 15\u201349 to the 2023 global 15\u201349 population, so the 2023 ASR equals "
    "the 2023 crude rate by construction. EAPC = estimated annual percentage change from a log-linear "
    "regression of the annual ASR on calendar year; CI = 95% confidence interval; p = p value for the EAPC. "
    "Source: results/sdi_quintile_asr_trends.csv.",
)

# --------------------------------------------------------------------------- #
# README (created first in the final workbook)
# --------------------------------------------------------------------------- #
readme_rows = [
    ("Purpose", "Tables and supplementary tables for the manuscript 'Global, regional, and national burden of "
                "early-onset cancer and its metabolic-risk-attributable fraction, 1990-2023: a systematic "
                "analysis for the Global Burden of Disease Study 2023', prepared for submission to "
                "eClinicalMedicine. This workbook contains Tables 1\u20135 and Supplementary Tables S1\u2013S13."),
    ("Data source", "Global Burden of Disease Study 2023 (GBD 2023), release 8352; cause 1059 (all cancers "
                    "combined, including non-melanoma skin cancer as a separate child cause); age range 15-49 "
                    "years; both sexes and sex-specific estimates; 204 countries and territories, 21 GBD "
                    "regions and 5 SDI quintiles."),
    ("Population and rates", "All rates are per 100,000 population aged 15-49 unless stated otherwise. Counts "
                             "are numbers of incident cases, deaths or DALYs."),
    ("ASR", "Age-standardised rate, directly standardised within ages 15-49 to the 2023 global 15-49 "
            "population; by construction the 2023 ASR equals the 2023 crude rate."),
    ("Country and regional estimates", "Country-level and regional rates and trends (Tables S1 and S8) are "
                                       "computed by direct age-standardisation within ages 15-49 to the 2023 "
                                       "global 15-49 population (scripts/location_asr.py); the GBD crude rates "
                                       "are not used for these estimates, and the crude-rate country and "
                                       "regional trend files are superseded. SDI quintiles use the fixed 2023 "
                                       "quintile assignment for all years."),
    ("PAF", "Population-attributable fraction: attributable burden divided by total cancer burden excluding "
            "non-melanoma skin cancer (GBD comparative risk assessment for high body-mass index and high "
            "fasting plasma glucose)."),
    ("EAPC", "Estimated annual percentage change from a log-linear regression of the annual rate on calendar "
             "year; AAPC is the average annual percentage change over the whole period from the joinpoint "
             "model and APC the annual percentage change within a joinpoint segment."),
    ("RII and SII", "Relative index of inequality (ratio of the fitted burden at the lowest vs the highest "
                    "end of the population-weighted SDI ranking, from a log-link model) and slope index of "
                    "inequality (absolute counterpart); the concentration index is its relative-rank analogue, "
                    "with a positive value indicating a pro-rich distribution."),
    ("UI and PI", "UI = GBD 95% uncertainty interval (sampling and model uncertainty of the GBD estimation "
                  "process); PI = 95% prediction interval of the projection model."),
    ("Display conventions", "Counts are shown with thousands separators; rates to 1 decimal (2-3 decimals for "
                            "small attributable rates); EAPC, APCs and their confidence intervals to 2 "
                            "decimals; PAF to 2 decimals; p and q values < 0.001 in scientific notation. "
                            "'-' (em dash) marks a value that is not defined or not estimated."),
    ("Table1", "Global counts, crude rates and ASRs in 1990 and 2023 with relative changes, by sex and outcome."),
    ("Table2", "EAPC, joinpoint years, AAPC and last-segment APC for the ASR, 1990-2023 and 2013-2023."),
    ("Table3", "Metabolic-risk-attributable deaths and DALYs, PAF and attributable rates in 1990 and 2023, "
               "overall and by sex."),
    ("Table4", "Socioeconomic inequality: concentration index, SII and RII across 204 countries in 2023, and "
               "trends in inequality 1990-2023."),
    ("Table5", "Decomposition of changes in the early-onset cancer burden 1990-2023: Das Gupta decomposition "
               "of changes in death counts and Kitagawa decomposition of the change in the crude rate."),
    ("TableS1", "Country-level trends in directly age-standardised rates for 1990-2023 and 2013-2023, with ASR "
                "endpoints, EAPC, p values, FDR q values and significance flags."),
    ("TableS2", "Country-level metabolic-attributable deaths, DALYs, PAF and attributable death rate, 2023, "
                "ranked on the metabolic-risk aggregate."),
    ("TableS3", "Country-level SDI, population and burden inputs used in the inequality analysis, 2023."),
    ("TableS4", "Age-period-cohort intrinsic-estimator effects on incidence and mortality."),
    ("TableS5", "Site-specific EAPC for incidence, mortality and DALY rates, 1990-2023."),
    ("TableS6", "Joinpoint segments for the ASR and sensitivity of the BIC selection to the maximum number of "
                "joinpoints."),
    ("TableS7", "Burden by age band (15-29, 30-39, 40-49) in 1990 and 2023 with EAPC."),
    ("TableS8", "Regional ASRs in 1990 and 2023 with change and EAPC, plus population, 2023 counts and "
                "metabolic attribution, for the 21 GBD regions."),
    ("TableS9", "Projection backtest performance, ARIMA diagnostics and the 2050 ASR-based scenarios."),
    ("TableS10", "SDI-quintile ASRs in 1990 and 2023 with relative change and EAPCs, directly age-standardised."),
    ("TableS11", "Site-specific incidence, deaths, DALYs, incidence EAPC and metabolic attribution for the 29 "
                 "cancer sites in 2023 (main-text display item before the display-item reduction)."),
    ("TableS12", "Projections of directly age-standardised rates, counts and attributable burden to 2050 under "
                 "alternative scenarios (ARIMA refitted to the ASR series)."),
    ("TableS13", "Sensitivity analyses: 1990 instead of 2023 standard population (levels, trends, country and "
                 "regional rankings, SDI extremes) and ages 15-39 instead of 15-49 (counts, ASRs, trends, sex "
                 "differences and metabolic PAF)."),
    ("TableS14", "Serial-correlation-robust (Newey-West HAC) standard errors for all 82 trend series, lag "
                 "sensitivity, and the four site-specific series whose significance changes."),
    ("TableS15", "Detection-effect sensitivity: incidence after removing thyroid, prostate, kidney and melanoma "
                 "cancers (and breast cancer), with scenario-level trends and site contributions."),
    ("TableS16", "External benchmark against GLOBOCAN 2024 at ages 15-49: global and sex-specific levels, 28 "
                 "sites, country and regional rank concordance, and the site decomposition of the level gap."),
    ("TableS17", "Site x sex Das Gupta decomposition of the 2019-2023 change in total and metabolic-attributable "
                 "deaths, with the female breast and cervical cancer contribution."),
    ("Software", "Python 3.11 (pandas, numpy, statsmodels, openpyxl); tables built by "
                 "scripts/build_tables.py from the CSV files in results/."),
    ("Date", "2026-11-08."),
    ("Note", "All values derive from public GBD 2023 estimates; no new primary data were collected. Point "
             "estimates are reproducible from the CSV files in results/, which retain full precision - "
             "displayed values are rounded and may not sum exactly. Site-level values do not sum to the "
             "all-cancer total (non-melanoma skin cancer is excluded from the total); the 14 metabolically "
             "attributable sites do sum to the global attributable total."),
]
ws = wb.create_sheet("README", 0)
ws.cell(1, 1, "README. Main tables and supplementary tables - early-onset cancer (ages 15-49), GBD 2023")
ws.cell(1, 1).font = TITLE_FONT
ws.cell(1, 1).alignment = Alignment(horizontal="left", vertical="center")
ws.merge_cells("A1:B1")
ws.row_dimensions[1].height = 22
for j, h in enumerate(["Item", "Detail"], 1):
    c = ws.cell(2, j, h)
    c.font = WHITE_BOLD
    c.fill = DARK
    c.alignment = CENTER
r = 3
for item, detail in readme_rows:
    c1 = ws.cell(r, 1, item)
    c1.font = Font(bold=True, size=10)
    c1.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    c2 = ws.cell(r, 2, detail)
    c2.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    r += 1
ws.column_dimensions["A"].width = 26
ws.column_dimensions["B"].width = 118
ws.freeze_panes = "A3"
ws.auto_filter.ref = f"A2:B{r - 1}"
c = ws.cell(r + 1, 1, "Footnote: all values derive from public GBD 2023 estimates (release 8352, cause 1059, "
                       "ages 15-49); no primary data were collected. Displayed values are rounded from the "
                       "CSV files in results/, which retain full precision, so columns and panels may not sum "
                       "exactly. This workbook accompanies the manuscript text; see manuscript/table_legends.md "
                       "for the legend of every table.")
c.font = FOOT_FONT
c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
ws.merge_cells(f"A{r + 1}:B{r + 1}")
ws.row_dimensions[r + 1].height = 40

# --------------------------------------------------------------------------- #
# sheet order: README, main Tables 1-5, Supplementary Tables S1-S17
# (S11 = site-specific, S12 = projections, S13 = sensitivity analyses,
#  S14-S17 = HAC inference, detection-excluded sites, GLOBOCAN benchmark and
#  site x sex decomposition)
# --------------------------------------------------------------------------- #
SHEET_ORDER = (["README", "Table1", "Table2", "Table3", "Table4", "Table5"]
               + [f"TableS{i}" for i in range(1, 18)])
assert len(SHEET_ORDER) == 23 and len(set(SHEET_ORDER)) == 23
for name in SHEET_ORDER:
    assert name in wb.sheetnames, name
# superseded sheet keys must be gone (old Table4/Table5/Table6/Table7 numbering)
for old_name in ("Table6", "Table7"):
    assert old_name not in wb.sheetnames, old_name
assert set(wb.sheetnames) == set(SHEET_ORDER), sorted(set(wb.sheetnames) ^ set(SHEET_ORDER))
assert wb["Table4"]["A1"].value.startswith("Table 4.")
assert wb["Table5"]["A1"].value.startswith("Table 5.")
assert wb["TableS11"]["A1"].value.startswith("Table S11.")
assert wb["TableS12"]["A1"].value.startswith("Table S12.")
assert wb["TableS13"]["A1"].value.startswith("Table S13.")
assert wb["TableS14"]["A1"].value.startswith("Table S14.")
assert wb["TableS15"]["A1"].value.startswith("Table S15.")
assert wb["TableS16"]["A1"].value.startswith("Table S16.")
assert wb["TableS17"]["A1"].value.startswith("Table S17.")
# openpyxl writes the tabs in wb._sheets order, so reorder before saving
wb._sheets = [wb[name] for name in SHEET_ORDER]
assert wb.sheetnames == SHEET_ORDER, wb.sheetnames

OUT.parent.mkdir(parents=True, exist_ok=True)
wb.save(OUT)
print("saved", OUT, "sheets:", wb.sheetnames)
