#!/usr/bin/env python3
"""Run the full GBD 2023 early-onset cancer pipeline in dependency order.

Inputs are the GBD 2023 panels downloaded by ``load_data.py`` (release 8352,
public but authenticated). Every later step consumes derived CSVs in
``results/``; tables and figures are rebuilt from those CSVs, and the three
submission DOCX files are rebuilt last.

Usage:
    .venv/bin/python scripts/run_all.py            # full pipeline
    .venv/bin/python scripts/run_all.py --no-docx  # stop after figures

The pipeline stops at the first failing step so that inconsistent downstream
outputs are never produced silently.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# Dependency order (verified against each script's read_csv/to_csv calls):
#   panels -> PAF correction -> trends -> attribution/sites/decomposition
#   -> inequality -> ASR trends by location/sex -> APC -> projections
#   -> projection reconciliation -> tables -> figures -> DOCX.
STEPS = [
    # data preparation
    "load_data.py",
    "fix_paf.py",
    # descriptive trends and joinpoints
    "trends.py",
    "fdr_country.py",
    "joinpoint_segments.py",
    # attribution, sites, decomposition
    "attributable.py",
    "attributable_sex.py",
    "site_paf.py",
    "sites.py",
    "decomposition.py",
    # inequality and age-standardised location trends
    "inequality.py",
    "age_bands.py",
    "location_asr.py",
    "regions.py",
    "sex_asr.py",
    "apc_ie.py",
    "sex_interaction.py",
    # reviewer-preemption sensitivity analyses (1990 standard; ages 15-39)
    "sensitivity_asr.py",
    # projections (projection.py must precede asr_trends.py)
    "projection.py",
    "asr_trends.py",
    "projection_asr_series.py",
    "projection_asr.py",
    "projection_validation.py",
    "attributable_projection_fix.py",
    "asr_joinpoint_segments.py",
    # summary, tables, figures, documents
    "summarize.py",
    "build_tables.py",
    "figures_revision.py",
    "build_docx.py",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-docx", action="store_true",
                    help="stop before build_docx.py")
    args = ap.parse_args()
    steps = [s for s in STEPS if not (args.no_docx and s == "build_docx.py")]

    t0 = time.time()
    for i, step in enumerate(steps, 1):
        print(f"\n===== [{i}/{len(steps)}] {step} =====", flush=True)
        started = time.time()
        r = subprocess.run([PY, os.path.join(ROOT, "scripts", step)], cwd=ROOT)
        if r.returncode != 0:
            print(f"!! {step} failed with exit code {r.returncode}; stopping.",
                  flush=True)
            return r.returncode
        print(f"     done in {time.time() - started:.1f}s", flush=True)

    print(f"\nPipeline complete in {(time.time() - t0) / 60:.1f} min.")
    print("Outputs: results/*.csv, manuscript/tables.xlsx, figures/*.png, "
          "manuscript/*.docx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
