# Analysis code and derived results — early-onset cancer and metabolic risks, GBD 2023

**Manuscript:** Global, regional, and national burden of early-onset cancer and its metabolic-risk-attributable fraction, 1990–2023: a systematic analysis for the Global Burden of Disease Study 2023

**Authors:** Haohao Xu, Shishi Zhou, Jianfei Fu (corresponding author, 11218276@zju.edu.cn)

**Version:** 1.0.0 (9 November 2026)

This repository contains the analysis code, the exact GBD Results Tool query definitions, and the derived result tables for the manuscript. It does **not** contain the raw GBD or GLOBOCAN extracts, which are not redistributed (see *Data sources and terms*).

## Contents

| Path | Contents |
|---|---|
| `code/` | Python analysis scripts (Python 3.11). `run_all.py` orchestrates the pipeline; individual scripts can be run in the order below. |
| `data/queries.json` | The 38 GBD Results Tool query definitions (context, population group, measure, metric, location, age, sex, cause and risk arrays, `version = 8352`). |
| `data/id_map.json` | Cause, location and risk identifier mappings. |
| `results/` | Derived CSV and JSON outputs, including all tables and figures in the manuscript and supplement. |

## Data sources and terms

- **GBD 2023 (release version 8352)** was accessed through the GBD Results Tool (<https://vizhub.healthdata.org/gbd-results/>) and the Global Health Data Exchange (<https://ghdx.healthdata.org/>) on 8 November 2026. GBD data are made available by the Institute for Health Metrics and Evaluation (IHME) for non-commercial use under the IHME Free-of-Charge Non-Commercial User Agreement and the IHME Terms and Conditions (<https://www.healthdata.org/terms-details>). Re-running `data/queries.json` against release 8352 reproduces the inputs; the raw responses (~167 MB) are not included here.
- **GLOBOCAN 2024** was accessed through the IARC Global Cancer Observatory (<https://gco.iarc.fr/>; API `https://gco-api.iarc.fr/api/globocan/v3/2024/`) on 9 November 2026. The age-group token for 15–49 is `3_9` (the earlier `4_10` token returns ages 20–54). Cite Sung H, Filho AM, Laversanne M, Ferlay J, Siegel RL, Soerjomataram I, et al. Global cancer statistics 2024. CA Cancer J Clin. 2026;76(4):e70090. doi:10.3322/caac.70090.
- **SDI 1950–2023** (doi:10.6069/BCX2-VV69) and **GBD 2023 Cancer Forecasts 2024–2050** (doi:10.6069/QZN6-GM87) were obtained from GHDx.
- No individual-level or restricted data were used. Any reuse of the underlying estimates must comply with the IHME and IARC terms and cite the source publications.

## Reproducing the analysis

1. Create a Python 3.11 environment and install `requirements.txt` (for example, `uv venv --python 3.11 && uv pip install -r requirements.txt`).
2. Download the GBD slices described by `data/queries.json` (version 8352) into `data/raw/`, keeping the file names used by `load_data.py`. GLOBOCAN data are fetched by `code/fetch_globocan_2024.py`.
3. Run the scripts from the repository root. The pipeline order is:

```
load_data.py → trends.py → joinpoint_segments.py → apc_ie.py → attributable.py →
fix_paf.py → decomposition.py → location_asr.py → sex_asr.py → inequality.py →
projection.py → projection_asr_series.py → projection_validation.py → sites.py →
site_paf.py → sex_interaction.py → fdr_country.py → sensitivity_asr.py →
detection_excluded.py → hac_robustness.py → fetch_globocan_2024.py →
globocan_benchmark.py → site_sex_decomposition.py → tables.py / build_tables.py →
figures.py / figures_revision.py → summarize.py
```

`run_all.py` runs the core pipeline. Paths assume the project layout `data/raw/` and `results/`; adjust the constants at the top of the scripts if you use a different layout.

## Licensing

- **Code** (`code/`, query definitions): MIT License — see `LICENSE`.
- **Derived result tables** (`results/`): Creative Commons Attribution 4.0 International (CC BY 4.0) — see `DATA_LICENSE.md`.
- GBD and GLOBOCAN estimates remain subject to the IHME and IARC terms; the licenses above do not override them.

## Citation

Cite the article (DOI to be assigned) and this repository. A machine-readable citation is provided in `CITATION.cff`; the repository DOI will be inserted here and in the manuscript once minted.

## Contact

Jianfei Fu, Department of Medical Oncology, Jinhua Hospital, Zhejiang University School of Medicine, Jinhua, Zhejiang, China. Email: 11218276@zju.edu.cn. Telephone: +86 579 82552733.
