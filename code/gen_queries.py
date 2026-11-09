#!/usr/bin/env python3
"""Generate GBD 2023 query plan for Direction A (early-onset cancer x metabolic risks).
Reads /tmp/gbd_metadata.json + /tmp/gbd_hierarchy.json (downloaded from vizhub).
Outputs data/id_map.json and data/queries.json. No credentials involved.
"""
import json, itertools, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = json.load(open('/tmp/gbd_metadata.json'))['data']
h = json.load(open('/tmp/gbd_hierarchy.json'))['data']

lnames = {int(k): v.get('name', '?') for k, v in m['location'].items() if k.isdigit()}
cnames = {int(k): v['name'] for k, v in m['cause'].items()}
rnames = {int(k): v['name'] for k, v in m['rei'].items()}

# --- locations ---
SUPER = [31, 64, 103, 137, 158, 4, 166]          # 7 GBD super-regions
SDI = [44637, 44636, 44639, 44634, 44635]        # low -> high quintiles
regions, countries = [], []
def walk(n, depth, parent):
    if depth == 2 and parent in SUPER:
        regions.append(n['id'])
    if depth == 3 and parent in regions:
        countries.append(n['id'])
    for c in n.get('children', []):
        walk(c, depth + 1, n['id'])
for n in h['locations']:
    if isinstance(n, dict) and n['id'] == 1:
        walk(n, 0, None)
assert len(countries) == 204, f"countries={len(countries)}"
assert len(regions) == 21, f"regions={len(regions)}"
LOCS_ALL = [1] + SDI + regions + countries       # 231

YEARS = list(range(1990, 2024))
AGES_EO5 = [8, 9, 10, 11, 12, 13, 14]            # 15-19 ... 45-49
AGES_AGG = [24, 27]                              # 15-49 aggregate, age-standardized
MEAS = [1, 2, 6]                                 # deaths, DALYs, incidence
RMEAS = [1, 2]                                   # risk: deaths, DALYs only
SITES = [444,447,450,411,414,441,417,453,456,423,426,459,462,429,432,435,
         465,438,468,471,474,477,480,483,484,485,486,487,489]

id_map = {
    'version': 8352, 'gbd_round': 2023,
    'locations': {str(i): lnames[i] for i in LOCS_ALL},
    'sdi_quintiles': {str(i): lnames[i] for i in SDI},
    'regions': {str(i): lnames[i] for i in regions},
    'countries': {str(i): lnames[i] for i in countries},
    'causes': {str(i): cnames[i] for i in [410] + SITES},
    'risks': {str(i): rnames[i] for i in [104, 105, 108]},
    'ages': {'8':'15-19','9':'20-24','10':'25-29','11':'30-34','12':'35-39',
             '13':'40-44','14':'45-49','24':'15-49','27':'Age-standardized',
             '22':'All ages','25':'50-69','26':'70+','15':'50-54','16':'55-59',
             '17':'60-64','18':'65-69','19':'70-74','20':'75-79','21':'80+'},
    'measures': {'1':'Deaths','2':'DALYs','6':'Incidence','44':'Population'},
    'metrics': {'1':'Number','2':'Percent','3':'Rate'},
    'sexes': {'1':'Male','2':'Female','3':'Both'},
}
json.dump(id_map, open(f'{BASE}/data/id_map.json', 'w'), ensure_ascii=False, indent=1)

queries = []
def q(name, context, causes=None, reis=None, locs=None, ages=None, measures=None,
      metrics=None, sexes=None, years=None):
    p = {'context': context, 'population_group': 1, 'base': 'single',
         'version': 8352, 'start_year': 1990, 'fetch_all_years': 'false'}
    arr = {'year': years or YEARS, 'location': locs or [1], 'age': ages or AGES_AGG,
           'measure': measures or MEAS, 'metric': metrics or [1, 3], 'sex': sexes or [3]}
    if context == 'risk':
        arr['rei'] = reis
        arr['cause'] = causes or [410]
    elif context == 'population':
        pass
    else:
        arr['cause'] = causes or [410]
    n = 1
    for v in arr.values():
        n *= len(v)
    queries.append({'name': name, 'params': p, 'arrays': arr, 'expected_rows': n})

# A. global trends: all age detail + aggregates, 3 measures, both sexes + M/F
q('A_global_trends', 'cause', locs=[1],
  ages=AGES_EO5 + [24, 27, 22, 25, 26], measures=MEAS, metrics=[1, 3], sexes=[1, 2, 3])

# B/C/D. country-level early-onset + ASR, per sex
for sx, tag in [(3, 'both'), (1, 'male'), (2, 'female')]:
    q(f'BCD_country_eo_{tag}', 'cause', locs=LOCS_ALL, ages=AGES_AGG,
      measures=MEAS, metrics=[1, 3], sexes=[sx])

# E. country x 5y age groups (both sexes), per measure
for meas in MEAS:
    q(f'E_country_age_m{meas}', 'cause', locs=[1] + countries, ages=AGES_EO5,
      measures=[meas], metrics=[1, 3], sexes=[3])

# F. cancer sites x (global+SDI+regions), per measure x age
for meas in MEAS:
    for age in AGES_AGG:
        q(f'F_sites_m{meas}_a{age}', 'cause', causes=SITES, locs=[1] + SDI + regions,
          ages=[age], measures=[meas], metrics=[1, 3], sexes=[3])

# G/H/I. risk-attributable neoplasms, country level, per rei x sex
for rei in [104, 105, 108]:
    for sx, tag in [(3, 'both'), (1, 'male'), (2, 'female')]:
        q(f'GHI_risk{rei}_country_{tag}', 'risk', reis=[rei], causes=[410],
          locs=LOCS_ALL, ages=AGES_AGG, measures=RMEAS, metrics=[1, 2, 3], sexes=[sx])

# J. risk x age groups, global, all sexes
q('J_risk_age_global', 'risk', reis=[104, 105, 108], causes=[410], locs=[1],
  ages=AGES_EO5 + AGES_AGG, measures=RMEAS, metrics=[1, 2, 3], sexes=[1, 2, 3])

# K. risk x cancer sites (valid pairs only), global+SDI+regions, both sexes
valid = m['validCausesByREI']['8352']
for rei in [104, 105, 108]:
    ok = [c for c in SITES if c in valid[str(rei)]['causes']]
    for meas in RMEAS:
        for age in AGES_AGG:
            q(f'K_risk{rei}_sites_m{meas}_a{age}', 'risk', reis=[rei], causes=ok,
              locs=[1] + SDI + regions, ages=[age], measures=[meas],
              metrics=[1, 3], sexes=[3])

# L. population counts (offsets for rates / APC / projections)
for sx in [1, 2, 3]:
    q(f'L_pop_sex{sx}', 'population', locs=[1] + countries,
      ages=AGES_EO5 + [24], measures=[44], metrics=[1], sexes=[sx])

over = [x['name'] for x in queries if x['expected_rows'] > 99000]
total = sum(x['expected_rows'] for x in queries)
json.dump(queries, open(f'{BASE}/data/queries.json', 'w'), indent=1)
print(f"queries: {len(queries)}, total expected rows: {total:,}")
print(f"over-limit (>99k): {over}")
for x in queries:
    print(f"  {x['name']}: {x['expected_rows']:,}")