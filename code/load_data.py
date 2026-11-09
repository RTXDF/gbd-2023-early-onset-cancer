#!/usr/bin/env python3
"""Load raw GBD JSON chunks -> tidy per-family panels (pickle) with ID->name mapping.

Raw files: data/raw/{family}[_y{start}-{end}].json, schema {"cols":[...],"data":[[...]]}
Outputs:   data/panels/{family}.pkl  (DataFrame, IDs kept as int cols + *_name cols)
           data/panels/_load_report.json (row counts, suppression vs expected)
"""
import json, os, re, glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw')
OUT = os.path.join(ROOT, 'data', 'panels')
os.makedirs(OUT, exist_ok=True)

idmap = json.load(open(os.path.join(ROOT, 'data', 'id_map.json')))
queries = json.load(open(os.path.join(ROOT, 'data', 'queries.json')))

ID_COLS = {
    'location': 'locations', 'age': 'ages', 'measure': 'measures',
    'metric': 'metrics', 'sex': 'sexes', 'cause': 'causes', 'rei': 'risks',
}
SUFFIX = re.compile(r'_y\d{4}-\d{4}$')

def family_of(fname):
    return SUFFIX.sub('', fname[:-5] if fname.endswith('.json') else fname)

# group raw files by family
files = {}
for path in sorted(glob.glob(os.path.join(RAW, '*.json'))):
    base = os.path.basename(path)
    if base.startswith('dl_spec') or base in ('loopback_test.json',):
        continue
    fam = family_of(base)
    files.setdefault(fam, []).append(path)

expected = {q['name']: q['expected_rows'] for q in queries}
report = {}
for fam, paths in sorted(files.items()):
    frames = []
    for p in paths:
        j = json.load(open(p))
        df = pd.DataFrame(j['data'], columns=j['cols'])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # map IDs to names
    for col, mapname in ID_COLS.items():
        if col in df.columns:
            m = {int(k): v for k, v in idmap[mapname].items()}
            df[col + '_name'] = df[col].map(m)
    # dedupe safety (chunks are disjoint by year, but be safe)
    keycols = [c for c in df.columns if c not in ('val', 'upper', 'lower')]
    before = len(df)
    df = df.drop_duplicates(subset=keycols)
    dup = before - len(df)
    df.to_pickle(os.path.join(OUT, fam + '.pkl'))
    exp = expected.get(fam)
    report[fam] = {
        'files': len(paths), 'rows': len(df), 'expected': exp,
        'coverage': round(len(df) / exp, 4) if exp else None, 'duplicates_removed': dup,
        'cols': list(df.columns),
        'years': [int(df.year.min()), int(df.year.max())] if 'year' in df else None,
    }
    print(f"{fam:42s} files={len(paths):3d} rows={len(df):8,d} expected={exp or '-':>8} "
          f"cov={report[fam]['coverage']}")

json.dump(report, open(os.path.join(OUT, '_load_report.json'), 'w'), indent=2)
print(f"\nTotal families: {len(report)}; panels -> {OUT}")