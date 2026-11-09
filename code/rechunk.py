#!/usr/bin/env python3
"""Re-chunk GBD queries to <=9000 cross-product combos per request (API 413 cap ~10k)."""
import json, math, itertools, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = 9000
queries = json.load(open(os.path.join(ROOT, 'data', 'queries.json')))

# already downloaded successfully
SKIP = {'A_global_trends'}

out = []
for q in queries:
    if q['name'] in SKIP:
        continue
    arrays = q['arrays']
    years = arrays['year']
    per_year = math.prod(len(v) for k, v in arrays.items() if k != 'year')
    assert per_year <= CAP, f"{q['name']}: per-year combos {per_year} exceed cap, need finer splitting"
    ypc = max(1, CAP // per_year)
    nchunks = math.ceil(len(years) / ypc)
    for i in range(nchunks):
        ys = years[i * ypc:(i + 1) * ypc]
        a = {k: (ys if k == 'year' else v) for k, v in arrays.items()}
        out.append({
            'name': f"{q['name']}_y{ys[0]}-{ys[-1]}",
            'p': q['params'],
            'a': a,
            'n': per_year * len(ys),
        })

spec = {'queries': out}
dst = '/tmp/dl_spec3.json'
json.dump(spec, open(dst, 'w'), separators=(',', ':'))
total_rows = sum(x['n'] for x in out)
print(f"queries: {len(out)}, expected combo-rows: {total_rows:,}, max per query: {max(x['n'] for x in out):,}")
print(f"file size: {os.path.getsize(dst):,} bytes -> {dst}")