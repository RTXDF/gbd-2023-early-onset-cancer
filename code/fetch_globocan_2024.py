"""Fetch GLOBOCAN 2024 (IARC GCO Cancer Today v3) for ages 15-49.

The API encodes 15-19 as age index 3 and 45-49 as index 9, so the correct
`ages_group` token for 15-49 is `3_9`.  The earlier `4_10` token silently
returned ages 20-54 and must not be used.

Caches raw JSON in data/raw/globocan2024_3_9/.
"""
from __future__ import annotations

import json
import os
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "globocan2024_3_9")
os.makedirs(RAW, exist_ok=True)

BASE = "https://gco-api.iarc.fr/api/globocan/v3/2024"
AGES = "3_9"
CANCER_ALL = 40
SITE_CODES = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 20, 23,
              24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (research data retrieval)"})


def get(path: str, fname: str, params=None):
    dst = os.path.join(RAW, fname)
    if os.path.exists(dst):
        return json.loads(open(dst).read())
    url = f"{BASE}/{path}"
    for attempt in range(4):
        try:
            r = s.get(url, params=params, timeout=60)
            if r.status_code == 200:
                open(dst, "w").write(r.text)
                return r.json()
            print("HTTP", r.status_code, url)
        except Exception as exc:
            print("retry", attempt + 1, exc)
        time.sleep(1 + attempt)
    raise RuntimeError(url)


# metadata
get("meta/populations/all/", "meta_populations.json")

# world aggregate, all cancers, by type and sex
for typ in (0, 1):
    for sex in (0, 1, 2):
        get(f"data/rate/{typ}/{sex}/900/{CANCER_ALL}/",
            f"world_type{typ}_sex{sex}.json", {"ages_group": AGES})

# all countries, all cancers, by type and sex
for typ in (0, 1):
    for sex in (0, 1, 2):
        get(f"data/rate/{typ}/{sex}/all/{CANCER_ALL}/",
            f"countries_type{typ}_sex{sex}.json", {"ages_group": AGES})

# world, each site, both sexes
for typ in (0, 1):
    for code in SITE_CODES:
        get(f"data/rate/{typ}/0/900/{code}/",
            f"world_type{typ}_sex0_cancer{code}.json", {"ages_group": AGES})

print("cached in", RAW)
