from __future__ import annotations

import random
import time

import pyproj
import pystac_client

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

def utm_epsg_from_bbox(bbox):
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def bbox_to_projected_bounds(bbox, epsg):
    transformer = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    minx, miny = transformer.transform(bbox[0], bbox[1])
    maxx, maxy = transformer.transform(bbox[2], bbox[3])
    return (min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))


def search_items(bbox, start_date, end_date, max_cloud=60, limit=15, retries=6):
    last_err = None
    for k in range(retries):
        try:
            catalog = pystac_client.Client.open(STAC_URL)
            search = catalog.search(
                collections=[COLLECTION],
                bbox=list(bbox),
                datetime=f"{start_date}/{end_date}",
                query={"eo:cloud_cover": {"lt": max_cloud}},
                limit=limit,
            )
            items = list(search.items())
            items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 1e9))
            return items
        except Exception as e:
            last_err = e
            time.sleep((2 ** k) + random.random())  # 1s,2s,4s...
    raise last_err

