from __future__ import annotations
 
from pathlib import Path
import os
import requests
import zipfile
import pystac_client
import pyproj
from pyproj import Transformer
import stackstac
import pandas as pd
import geopandas as gpd
import time, random
import re
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
 
from shapely.geometry import box
from rasterio.features import rasterize
from shapely.ops import transform as shp_transform
from glacier.visualisation import rank_dates_from_items_glacier
 
 
BASE_URL = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0272_GLIMS_v1/"
VERSION_TAG = "v01.0"
 
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
 
 
def _session() -> requests.Session:
    """ Pour télécharger le jeu de données il faut créer un compte Earthdata.
    Le truc c'est qu'on ne veut pas commit le username et mot de passe du compte sur github.
    Donc on mets dans le .gitignore un fichier projet_glacier/_netrc qui contient:
    machine urs.earthdata.nasa.gov
     login ...
     password ...
    """
    root = repo_root()
    netrc_path = root / "_netrc"
    os.environ["NETRC"] = str(netrc_path)  
 
    s = requests.Session()
    s.trust_env = True
    return s
 
def _targets_for_date(date_yyyymmdd: str) -> list[str]:
    """
    Download les 4 fichiers (north/south + md5).
    date_yyyymmdd: "20260114"
    """
    if not (isinstance(date_yyyymmdd, str) and date_yyyymmdd.isdigit() and len(date_yyyymmdd) == 8):
        raise ValueError("date_yyyymmdd must be a string like '20260114'")
 
    def zip_name(region: str) -> str:
        return f"NSIDC-0272_glims_db_{region}_{date_yyyymmdd}_{VERSION_TAG}.zip"
 
    north = zip_name("north")
    south = zip_name("south")
    return [north, north + ".md5", south, south + ".md5"]
 
def _download_one(session: requests.Session, url: str, out_path: Path) -> None:
    """
    Atomic download: writes to .part then renames, to avoid partial/corrupt files.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
 
    # skip si déjà downloaded
    if out_path.exists() and out_path.stat().st_size > 0:
        return
 
    with session.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
 
    tmp.replace(out_path)
 
def repo_root(start: Path | None = None) -> Path:
    """
    Helper pour s'assurer que l'on mets les données dans le dossier data au lieu de dans notebooks.
    """
    p = (start or Path(__file__).resolve()).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Repo root not found (no .git).")
 
def fetch_data(date_yyyymmdd: str, raw_dir: str | Path = "data/raw/glims_v1") -> list[Path]:
    """
    Fetch GLIMS NSIDC-0272 pour une date spécifique.
    Télécharge north+south zip + md5 dans raw_dir.
    Le md5 sert à vérifier que le zip n'est pas corrompu.
    """
    root = repo_root()
 
    raw_dir = Path(raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
 
    s = _session()
    targets = _targets_for_date(date_yyyymmdd)
 
    downloaded: list[Path] = []
    for name in targets:
        out = raw_dir / name
        _download_one(s, BASE_URL + name, out)
        downloaded.append(out)
 
    return downloaded
 
def unzip_to(paths: list[Path], extracted_root: Path) -> list[Path]:
    """
    Dézippe les fichiers .zip dans extracted_root.
    Skip si le dossier d'extraction existe déjà et n'est pas vide.
    Ignore les fichiers non-.zip (.zip.md5 dans notre cas).
    """
    extracted_root = Path(extracted_root)
    extracted_root.mkdir(parents=True, exist_ok=True)
 
    out_dirs: list[Path] = []
    for p in paths:
        p = Path(p)
 
        if not p.name.lower().endswith(".zip"): # ignore les .zip.md5
            continue
 
        dest = extracted_root / p.stem  # stem enlève juste .zip
 
        # skip si déjà extrait (dossier existe et contient qqch)
        if dest.exists() and any(dest.iterdir()):
            out_dirs.append(dest)
            continue
 
        dest.mkdir(parents=True, exist_ok=True)
 
        try:
            with zipfile.ZipFile(p, "r") as z:
                z.extractall(dest)
        except zipfile.BadZipFile as e:
            continue
        out_dirs.append(dest)
    return out_dirs
 
# Ici si vous voulez savoir pourquoi on choisi ces periodes precisement c'est avec ces periodes que nous avions les meilleurs
# resultats dans le notebook de visualisation
 
REGION_WINDOWS = {
    "alps":     {"start": (9, 15),  "end": (10, 10)},
    "caucasus": {"start": (8, 15),  "end": (9, 15)},
    "andes":    {"start": (4, 1),  "end": (5, 1)},
}
 
 
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
 
def build_requests(
    glaciers_gdf,
    out_root: Path,
    patch_size_px=314,
    resolution=10,
):
    out_root = Path(out_root)
    rows = []
 
    for _, r in glaciers_gdf.iterrows():
        reg = r["region"]
        glac_id = r["glac_id"]
        y = int(r["year_img"])
        geom = r["geometry"]
 
        bbox = fixed_bbox_around_geom(
            geom_wgs84=geom,
            patch_size_px=patch_size_px,
            resolution=resolution,
        )
 
        out_path = out_root / "composites" / reg / f"{glac_id}_{y}_topk.tif"
 
        rows.append({
            "glac_id": glac_id,
            "region": reg,
            "year": y,
            "geometry": geom,
            "bbox_minlon": bbox[0],
            "bbox_minlat": bbox[1],
            "bbox_maxlon": bbox[2],
            "bbox_maxlat": bbox[3],
            "out_path": str(out_path),
            "patch_size_px": patch_size_px,
            "resolution": resolution,
        })
 
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=glaciers_gdf.crs)
 
 
def dates_for_region_year(region: str, year: int, region_windows=None):
    if region_windows is None:
        region_windows = REGION_WINDOWS
 
    cfg = region_windows.get(region)
    if cfg is None:
        raise ValueError(f"Fenêtre temporelle absente pour la région: {region}")
 
    sm, sd = cfg["start"]
    em, ed = cfg["end"]
 
    start_date = f"{year}-{sm:02d}-{sd:02d}"
    end_date   = f"{year}-{em:02d}-{ed:02d}"
    return start_date, end_date
 
def run_fetch(
    requests_gdf,
    region_windows=None,
    bands=("blue", "green", "red", "nir", "swir16"),
    resolution=10,
    max_cloud=95,
    limit=80,
    topk=3,
    reducer="median",
    use_scl=True,
    scl_resolution=20,
    ring_iters=12,
):
    statuses = []
 
    for _, row in requests_gdf.iterrows():
        y = int(row["year"])
        region = row["region"]
        glacier_geom_wgs84 = row["geometry"]
 
        start_date, end_date = dates_for_region_year(
            region=region,
            year=y,
            region_windows=region_windows,
        )
 
        bbox = (
            float(row["bbox_minlon"]),
            float(row["bbox_minlat"]),
            float(row["bbox_maxlon"]),
            float(row["bbox_maxlat"]),
        )
 
        out_path = Path(row["out_path"])
 
        try:
            meta = fetch_composite_topk(
                bbox=bbox,
                glacier_geom_wgs84=glacier_geom_wgs84,
                start_date=start_date,
                end_date=end_date,
                out_path=out_path,
                bands=bands,
                resolution=resolution,
                max_cloud=max_cloud,
                limit=limit,
                topk=topk,
                reducer=reducer,
                use_scl=use_scl,
                scl_resolution=scl_resolution,
                ring_iters=ring_iters,
            )
        except Exception as e:
            meta = {"status": "error", "path": str(out_path), "error": repr(e)}
 
        statuses.append({
            "glac_id": row["glac_id"],
            "region": region,
            "year": y,
            **meta,
        })
 
    return pd.DataFrame(statuses)
 
 
 
 
def glims_mask_for_composite(tif_path, glims_gdf, max_gap_years=None):
    """
    Build a binary glacier mask aligned with a Sentinel-2 composite GeoTIFF.

    The function:
    - reads the composite GeoTIFF,
    - extracts the image year from the filename,
    - keeps all GLIMS outlines intersecting the patch extent,
    - for each glac_id, keeps only the outlines closest to the image year
      and dissolves them (a single glac_id can have multiple polygons),
    - optionally filters by max_gap_years,
    - clips them to the patch extent,
    - rasterizes them into a binary mask where glacier=1 and background=0.

    Parameters
    ----------
    tif_path : str or Path
        Path to a composite file named like <glac_id>_<year>_topk.tif
    glims_gdf : GeoDataFrame
        GLIMS outlines with at least columns: glac_id, src_date_dt, geometry
    max_gap_years : int or None
        Maximum allowed gap in years between the image year and a GLIMS
        outline year. None = no filter (keep all).

    Returns
    -------
    mask : np.ndarray
        Binary mask of shape (H, W), dtype uint8
    year_img : int
        Year extracted from the composite filename
    inter : GeoDataFrame
        All GLIMS outlines used to create the mask
    """
    tif_path = Path(tif_path)

    year_img = int(re.search(r"_(\d{4})_(summer|topk)$", tif_path.stem).group(1))

    da = rxr.open_rasterio(tif_path)
    img_crs = da.rio.crs
    transform = da.rio.transform()
    H, W = da.rio.height, da.rio.width
    patch_geom = box(*da.rio.bounds())

    gl = glims_gdf.copy()
    if gl.crs is None:
        gl = gl.set_crs(4326)

    gl["glac_id"] = gl["glac_id"].astype(str).str.strip()
    gl = gl.to_crs(img_crs)

    # garder tous les outlines qui intersectent le patch
    inter = gl[gl.intersects(patch_geom)].copy()

    # pour chaque glacier, garder les outlines les plus proches de l'année image
    # puis fusionner toutes les géométries (un même glac_id peut avoir plusieurs polygones)
    inter["gap"] = (inter["src_date_dt"].dt.year - year_img).abs()
    inter["best_gap"] = inter.groupby("glac_id")["gap"].transform("min")
    inter = inter[inter["gap"] == inter["best_gap"]].copy()
    inter = inter.dissolve(by="glac_id").reset_index()

    if max_gap_years is not None:
        inter = inter[inter["best_gap"] <= max_gap_years].copy()

    inter = inter.drop(columns=["gap", "best_gap"])

    # clip à l'emprise du patch
    inter["geometry"] = inter.geometry.intersection(patch_geom)

    shapes = [
        (geom, 1)
        for geom in inter.geometry
        if geom is not None and not geom.is_empty
    ]

    mask = rasterize(
        shapes=shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    return mask, year_img, inter
 
def stretch(x):
    lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
 
def get_glims_outlines_for_patch(tif_path, glims_gdf, max_gap_years=None):
    tif_path = Path(tif_path)
    year_img = int(re.search(r"_(\d{4})_(summer|topk)$", tif_path.stem).group(1))

    da = rxr.open_rasterio(tif_path)
    img_crs = da.rio.crs
    patch_geom = box(*da.rio.bounds())

    gl = glims_gdf.copy()
    if gl.crs is None:
        gl = gl.set_crs(4326)

    gl["glac_id"] = gl["glac_id"].astype(str).str.strip()
    gl = gl.to_crs(img_crs)

    inter = gl[gl.intersects(patch_geom)].copy()

    if len(inter) == 0:
        return inter, year_img

    inter["gap"] = (inter["src_date_dt"].dt.year - year_img).abs()
    inter["best_gap"] = inter.groupby("glac_id")["gap"].transform("min")
    inter = inter[inter["gap"] == inter["best_gap"]].copy()
    inter = inter.dissolve(by="glac_id").reset_index()

    if max_gap_years is not None:
        inter = inter[inter["best_gap"] <= max_gap_years].copy()

    inter = inter.drop(columns=["gap", "best_gap"])
    inter["geometry"] = inter.geometry.intersection(patch_geom)
    inter = inter[~inter.geometry.is_empty].copy()

    return inter, year_img
 
def show_tif_rgb_with_outline(tif_path, glims_gdf=None, max_gap_years=3, ax=None):
    da = rxr.open_rasterio(tif_path).astype("float32")
    minx, miny, maxx, maxy = da.rio.bounds()
 
    blue  = stretch(da[0].values)
    green = stretch(da[1].values)
    red   = stretch(da[2].values)
    rgb = np.stack([red, green, blue], axis=-1)
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
 
    ax.imshow(
        rgb,
        extent=[minx, maxx, miny, maxy],
        origin="upper",
        interpolation="bilinear"
    )
 
    if glims_gdf is not None:
        inter, year_img = get_glims_outlines_for_patch(
            tif_path, glims_gdf, max_gap_years=max_gap_years
        )
        if len(inter) > 0:
            dissolved = inter.dissolve()
            dissolved.plot(ax=ax, color="yellow", alpha=0.18, edgecolor="none")
            dissolved.boundary.plot(ax=ax, color="yellow", linewidth=1.0)
    else:
        year_img = "?"
 
    ax.set_title(f"{Path(tif_path).parent.name} | {Path(tif_path).stem} | year={year_img}", fontsize=8)
    ax.axis("off")
    return ax
 
def build_composite_from_items(
    items,
    bbox,
    bands=("blue", "green", "red", "nir"),
    resolution=10,
    reducer="median",
):
    if len(items) == 0:
        return None
 
    epsg = utm_epsg_from_bbox(bbox)
    bounds = bbox_to_projected_bounds(bbox, epsg)
 
    da = stackstac.stack(
        items,
        assets=list(bands),
        epsg=epsg,
        resolution=resolution,
        bounds=bounds,
    ).chunk({"time": 1, "x": 1024, "y": 1024})
 
    if reducer == "median":
        comp = da.median(dim="time", skipna=True)
    elif reducer == "first":
        comp = da.isel(time=0)
    else:
        raise ValueError(f"Reducer inconnu: {reducer}")
 
    comp = comp.assign_coords(band=list(bands))
    comp.rio.write_crs(f"EPSG:{epsg}", inplace=True)
    return comp
 
def save_composite_xarray(comp, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
 
    comp.rio.to_raster(
        out_path,
        compress="DEFLATE",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    return out_path

def fetch_composite(
    bbox,
    start_date,
    end_date,
    out_path: Path,
    bands=("blue", "green", "red", "nir"),
    resolution=10,
    max_cloud=60,
    limit=15,
    reducer="first",  # "median" or "first"
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return {"status": "skipped", "path": str(out_path)}

    items = search_items(bbox, start_date, end_date, max_cloud=max_cloud, limit=limit)
    if len(items) == 0:
        return {"status": "no_items", "path": str(out_path)}

    epsg = utm_epsg_from_bbox(bbox)
    bounds = bbox_to_projected_bounds(bbox, epsg)

    items = filter_readable_items(items, bbox, epsg, bounds, resolution, test_asset=bands[0])
    if len(items) == 0:
        return {"status": "no_readable_items", "path": str(out_path)}

    da = stackstac.stack(
        items,
        assets=list(bands),
        epsg=epsg,
        resolution=resolution,
        bounds=bounds,
    ).chunk({"time": 1, "x": 1024, "y": 1024})

    if reducer == "median":
        comp = da.median(dim="time", skipna=True)
    else:
        comp = da.isel(time=0)

    comp = comp.assign_coords(band=list(bands))
    comp.rio.write_crs(f"EPSG:{epsg}", inplace=True)

    comp.rio.to_raster(
        out_path,
        compress="DEFLATE",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    return {"status": "ok", "path": str(out_path), "n_items": len(items), "epsg": epsg} 

def filter_readable_items(items, bbox, epsg, bounds, resolution, test_asset="blue"):
    good = []
    for it in items:
        try:
            da = stackstac.stack(
                [it],
                assets=[test_asset],
                epsg=epsg,
                resolution=resolution,
                bounds=bounds,
            )
            # force une mini lecture
            _ = da.isel(time=0, band=0).data
            # déclenche un petit compute
            import dask.array as da_  # local import
            da_.asarray(_)[0:10, 0:10].compute()
            good.append(it)
        except Exception:
            continue
    return good
 
def fetch_composite_topk(
    bbox,
    glacier_geom_wgs84,
    start_date,
    end_date,
    out_path,
    bands=("blue", "green", "red", "nir"),
    resolution=10,
    max_cloud=95,
    limit=80,
    topk=3,
    reducer="median",
    use_scl=True,
    scl_resolution=20,
    ring_iters=12,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
 
    if out_path.exists():
        return {"status": "skipped", "path": str(out_path)}
 
    items = search_items(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        max_cloud=max_cloud,
        limit=limit,
    )
    if len(items) == 0:
        return {"status": "no_items", "path": str(out_path)}
 
    df_rank, _ = rank_dates_from_items_glacier(
        items=items,
        bbox=bbox,
        glacier_geom_wgs84=glacier_geom_wgs84,
        use_scl=use_scl,
        scl_resolution=scl_resolution,
        ring_iters=ring_iters,
    )
 
    best_dts = list(pd.to_datetime(df_rank.head(topk)["datetime"]))
    selected_items = [
        it for it in items
        if pd.Timestamp(it.datetime) in best_dts
    ]
 
    if len(selected_items) == 0:
        return {"status": "no_ranked_items", "path": str(out_path)}
 
    comp = build_composite_from_items(
        selected_items,
        bbox=bbox,
        bands=bands,
        resolution=resolution,
        reducer=reducer,
    )
 
    if comp is None:
        return {"status": "empty_composite", "path": str(out_path)}
 
    save_composite_xarray(comp, out_path)
 
    return {
        "status": "ok",
        "path": str(out_path),
        "n_items_all": len(items),
        "n_items_selected": len(selected_items),
        "topk": topk,
        "start_date": start_date,
        "end_date": end_date,
    }
 
def fixed_bbox_around_geom(geom_wgs84, patch_size_px=314, resolution=10):
    """
    Retourne une bbox WGS84 correspondant à un patch carré fixe.
    patch_size_px=314 et resolution=10 -> patch de 3140 m.
    """
    minx, miny, maxx, maxy = geom_wgs84.bounds
    center_lon = 0.5 * (minx + maxx)
    center_lat = 0.5 * (miny + maxy)
 
    epsg = utm_epsg_from_bbox((minx, miny, maxx, maxy))
 
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True).transform
    to_wgs = Transformer.from_crs(epsg, 4326, always_xy=True).transform
 
    cx, cy = to_utm(center_lon, center_lat)
 
    half_size_m = 0.5 * patch_size_px * resolution
    patch_utm = box(
        cx - half_size_m,
        cy - half_size_m,
        cx + half_size_m,
        cy + half_size_m,
    )
 
    patch_wgs = shp_transform(to_wgs, patch_utm)
    return patch_wgs.bounds
 