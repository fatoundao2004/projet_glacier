from __future__ import annotations

from pathlib import Path
import os
import requests
from dotenv import load_dotenv
import zipfile
import pystac_client
import pyproj
import stackstac
import rioxarray
import pandas as pd
import time, random


BASE_URL = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0272_GLIMS_v1/"
VERSION_TAG = "v01.0"  # si ça change un jour, vous modifiez ici

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

def adaptive_buffer_deg(geom, k=0.5, min_buf=0.005, max_buf=0.02):
    minx, miny, maxx, maxy = geom.bounds
    max_dim = max(maxx - minx, maxy - miny)
    buf = k * max_dim
    return max(min_buf, min(max_buf, buf))

def build_requests(glaciers_gdf, out_root: Path, k=0.5, min_buf=0.005, max_buf=0.02):
    """
    glaciers_gdf must contain: glac_id, region, geometry (EPSG:4326), year_img
    out_root: .../data/sentinel2
    """
    out_root = Path(out_root)
    rows = []

    for _, r in glaciers_gdf.iterrows():
        reg = r["region"]
        glac_id = r["glac_id"]
        y = int(r["year_img"])

        minx, miny, maxx, maxy = r["geometry"].bounds
        buf = adaptive_buffer_deg(r["geometry"], k=k, min_buf=min_buf, max_buf=max_buf)

        bbox = (minx - buf, miny - buf, maxx + buf, maxy + buf)

        out_path = out_root / "composites" / reg / f"{glac_id}_{y}_summer.tif"

        rows.append({
            "glac_id": glac_id,
            "region": reg,
            "year": y,
            "bbox_minlon": bbox[0],
            "bbox_minlat": bbox[1],
            "bbox_maxlon": bbox[2],
            "bbox_maxlat": bbox[3],
            "out_path": str(out_path),
            "buffer_deg": buf,   # utile pour debug
        })

    return pd.DataFrame(rows)


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


def run_fetch(
    requests_df: pd.DataFrame,
    months=(6, 7, 8, 9),
    bands=("blue", "green", "red", "nir"),
    resolution=10,
    max_cloud=60,
    limit=15,
    reducer="first",
):
    """
    Loop over requests and fetch composites. Returns status df.
    """
    statuses = []
    for _, row in requests_df.iterrows():
        y = int(row["year"])
        region = row["region"]

        # mois par région (simple)
        if region == "andes":
            months_use = (1, 2, 3)      # été austral (simple, sans wrap)
        else:
            months_use = (6, 7, 8, 9)   # été boréal

        start_date = f"{y}-{months_use[0]:02d}-01"
        end_date   = f"{y}-{months_use[-1]:02d}-30"

        bbox = (
            float(row["bbox_minlon"]),
            float(row["bbox_minlat"]),
            float(row["bbox_maxlon"]),
            float(row["bbox_maxlat"]),
        )

        out_path = Path(row["out_path"])

        try:
            meta = fetch_composite(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                out_path=out_path,
                bands=bands,
                resolution=resolution,
                max_cloud=max_cloud,
                limit=limit,
                reducer=reducer,
            )
        except Exception as e:
            meta = {"status": "error", "path": str(out_path), "error": repr(e)}

        statuses.append({
            "glac_id": row["glac_id"],
            "region": row["region"],
            "year": y,
            **meta,
        })

    return pd.DataFrame(statuses)

def year_from_filename(path: str | Path) -> int:
    """
    Extract year from filenames like: <glac_id>_2019_summer.tif
    Returns an int year.
    """
    p = Path(path)
    m = re.search(r"_(\d{4})_summer$", p.stem)
    if not m:
        raise ValueError(f"Could not parse year from filename: {p.name}")
    return int(m.group(1))


def glims_mask_for_composite(
    tif_path: str | Path,
    glims_gdf: gpd.GeoDataFrame,
    area_min: float = 0.05,
) -> tuple[np.ndarray, int, gpd.GeoDataFrame]:
    """
    Build a binary mask aligned with a Sentinel-2 composite GeoTIFF.

    Steps:
    - Read GeoTIFF to get CRS, bounds, transform, shape.
    - Parse target year from filename (..._<year>_summer.tif).
    - Reproject GLIMS to image CRS.
    - Select polygons intersecting the image footprint.
    - Optionally filter by area_min.
    - For each glac_id, keep the outline whose src_date_dt year is closest to year_img.
    - Rasterize to a (H, W) uint8 mask where glacier=1, background=0.

    Returns:
      mask: np.ndarray (H, W) dtype uint8
      year_img: int
      inter_one: GeoDataFrame of selected outlines (one per glac_id)
    """
    tif_path = Path(tif_path)
    year_img = year_from_filename(tif_path)

    da = rxr.open_rasterio(tif_path)  # (band, y, x)
    img_crs = da.rio.crs
    if img_crs is None:
        raise ValueError(f"GeoTIFF has no CRS: {tif_path}")

    transform = da.rio.transform()
    H, W = da.rio.height, da.rio.width
    minx, miny, maxx, maxy = da.rio.bounds()
    patch_geom = box(minx, miny, maxx, maxy)

    gl = glims_gdf.copy()

    # Ensure GLIMS has CRS and required columns
    if gl.crs is None:
        gl = gl.set_crs(4326)

    if "src_date_dt" not in gl.columns:
        raise ValueError("glims_gdf must contain 'src_date_dt' (datetime) column.")

    if "glac_id" not in gl.columns:
        raise ValueError("glims_gdf must contain 'glac_id' column.")

    gl["glac_id"] = gl["glac_id"].astype(str).str.strip()
    gl = gl.to_crs(img_crs)

    # Intersect with patch footprint
    inter = gl[gl.intersects(patch_geom)].copy()

    if len(inter) == 0:
        # No outlines intersect; return empty mask
        return np.zeros((H, W), dtype=np.uint8), year_img, inter

    if area_min is not None and "area" in inter.columns:
        inter = inter[inter["area"] >= area_min].copy()

    if len(inter) == 0:
        return np.zeros((H, W), dtype=np.uint8), year_img, inter

    # One outline per glacier: closest year
    inter["src_year"] = inter["src_date_dt"].dt.year
    inter["gap"] = (inter["src_year"] - year_img).abs()

    inter_one = (
        inter.sort_values(["glac_id", "gap"])
        .drop_duplicates("glac_id", keep="first")
        .copy()
    )

    shapes = [
        (geom, 1)
        for geom in inter_one.geometry
        if geom is not None and not geom.is_empty
    ]

    mask = rasterize(
        shapes=shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    return mask, year_img, inter_one