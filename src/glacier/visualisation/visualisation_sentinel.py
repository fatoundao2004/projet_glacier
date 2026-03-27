from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stackstac
import calendar

from glacier.data.stac_utils import utm_epsg_from_bbox, bbox_to_projected_bounds, search_items
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from IPython.display import display
from scipy.ndimage import binary_dilation

CLS_LABELS = {
    0: "autre",
    1: "nuages (7/8/9/10)",
    2: "ombre nuage (3)",
    3: "neige/glace (11)",
    4: "pixels sombres (2)",
}

CLS_CMAP = ListedColormap(["black", "white", "purple", "cyan", "gray"])
CLS_NORM = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5], CLS_CMAP.N)

# Sentinel-2 L2A SCL classes
SCL_CLOUD  = {7, 8, 9, 10}  # cloud low/med/high + cirrus
SCL_DARK = {2}
SCL_SHADOW = {3}            # cloud shadow
SCL_SNOW   = {11}           # snow/ice


def load_patch_for_item(item, bbox, bands=("blue", "green", "red", "nir"), resolution=10):
    epsg = utm_epsg_from_bbox(bbox)
    bounds = bbox_to_projected_bounds(bbox, epsg)
    try:
        da = stackstac.stack([item], assets=list(bands), epsg=epsg, resolution=resolution, bounds=bounds)
        return da.isel(time=0).astype("float32").values  # (band,y,x)
    except Exception:
        return None


def load_scl_for_item(item, bbox, resolution=20):
    asset = None
    if "scl" in item.assets:
        asset = "scl"
    elif "SCL" in item.assets:
        asset = "SCL"
    else:
        return None

    epsg = utm_epsg_from_bbox(bbox)
    bounds = bbox_to_projected_bounds(bbox, epsg)
    try:
        da = stackstac.stack([item], assets=[asset], epsg=epsg, resolution=resolution, bounds=bounds)
        return da.isel(time=0, band=0).astype("int16").values  # (y,x)
    except Exception:
        return None


def scl_fractions(scl):
    valid = (scl != 0)
    if scl.size == 0 or valid.mean() == 0:
        return dict(valid_frac=0.0, cloud_frac=np.nan, shadow_frac=np.nan, snow_frac=np.nan)

    cloud  = np.isin(scl, list(SCL_CLOUD))  & valid
    shadow = np.isin(scl, list(SCL_SHADOW)) & valid
    snow   = np.isin(scl, list(SCL_SNOW))   & valid

    return dict(
        valid_frac=float(valid.mean()),
        cloud_frac=float(cloud.mean()),
        shadow_frac=float(shadow.mean()),
        snow_frac=float(snow.mean()),
    )


def proxy_quality_metrics(x):
    X = x.copy()
    X[~np.isfinite(X)] = np.nan
    rgb = X[:3]
    bright = np.nanmean(rgb, axis=0)
    p90 = np.nanpercentile(bright, 90)
    p10 = np.nanpercentile(bright, 10)
    return dict(
        valid_frac=float(np.isfinite(bright).mean()),
        cloud_frac=float(np.nanmean(bright >= p90)),
        shadow_frac=float(np.nanmean(bright <= p10)),
        snow_frac=np.nan,
    )


def quality_score(metrics, w_cloud=1.0, w_snow=1.0, w_shadow=0.3):
    def _clean(v):
        if v is None: return 0.0
        if isinstance(v, float) and np.isnan(v): return 0.0
        return float(v)
    return (
        w_cloud  * _clean(metrics.get("cloud_frac"))
        + w_snow * _clean(metrics.get("snow_frac"))
        + w_shadow * _clean(metrics.get("shadow_frac"))
    )


def rank_dates_from_items(items, bbox, use_scl=True):
    items = sorted(items, key=lambda it: it.datetime)
    rows = []

    for it in items:
        m = None
        if use_scl:
            scl = load_scl_for_item(it, bbox)
            if scl is not None:
                m = scl_fractions(scl)

        if m is None:
            x = load_patch_for_item(it, bbox, bands=("blue","green","red","nir"), resolution=10)
            if x is None:
                continue
            m = proxy_quality_metrics(x)

        rows.append({
            "datetime": pd.Timestamp(it.datetime),
            "eo_cloud_cover": float(it.properties.get("eo:cloud_cover", np.nan)),
            **m,
            "score": quality_score(m),
        })

    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    return df, items


def plot_score_timeseries(df, title="Score vs date (plus bas = mieux)"):
    plt.figure(figsize=(10,3))
    plt.plot(df["datetime"], df["score"], marker="o")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def _rgb_from_patch(x):
    rgb = np.transpose(x, (1,2,0)).astype("float32")

    valid = np.isfinite(rgb).all(axis=2)  # pixel valide si R,G,B sont finis
    denom = np.nanpercentile(rgb[valid], 98) if valid.any() else np.nan

    if not np.isfinite(denom) or denom == 0:
        denom = np.nanmax(rgb[valid]) if valid.any() else 1.0
        if not np.isfinite(denom) or denom == 0:
            denom = 1.0

    rgb = np.clip(rgb / denom, 0, 1)

    # Ne pas convertir NaN en 0 : on mettra alpha=0 sur ces pixels
    rgb[~np.isfinite(rgb)] = 0.0
    alpha = valid.astype("float32")
    return rgb, alpha


def show_topk_gallery(df, items, bbox, topk=8, scl_panel=True):
    item_by_dt = {pd.Timestamp(it.datetime): it for it in items}
    n = min(int(topk), len(df))
    if n <= 0:
        print("Aucun item à afficher.")
        return

    cols = 2 if scl_panel else 1
    fig, axes = plt.subplots(n, cols, figsize=(10, 3*n))
    if n == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes.reshape(n, 1)

    for i in range(n):
        dt = pd.Timestamp(df.loc[i, "datetime"])
        it = item_by_dt.get(dt)
        if it is None:
            continue

        # RGB (gauche)
        x_rgb = load_patch_for_item(it, bbox, bands=("red","green","blue"), resolution=10)
        if x_rgb is None:
            axes[i,0].text(0.5, 0.5, "Image corrompue", ha="center", va="center")
            axes[i,0].axis("off")
            if scl_panel:
                axes[i,1].axis("off")
            continue
        rgb, alpha = _rgb_from_patch(x_rgb)
        axes[i,0].imshow(rgb, alpha=alpha)
        axes[i,0].set_title(f"{dt.date()} | score={df.loc[i,'score']:.3f}")
        axes[i,0].axis("off")

        # Masque SCL (droite)
        if not scl_panel:
            continue

        scl = load_scl_for_item(it, bbox)
        if scl is None:
            axes[i,1].text(0.5, 0.5, "SCL absent\n(proxy only)", ha="center", va="center")
            axes[i,1].axis("off")
        else:
            cls = np.zeros_like(scl, dtype=np.uint8)
            cls[np.isin(scl, list(SCL_CLOUD))]  = 1
            cls[np.isin(scl, list(SCL_SHADOW))] = 2
            cls[np.isin(scl, list(SCL_SNOW))]   = 3
            cls[np.isin(scl, list(SCL_DARK))]   = 4

            axes[i,1].imshow(cls, cmap=CLS_CMAP, norm=CLS_NORM, interpolation="nearest")
            axes[i,1].set_title("Masque SCL")
            axes[i,1].axis("off")

    # Légende
    handles = [Patch(facecolor=CLS_CMAP(k), label=CLS_LABELS[k]) for k in [1, 2, 3, 4]]
    fig.legend(handles=handles, loc="upper right", frameon=True, title="Classes")

    plt.tight_layout()
    plt.show()

def monthly_best_grid(
    bbox,
    year: int,
    months=range(1, 13),
    max_cloud=90,
    limit=120,
    use_scl=True,
    top_per_month=1,
    day_end=30,
    glacier_geom_wgs84=None,
    scl_resolution=20,
    ring_iters=12,
):
    """
    Pour chaque mois:
      - récupère des items STAC sur [YYYY-MM-01, YYYY-MM-last_day]
      - calcule un score :
          * global bbox si glacier_geom_wgs84 is None
          * glacier-aware sinon
      - affiche la/les meilleures dates du mois (RGB + masque)
    """
    for m in months:
        last_day = calendar.monthrange(year, m)[1]
        start_date = f"{year}-{m:02d}-01"
        end_date   = f"{year}-{m:02d}-{last_day:02d}"

        items = search_items(bbox, start_date, end_date, max_cloud=max_cloud, limit=limit)
        if len(items) == 0:
            print(f"[{year}-{m:02d}] aucun item")
            continue

        if glacier_geom_wgs84 is None:
            df, items_sorted = rank_dates_from_items(items, bbox, use_scl=use_scl)

            head_cols = ["datetime", "score", "cloud_frac", "snow_frac", "shadow_frac", "eo_cloud_cover"]
        else:
            df, items_sorted = rank_dates_from_items_glacier(
                items=items,
                bbox=bbox,
                glacier_geom_wgs84=glacier_geom_wgs84,
                use_scl=use_scl,
                scl_resolution=scl_resolution,
                ring_iters=ring_iters,
            )

            head_cols = [
                "datetime",
                "score",
                "g_cloud_frac",
                "g_shadow_frac",
                "g_snow_frac",
                "r_snow_frac",
                "g_valid_frac",
                "eo_cloud_cover",
            ]

        head = df.head(top_per_month)[head_cols]

        print(f"\n=== {year}-{m:02d} | n_items={len(items)} ===")
        display(head)

        show_topk_gallery(df, items_sorted, bbox, topk=top_per_month, scl_panel=True)

def rasterize_glacier_mask(glacier_geom_wgs84, bbox, out_shape, resolution, epsg):
    # reproj geometry -> epsg patch
    tr = Transformer.from_crs(4326, epsg, always_xy=True).transform
    geom_proj = shp_transform(tr, glacier_geom_wgs84)

    # rebuild transform (bounds + resolution)
    minx, miny, maxx, maxy = bbox_to_projected_bounds(bbox, epsg)
    transform = from_bounds(minx, miny, maxx, maxy, out_shape[1], out_shape[0])

    mask = rasterize([(geom_proj, 1)], out_shape=out_shape, transform=transform, fill=0, dtype="uint8")
    return mask.astype(bool)

def scl_fractions_on_mask(scl, mask):
    valid = (scl != 0) & mask
    denom = valid.sum()

    if denom == 0:
        return dict(
            valid_px=0,
            cloud_frac=np.nan,
            shadow_frac=np.nan,
            snow_frac=np.nan,
            dark_frac=np.nan,
            valid_frac=np.nan,
        )

    cloud  = np.isin(scl, list(SCL_CLOUD))  & valid
    shadow = np.isin(scl, list(SCL_SHADOW)) & valid
    snow   = np.isin(scl, list(SCL_SNOW))   & valid
    dark   = np.isin(scl, list(SCL_DARK))   & valid

    return dict(
        valid_px=int(denom),
        cloud_frac=float(cloud.sum() / denom),
        shadow_frac=float(shadow.sum() / denom),
        snow_frac=float(snow.sum() / denom),
        dark_frac=float(dark.sum() / denom),
        valid_frac=float(denom / mask.sum()) if mask.sum() > 0 else np.nan,
    )

def glacier_ring_mask(glacier_mask, n_iter=12):
    """
    Crée une couronne autour du glacier.
    n_iter dépend de la résolution; à 20 m, 12 pixels ~ 240 m.
    """
    dilated = binary_dilation(glacier_mask, iterations=n_iter)
    ring = dilated & (~glacier_mask)
    return ring


def glacier_aware_score(m_glacier, m_ring=None, month=None):
    """
    Plus bas = mieux.
    Idée:
      - nuages sur glacier = très mauvais
      - ombres sur glacier = mauvais
      - neige autour = mauvais modéré
      - neige sur glacier = faible pénalité seulement
      - peu de pixels valides sur glacier = pénalité
      - bonus saisonnier possible
    """
    def clean(d, k, default=0.0):
        v = d.get(k, default) if d is not None else default
        if v is None:
            return default
        if isinstance(v, float) and np.isnan(v):
            return default
        return float(v)

    g_cloud  = clean(m_glacier, "cloud_frac")
    g_shadow = clean(m_glacier, "shadow_frac")
    g_dark   = clean(m_glacier, "dark_frac")
    g_valid  = clean(m_glacier, "valid_frac")

    r_cloud  = clean(m_ring, "cloud_frac")
    r_shadow = clean(m_ring, "shadow_frac")
    r_snow   = clean(m_ring, "snow_frac")

    score = 0.0

    # Priorité absolue : voir le glacier
    score += 4.0 * g_cloud
    score += 2.0 * g_shadow
    score += 1.0 * g_dark

    # g_snow : quelque chose qu'on avait avant mais retiré, le SCL classe le glacier lui-même comme neige/glace,
    # c'est normal et ne devrait pas pénaliser le score.

    # La neige AUTOUR du glacier reste un signal utile :
    # si la couronne est enneigée, c'est de la neige saisonnière
    score += 1.5 * r_snow
    score += 0.8 * r_cloud
    score += 0.4 * r_shadow

    # Peu de pixels valides sur le glacier = grosse pénalité
    score += 3.0 * max(0.0, 1.0 - g_valid)

    return float(score)

def rank_dates_from_items_glacier(
    items,
    bbox,
    glacier_geom_wgs84,
    use_scl=True,
    scl_resolution=20,
    ring_iters=12,
):
    items = sorted(items, key=lambda it: it.datetime)
    rows = []

    epsg = utm_epsg_from_bbox(bbox)

    for it in items:
        dt = pd.Timestamp(it.datetime)
        month = dt.month

        # SCL
        scl = load_scl_for_item(it, bbox, resolution=scl_resolution) if use_scl else None

        if scl is None:
            # fallback proxy si SCL absent
            x = load_patch_for_item(it, bbox, bands=("blue", "green", "red", "nir"), resolution=10)
            if x is None:
                continue   # ← skip cet item complètement
            m_proxy = proxy_quality_metrics(x)

            rows.append({
                "datetime": dt,
                "eo_cloud_cover": float(it.properties.get("eo:cloud_cover", np.nan)),
                "score": quality_score(m_proxy),
                "mode": "proxy_only",
            })
            continue

        # masque glacier dans la grille du SCL
        glacier_mask = rasterize_glacier_mask(
            glacier_geom_wgs84=glacier_geom_wgs84,
            bbox=bbox,
            out_shape=scl.shape,
            resolution=scl_resolution,
            epsg=epsg,
        )

        ring_mask = glacier_ring_mask(glacier_mask, n_iter=ring_iters)

        m_glacier = scl_fractions_on_mask(scl, glacier_mask)
        m_ring    = scl_fractions_on_mask(scl, ring_mask)

        score = glacier_aware_score(m_glacier, m_ring, month=month)

        rows.append({
            "datetime": dt,
            "eo_cloud_cover": float(it.properties.get("eo:cloud_cover", np.nan)),
            "score": score,
            "mode": "scl_glacier",
            "g_valid_px": m_glacier["valid_px"],
            "g_valid_frac": m_glacier["valid_frac"],
            "g_cloud_frac": m_glacier["cloud_frac"],
            "g_shadow_frac": m_glacier["shadow_frac"],
            "g_snow_frac": m_glacier["snow_frac"],
            "g_dark_frac": m_glacier["dark_frac"],
            "r_valid_px": m_ring["valid_px"],
            "r_cloud_frac": m_ring["cloud_frac"],
            "r_shadow_frac": m_ring["shadow_frac"],
            "r_snow_frac": m_ring["snow_frac"],
        })

    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    return df, items
