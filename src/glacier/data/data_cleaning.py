from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

# Puisqu'on commencera par utiliser que les alpes et la caucase
# BBox approx (lon_min, lat_min, lon_max, lat_max) en EPSG:4326

BBOX_ALPS     = (5.0, 44.0, 16.5, 48.5)
BBOX_CAUCASUS = (37.0, 41.0, 49.5, 45.5)
BBOX_ANDES = (-76.0, -56.0, -66.0, -16.0)
 
 
def keep_outlines(gdf: gpd.GeoDataFrame, value: str = "glac_bound") -> gpd.GeoDataFrame:
    if "line_type" not in gdf.columns:
        return gdf
    return gdf[gdf["line_type"] == value]
 
 
def drop_empty_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "geometry" not in gdf.columns:
        return gdf
    gdf = gdf[gdf.geometry.notnull()]
    return gdf[~gdf.geometry.is_empty]
 
 
def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(4326)
    return gdf.to_crs(4326)
 
def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int = 4326) -> gpd.GeoDataFrame:
    if getattr(gdf, "crs", None) is None:
        return gdf.set_crs(epsg)
    return gdf
 
def fix_invalid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)
    return drop_empty_geometries(gdf)
 
def parse_anlys_time(gdf: gpd.GeoDataFrame, col: str = "anlys_time") -> gpd.GeoDataFrame:
    if col not in gdf.columns:
        return gdf
    gdf = gdf.copy()
    gdf[col] = pd.to_datetime(gdf[col], errors="coerce", utc=True)
    return gdf
 
 
def parse_src_date(gdf: gpd.GeoDataFrame, col: str = "src_date") -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    if col in gdf.columns:
        gdf["src_date_dt"] = pd.to_datetime(gdf[col], errors="coerce", utc=True)
    else:
        gdf["src_date_dt"] = pd.NaT
    return gdf
 
 
 
def filter_positive_area(gdf: gpd.GeoDataFrame, col: str = "area") -> gpd.GeoDataFrame:
    if col not in gdf.columns:
        return gdf
    gdf = gdf[gdf[col].notna()].copy()
    return gdf[gdf[col] > 0].copy()
 
def drop_exact_dupes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["_geom_wkb"] = gdf.geometry.to_wkb()
 
    subset = ["glac_id", "_geom_wkb"]
    if "src_date_dt" in gdf.columns:
        subset.insert(1, "src_date_dt")
 
    gdf = gdf.drop_duplicates(subset=subset)
    return gdf.drop(columns="_geom_wkb")
 
def _in_bbox(centroids, bbox):
    return (
        centroids.x.between(bbox[0], bbox[2]) &
        centroids.y.between(bbox[1], bbox[3])
    )
 
def filter_regions(gdf: gpd.GeoDataFrame, keep=("alps", "caucasus", "andes")) -> gpd.GeoDataFrame:
    gdf = ensure_wgs84(gdf).copy()
    cent = gdf.geometry.centroid
 
    masks = {
        "alps": _in_bbox(cent, BBOX_ALPS),
        "caucasus": _in_bbox(cent, BBOX_CAUCASUS),
        "andes": _in_bbox(cent, BBOX_ANDES),
    }
 
    keep_mask = np.zeros(len(gdf), dtype=bool)
    for r in keep:
        keep_mask |= masks[r].to_numpy()
 
    out = gdf.loc[keep_mask].copy()
 
    out["region"] = "other"
    for r in keep:
        out.loc[masks[r].loc[out.index], "region"] = r
 
    return out
 
def filter_years_for_satellite(
    gdf: gpd.GeoDataFrame,
    min_year: int = 2016,
    max_year: int = 2026,
    require_date: bool = True,
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
 
    if "src_date_dt" not in gdf.columns:
        return gdf if not require_date else gdf.iloc[0:0].copy()
 
    if require_date:
        gdf = gdf[gdf["src_date_dt"].notna()].copy()
 
    years = gdf["src_date_dt"].dt.year
    return gdf[years.between(min_year, max_year)].copy()
 
 
def keep_useful_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = [c for c in [
        "glac_id",
        "src_date",
        "src_date_dt",
        "area",
        "region",
        "geometry",
    ] if c in gdf.columns]
    return gdf[cols].copy()
 
 
def clean_glims_for_satellite(
    gdf: gpd.GeoDataFrame,
    min_year: int = 2016,
    max_year: int = 2026,
    keep_regions=("alps", "caucasus", "andes"),
) -> gpd.GeoDataFrame:
    gdf = keep_outlines(gdf)
    gdf = drop_empty_geometries(gdf)
    gdf = ensure_wgs84(gdf)
    gdf = fix_invalid_geometries(gdf)
    gdf = parse_src_date(gdf)
    gdf = filter_positive_area(gdf)
    gdf = filter_regions(gdf, keep=keep_regions)
    gdf = filter_years_for_satellite(gdf, min_year=min_year, max_year=max_year)
    gdf = drop_exact_dupes(gdf)
    gdf = keep_useful_columns(gdf)
    return gdf.reset_index(drop=True)
 
def compute_iou(geom_a, geom_b) -> float:
    """
    Calcule l'Intersection over Union (IoU) entre deux géométries Shapely.
 
    IoU = aire(intersection) / aire(union)
 
    Retourne 0.0 si l'une des géométries est vide ou invalide.
    Utilisé à la fois pour la déduplication des outlines GLIMS et pour
    évaluer la précision du modèle de segmentation (en comparant les
    masques prédits aux masques de référence).
    """
    try:
        inter = geom_a.intersection(geom_b).area
        if inter == 0.0:
            return 0.0
        union = geom_a.union(geom_b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0
 
 
def deduplicate_by_iou(
    gdf: gpd.GeoDataFrame,
    iou_thresh: float = 0.5,
) -> gpd.GeoDataFrame:
    """
    Supprime les doublons géométriques dans un GeoDataFrame GLIMS.
 
    Deux outlines avec des glac_id distincts peuvent représenter le même
    glacier physique (re-digitalisation d'une campagne à l'autre). Si leur
    IoU dépasse iou_thresh, on ne garde que le plus récent.
 
    Algorithme :
    - Trier par src_date_dt décroissant (plus récent en premier).
    - Parcourir les lignes dans cet ordre ; pour chacune, vérifier via un
      index spatial si elle chevauche fortement un outline déjà accepté.
    - Si IoU > seuil avec un outline accepté → doublon géométrique, on skip.
    - Sinon → on l'accepte.
 
    Complexité : O(n * k) avec k = voisins dans l'index spatial, ce qui
    reste raisonnable sur ~70 000 outlines filtrés par région.
    """
    from shapely.strtree import STRtree
 
    g = gdf.copy()
    g = g[g["src_date_dt"].notna()].copy()
    g = g.sort_values("src_date_dt", ascending=False).reset_index(drop=True)
 
    accepted_geoms: list = []
    accepted_indices: list[int] = []
 
    # STRtree se reconstruit à chaque ajout serait trop lent ;
    # on interroge un index statique et on filtre manuellement.
    tree = STRtree(g.geometry.values)
 
    for i, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
 
        # candidats qui ont une bbox qui se chevauche
        candidate_pos = tree.query(geom)
        # parmi les candidats, seuls ceux déjà acceptés nous intéressent
        accepted_set = set(accepted_indices)
        overlapping_accepted = [p for p in candidate_pos if p in accepted_set]
 
        is_duplicate = False
        for j in overlapping_accepted:
            if compute_iou(geom, g.geometry.iloc[j]) > iou_thresh:
                is_duplicate = True
                break
 
        if not is_duplicate:
            accepted_geoms.append(geom)
            accepted_indices.append(i)
 
    return g.iloc[accepted_indices].reset_index(drop=True)
 
 
def select_most_recent_outline(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Pour chaque glacier (glac_id), garde tous les outlines de la date
    la plus récente (un glacier peut avoir plusieurs polygones à la même date).
    """
    g = gdf.copy()
    g = g[g["src_date_dt"].notna()].copy()
    g["max_date"] = g.groupby("glac_id")["src_date_dt"].transform("max")
    g = g[g["src_date_dt"] == g["max_date"]].copy()
    return g.drop(columns=["max_date"]).reset_index(drop=True)
 
 
def clean_glims_for_masks(
    gdf: gpd.GeoDataFrame,
    keep_regions=("alps", "caucasus", "andes"),
    iou_thresh: float = 0.5,
) -> gpd.GeoDataFrame:
    """
    Version pour la création de masques et le preview.
 
    Pas de filtre temporel par rapport à l'année image : on garde tous les
    glaciers de la région. On applique deux passes de déduplication :
    1. Par glac_id  → 1 outline par identifiant (le plus récent).
    2. Par IoU      → si deux glac_id distincts représentent le même glacier
                      physique (IoU > iou_thresh), on garde le plus récent.
    """
    gdf = keep_outlines(gdf)
    gdf = drop_empty_geometries(gdf)
    gdf = ensure_wgs84(gdf)
    gdf = fix_invalid_geometries(gdf)
    gdf = parse_src_date(gdf)
    gdf = filter_positive_area(gdf)
    gdf = filter_regions(gdf, keep=keep_regions)
    gdf = drop_exact_dupes(gdf)
    gdf = keep_useful_columns(gdf)
    gdf = select_most_recent_outline(gdf)      # passe 1 : 1 outline par glac_id
    gdf = deduplicate_by_iou(gdf, iou_thresh)  # passe 2 : doublons géométriques
    return gdf.reset_index(drop=True)
 
def select_outline_closest_to_year(
    gdf: gpd.GeoDataFrame,
    target_year: int,
) -> gpd.GeoDataFrame:
    g = gdf.copy()
    g = g[g["src_date_dt"].notna()].copy()
    g["src_year"] = g["src_date_dt"].dt.year
    g["gap_year"] = (g["src_year"] - target_year).abs()
 
    g = (
        g.sort_values(["glac_id", "gap_year", "src_date_dt"])
         .drop_duplicates("glac_id", keep="first")
         .copy()
    )
    return g.reset_index(drop=True)
 
def clean_glims_outlines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
 
    gdf = keep_outlines(gdf)
    gdf = drop_empty_geometries(gdf)
    gdf = ensure_crs(gdf, epsg=4326)
    gdf = fix_invalid_geometries(gdf)
 
    gdf = parse_src_date(gdf, col="src_date")
    gdf = parse_anlys_time(gdf, col="anlys_time")
 
    gdf = clean_elevations_soft(
        gdf,
        sentinel=-9999,
        set_nonpositive_to_nan=False,
        enforce_order=False,
    )
 
    gdf = filter_positive_area(gdf, col="area")
    gdf = cast_categories(gdf)
    gdf = drop_exact_dupes(gdf)
 
    return gdf.reset_index(drop=True)
 
 
def make_temporal_view(
    gdf_base: gpd.GeoDataFrame,
    min_date: str = "1900-01-01",
    max_date: str = "2026-12-31",
) -> gpd.GeoDataFrame:
    if "src_date_dt" not in gdf_base.columns:
        raise ValueError("src_date_dt missing; call clean_glims_outlines first.")
 
    g = gdf_base.copy()
    g = g.dropna(subset=["glac_id", "src_date_dt"])
    g = g[g["src_date_dt"].between(pd.Timestamp(min_date, tz="UTC"), pd.Timestamp(max_date, tz="UTC"))]
 
    out = g.dissolve(by=["glac_id", "src_date_dt"], aggfunc={"area": "median"}).reset_index()
    return out
 
 
def make_prediction_view(
    gdf_base: gpd.GeoDataFrame,
    k: int = 3,
    min_date: str = "1900-01-01",
    max_date: str = "2026-12-31",
) -> gpd.GeoDataFrame:
    temporal = make_temporal_view(gdf_base, min_date=min_date, max_date=max_date)
    counts = temporal.groupby("glac_id")["src_date_dt"].nunique()
    keep_ids = counts[counts >= k].index
    return temporal[temporal["glac_id"].isin(keep_ids)].copy()
 
 
def make_polygon_view(gdf_base: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return explode_multipolygons(gdf_base).reset_index(drop=True)