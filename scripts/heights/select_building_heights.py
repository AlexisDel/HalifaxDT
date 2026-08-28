"""
Select the best building heights for the Halifax peninsula.

This is the single height-selection pipeline. It calculates DSM/DEM evidence,
matches OSM and NSTDB alternatives, scores every candidate, and writes the final
height layer used by later work.

Output:
  data/processed_data/building_heights_selected.gpkg
    layer: buildings
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds, transform as window_transform
from shapely.geometry.base import BaseGeometry


FT_TO_M = 0.3048
NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

DEFAULT_HEIGHT_BY_FCODE = {
    "BLDG": 6.5,
    "BLDGRL": 4.0,
    "BLDGPD": 6.5,
    "STRUCTURE": 5.0,
}
FALLBACK_DEFAULT_HEIGHT_M = 6.0
INVALID_ZVALUE = 9999.0
HEIGHT_PERCENTILES = [25, 50, 75, 90, 95, 98]
PROFILE_PERCENTILES = [50, 75, 90, 95, 98]


@dataclass
class Candidate:
    """A possible height value with its source and confidence score."""

    value: float
    score: float
    source: str
    reason: str


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def data_root() -> Path:
    return Path(__file__).parents[2] / "data"


ROOT = data_root()
FOOTPRINTS_PATH = ROOT / "interim_data" / "footprints_peninsula_hrm" / "Building_Polygons_Peninsula.shp"
DSM_PATH = ROOT / "interim_data" / "dsm_peninsula_1m" / "HRM_LiDAR_DSM_2018_1m_peninsula.tif"
DEM_PATH = ROOT / "interim_data" / "dem_peninsula_5m" / "HRM_LiDAR_DEM_2018_5m_peninsula.tif"
OSM_PATH = ROOT / "interim_data" / "osm_buildings_height_levels.geojson"
NSTDB_PATH = ROOT / "interim_data" / "nstdb_peninsula" / "nstdb_peninsula.gpkg"
OUT_DIR = ROOT / "processed_data"

# Set LIMIT to a number when testing on a small subset, or keep None for all buildings.
LIMIT = None

LEVEL_HEIGHT_M = 3.2
MIN_OVERLAP_RATIO = 0.20
LOW_OSM_OVERLAP_RATIO = 0.15
MAX_NSTDB_DISTANCE_M = 10.0
MIN_VALID_HEIGHT = 2.0
MAX_VALID_HEIGHT = 150.0
MIN_DSM_PIXELS = 3
INNER_BUFFER_M = 1.0
MIN_INNER_AREA_RATIO = 0.25
AGREEMENT_ABS_M = 2.5
AGREEMENT_REL = 0.20


@dataclass(frozen=True)
class HeightSelectionConfig:
    """Fixed pipeline configuration built from the constants above."""

    footprints: Path = FOOTPRINTS_PATH
    dsm: Path = DSM_PATH
    dem: Path = DEM_PATH
    osm: Path = OSM_PATH
    nstdb: Path = NSTDB_PATH
    out_dir: Path = OUT_DIR
    limit: int | None = LIMIT
    level_height_m: float = LEVEL_HEIGHT_M
    min_overlap_ratio: float = MIN_OVERLAP_RATIO
    low_osm_overlap_ratio: float = LOW_OSM_OVERLAP_RATIO
    max_nstdb_distance_m: float = MAX_NSTDB_DISTANCE_M
    min_valid_height: float = MIN_VALID_HEIGHT
    max_valid_height: float = MAX_VALID_HEIGHT
    min_dsm_pixels: int = MIN_DSM_PIXELS
    inner_buffer_m: float = INNER_BUFFER_M
    min_inner_area_ratio: float = MIN_INNER_AREA_RATIO
    agreement_abs_m: float = AGREEMENT_ABS_M
    agreement_rel: float = AGREEMENT_REL


# -----------------------------------------------------------------------------
# Raster sampling helpers
# -----------------------------------------------------------------------------


def valid_values(data: np.ma.MaskedArray | np.ndarray, nodata: float | None) -> np.ndarray:
    """Return finite, non-nodata values from a raster read."""

    arr = data.compressed() if isinstance(data, np.ma.MaskedArray) else np.asarray(data).ravel()
    if nodata is not None:
        arr = arr[arr != nodata]
    return arr[np.isfinite(arr)]


def values_inside_geometry(src: rasterio.DatasetReader, geom: BaseGeometry) -> np.ndarray:
    """Read one raster inside a polygon footprint."""

    if geom is None or geom.is_empty:
        return np.array([], dtype="float32")
    minx, miny, maxx, maxy = geom.bounds
    window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    window = window.round_offsets().round_lengths()
    if window.width <= 0 or window.height <= 0:
        return np.array([], dtype="float32")

    data = src.read(1, window=window, masked=True)
    if data.size == 0:
        return np.array([], dtype="float32")
    transform = window_transform(window, src.transform)
    mask = geometry_mask([geom], out_shape=data.shape, transform=transform, invert=True, all_touched=True)
    masked = np.ma.array(data, mask=np.ma.getmaskarray(data) | ~mask)
    return valid_values(masked, src.nodata)


def aligned_values_inside_geometry(
    primary: rasterio.DatasetReader,
    secondary: rasterio.DatasetReader,
    geom: BaseGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Read two already-aligned rasters and return paired values inside a footprint."""

    if geom is None or geom.is_empty:
        return np.array([], dtype="float32"), np.array([], dtype="float32")
    minx, miny, maxx, maxy = geom.bounds
    window = from_bounds(minx, miny, maxx, maxy, transform=primary.transform)
    window = window.round_offsets().round_lengths()
    if window.width <= 0 or window.height <= 0:
        return np.array([], dtype="float32"), np.array([], dtype="float32")

    primary_data = primary.read(1, window=window, masked=True)
    secondary_data = secondary.read(1, window=window, masked=True)
    if primary_data.size == 0 or secondary_data.size == 0:
        return np.array([], dtype="float32"), np.array([], dtype="float32")
    if primary_data.shape != secondary_data.shape:
        return np.array([], dtype="float32"), np.array([], dtype="float32")

    transform = window_transform(window, primary.transform)
    inside = geometry_mask([geom], out_shape=primary_data.shape, transform=transform, invert=True, all_touched=True)
    primary_values = np.asarray(primary_data, dtype="float64")
    secondary_values = np.asarray(secondary_data, dtype="float64")
    valid = inside & ~np.ma.getmaskarray(primary_data) & ~np.ma.getmaskarray(secondary_data)
    valid &= np.isfinite(primary_values) & np.isfinite(secondary_values)
    if primary.nodata is not None:
        valid &= primary_values != primary.nodata
    if secondary.nodata is not None:
        valid &= secondary_values != secondary.nodata

    return primary_values[valid], secondary_values[valid]


def sample_at_representative_point(src: rasterio.DatasetReader, geom: BaseGeometry) -> float | None:
    """Fallback raster sample for cases where polygon sampling returns no values."""

    point = geom.representative_point()
    try:
        value = next(src.sample([(point.x, point.y)]))[0]
    except (StopIteration, IndexError, ValueError):
        return None
    if src.nodata is not None and value == src.nodata:
        return None
    return float(value) if np.isfinite(value) else None


def inner_geometry(geom: BaseGeometry, buffer_m: float, min_area_ratio: float) -> BaseGeometry | None:
    """Shrink a footprint to reduce edge contamination from walls, trees, or neighbours."""

    if buffer_m <= 0 or geom.is_empty or geom.area <= 0:
        return None
    inner = geom.buffer(-buffer_m)
    if inner.is_empty or inner.area <= 0:
        return None
    return inner if inner.area / geom.area >= min_area_ratio else None


def percentile_map(values: np.ndarray, percentiles: list[int]) -> dict[int, float]:
    """Calculate named percentiles while preserving NaN for empty samples."""

    return {
        percentile: float(np.percentile(values, percentile)) if values.size else np.nan
        for percentile in percentiles
    }


# -----------------------------------------------------------------------------
# DSM / DEM evidence
# -----------------------------------------------------------------------------


def calculate_dsm_dem_evidence(buildings: gpd.GeoDataFrame, args: HeightSelectionConfig) -> pd.DataFrame:
    """Calculate height percentiles from local DSM - DEM values for each footprint."""

    records: list[dict[str, object]] = []
    with rasterio.open(args.dsm) as dsm, rasterio.open(args.dem) as dem:
        if dsm.crs != dem.crs:
            raise ValueError(f"DSM CRS ({dsm.crs}) does not match DEM CRS ({dem.crs})")

        # The DEM is 5 m and the DSM is 1 m. WarpedVRT lets us sample the DEM
        # on the DSM grid without creating a large intermediate raster on disk.
        with WarpedVRT(
            dem,
            crs=dsm.crs,
            transform=dsm.transform,
            width=dsm.width,
            height=dsm.height,
            resampling=Resampling.bilinear,
        ) as dem_on_dsm:
            buildings_calc = buildings.to_crs(dsm.crs)
            for i, (_, row) in enumerate(buildings_calc.iterrows(), start=1):
                geom = row.geometry
                dsm_values, dem_values = aligned_values_inside_geometry(dsm, dem_on_dsm, geom)

                dem_source = "dsm_grid_interpolated"
                if dem_values.size == 0:
                    # Rare fallback for footprints that miss the DEM grid/window.
                    dsm_values = values_inside_geometry(dsm, geom)
                    sample = sample_at_representative_point(dem, geom)
                    dem_values = np.array([sample], dtype="float64") if sample is not None else np.array([])
                    dem_source = "representative_point" if sample is not None else "missing"

                dem_median = float(np.median(dem_values)) if dem_values.size else np.nan
                ground_ref = dem_median

                # Height evidence is computed from local nDSM pixels, not from
                # DSM percentile minus one ground value for the whole footprint.
                ndsm_values = dsm_values - dem_values if dsm_values.size == dem_values.size else np.array([], dtype="float64")
                if ndsm_values.size == 0 and dsm_values.size and np.isfinite(ground_ref):
                    ndsm_values = dsm_values - ground_ref
                    dem_source = f"{dem_source}_constant"

                dsm_p = percentile_map(dsm_values, HEIGHT_PERCENTILES)
                height_p_raw = percentile_map(ndsm_values, HEIGHT_PERCENTILES)
                inner = inner_geometry(geom, args.inner_buffer_m, args.min_inner_area_ratio)
                if inner is not None:
                    inner_values, inner_dem_values = aligned_values_inside_geometry(dsm, dem_on_dsm, inner)
                else:
                    inner_values = np.array([], dtype="float32")
                    inner_dem_values = np.array([], dtype="float32")
                inner_ndsm_values = (
                    inner_values - inner_dem_values
                    if inner_values.size == inner_dem_values.size
                    else np.array([], dtype="float64")
                )
                inner_p = percentile_map(inner_values, HEIGHT_PERCENTILES)
                inner_height_p_raw = percentile_map(inner_ndsm_values, HEIGHT_PERCENTILES)

                height_p = {f"height_p{p:02d}_m": height_p_raw[p] for p in HEIGHT_PERCENTILES}
                inner_height_p = {f"inner_height_p{p:02d}_m": inner_height_p_raw[p] for p in HEIGHT_PERCENTILES}
                roof_elevation_p = {f"roof_elevation_p{p:02d}_m": dsm_p[p] for p in HEIGHT_PERCENTILES}
                inner_roof_elevation_p = {f"inner_roof_elevation_p{p:02d}_m": inner_p[p] for p in HEIGHT_PERCENTILES}

                height_m = height_p["height_p95_m"]
                status, confidence = classify_dsm_height(
                    height_m,
                    dsm_values.size,
                    dem_values.size,
                    args.min_dsm_pixels,
                    args.min_valid_height,
                    args.max_valid_height,
                )

                record = {
                    "height_m": height_m,
                    "height_source": "DSM_DEM_NDSM" if status == "ok" else "",
                    "height_confidence": confidence,
                    "height_status": status,
                    "ground_ref_m": ground_ref,
                    "ground_ref_source": dem_source,
                    "dsm_pixel_count": int(dsm_values.size),
                    "inner_dsm_pixel_count": int(inner_values.size),
                    "inner_area_ratio": float(inner.area / geom.area) if inner is not None and geom.area > 0 else np.nan,
                    "dem_pixel_count": int(dem_values.size),
                    "inner_dem_pixel_count": int(inner_dem_values.size),
                    "footprint_area_m2": float(geom.area),
                    **roof_elevation_p,
                    **height_p,
                    **inner_roof_elevation_p,
                    **inner_height_p,
                }
                records.append(record)
                if i % 1000 == 0 or i == len(buildings_calc):
                    print(f"DSM/DEM evidence: {i:,} / {len(buildings_calc):,}")
    return pd.DataFrame.from_records(records)


def classify_dsm_height(
    height: float,
    dsm_pixels: int,
    dem_pixels: int,
    min_dsm_pixels: int,
    min_height: float,
    max_height: float,
) -> tuple[str, str]:
    """Classify raw DSM/DEM evidence before comparing it with external sources."""

    if not np.isfinite(height):
        return "missing", "none"
    if dsm_pixels < min_dsm_pixels:
        return "suspect_too_few_dsm_pixels", "low"
    if dem_pixels == 0:
        return "suspect_no_dem_pixels", "low"
    if height < min_height:
        return "suspect_too_low", "low"
    if height > max_height:
        return "suspect_too_high", "low"
    return "ok", "high"


# -----------------------------------------------------------------------------
# OSM height parsing and matching
# -----------------------------------------------------------------------------


def parse_meters(value) -> float | None:
    """Parse OSM height-like values, including feet."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    match = NUMBER_RE.search(text.replace(",", "."))
    if not match:
        return None
    number = float(match.group(0))
    if "ft" in text or "feet" in text:
        number *= FT_TO_M
    return number if number > 0 else None


def parse_levels(value) -> float | None:
    """Parse OSM building:levels values, keeping the highest value if multiple exist."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower().replace(",", ".")
    if ";" in text:
        levels = [parse_levels(part) for part in text.split(";")]
        levels = [level for level in levels if level is not None]
        return max(levels) if levels else None
    match = NUMBER_RE.search(text)
    if not match:
        return None
    levels = float(match.group(0))
    return levels if levels > 0 else None


def osm_height(row: pd.Series, level_height_m: float) -> tuple[float | None, str]:
    """Convert OSM height tags to metres.

    OSM height is treated as total building height. roof:height is only added
    when height is estimated from building:levels.
    """

    height = parse_meters(row.get("height"))
    roof_height = parse_meters(row.get("roof:height")) or 0.0
    if height is not None:
        return height, "osm_height"
    levels = parse_levels(row.get("building:levels"))
    if levels is not None:
        return levels * level_height_m + roof_height, "osm_levels"
    return None, ""


def add_osm_matches(buildings: gpd.GeoDataFrame, args: HeightSelectionConfig) -> gpd.GeoDataFrame:
    """Attach the best OSM height candidate to each HRM footprint."""

    out = buildings.copy().reset_index(drop=True)
    for column in [
        "osm_idx",
        "osm_id",
        "osm_height_m",
        "osm_height_source",
        "osm_building",
        "osm_levels",
        "osm_name",
        "osm_overlap_m2",
        "osm_overlap_ratio",
        "osm_overlap_ratio_osm",
        "osm_geometry_score",
        "osm_area_m2",
        "osm_hrm_match_count",
        "osm_match_quality",
    ]:
        out[column] = "" if column in {"osm_id", "osm_height_source", "osm_building", "osm_levels", "osm_name", "osm_match_quality"} else np.nan

    if not args.osm.exists():
        return out

    osm = gpd.read_file(args.osm)
    osm = osm[osm.geometry.notna() & osm.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    for column in ["@id", "building", "building:levels", "name", "height", "roof:height"]:
        if column not in osm.columns:
            osm[column] = ""
    heights = osm.apply(lambda row: osm_height(row, args.level_height_m), axis=1)
    osm["osm_height_m"] = [item[0] for item in heights]
    osm["osm_height_source"] = [item[1] for item in heights]
    osm["osm_id"] = osm["@id"]
    osm = osm[osm["osm_height_m"].notna()].copy()
    if osm.empty:
        return out

    metric_crs = osm.estimate_utm_crs()
    buildings_m = out.to_crs(metric_crs).reset_index(drop=False).rename(columns={"index": "hrm_idx"})
    osm_m = osm.to_crs(metric_crs).reset_index(drop=False).rename(columns={"index": "osm_idx"})
    buildings_m["hrm_area_m2"] = buildings_m.geometry.area
    osm_m["osm_area_m2"] = osm_m.geometry.area
    candidates = gpd.sjoin(
        buildings_m[["hrm_idx", "BL_ID", "hrm_area_m2", "geometry"]],
        osm_m[["osm_idx", "osm_id", "osm_height_m", "osm_height_source", "building", "building:levels", "name", "osm_area_m2", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if candidates.empty:
        return out

    osm_geom = osm_m.set_index("osm_idx").geometry
    candidates["osm_overlap_m2"] = [
        row.geometry.intersection(osm_geom.loc[row["osm_idx"]]).area
        for _, row in candidates.iterrows()
    ]
    candidates["osm_overlap_ratio"] = candidates["osm_overlap_m2"] / candidates["hrm_area_m2"]
    candidates["osm_overlap_ratio_osm"] = candidates["osm_overlap_m2"] / candidates["osm_area_m2"]
    candidates = candidates[candidates["osm_overlap_ratio"] >= args.min_overlap_ratio].copy()

    # A good OSM match should cover the HRM footprint and also represent most
    # of the OSM polygon. This prevents one broad OSM block from dominating
    # several smaller HRM footprints.
    candidates["osm_geometry_score"] = (
        0.60 * candidates["osm_overlap_ratio"].clip(upper=1.0)
        + 0.40 * candidates["osm_overlap_ratio_osm"].clip(upper=1.0)
    )
    match_counts = candidates.groupby("osm_idx")["hrm_idx"].nunique().rename("osm_hrm_match_count")
    candidates = candidates.join(match_counts, on="osm_idx")
    candidates["osm_match_quality"] = np.where(
        candidates["osm_overlap_ratio_osm"] < args.low_osm_overlap_ratio,
        "broad_osm_polygon",
        np.where(candidates["osm_hrm_match_count"] > 1, "multi_hrm_match", "good"),
    )
    candidates = candidates.sort_values(
        ["hrm_idx", "osm_geometry_score", "osm_overlap_ratio", "osm_overlap_m2"],
        ascending=[True, False, False, False],
    )
    matches = candidates.drop_duplicates("hrm_idx").set_index("hrm_idx")

    out.loc[matches.index, "osm_idx"] = matches["osm_idx"]
    out.loc[matches.index, "osm_id"] = matches["osm_id"].fillna("")
    out.loc[matches.index, "osm_height_m"] = matches["osm_height_m"]
    out.loc[matches.index, "osm_height_source"] = matches["osm_height_source"]
    out.loc[matches.index, "osm_building"] = matches["building"].fillna("")
    out.loc[matches.index, "osm_levels"] = matches["building:levels"].fillna("")
    out.loc[matches.index, "osm_name"] = matches["name"].fillna("")
    out.loc[matches.index, "osm_overlap_m2"] = matches["osm_overlap_m2"]
    out.loc[matches.index, "osm_overlap_ratio"] = matches["osm_overlap_ratio"]
    out.loc[matches.index, "osm_overlap_ratio_osm"] = matches["osm_overlap_ratio_osm"]
    out.loc[matches.index, "osm_geometry_score"] = matches["osm_geometry_score"]
    out.loc[matches.index, "osm_area_m2"] = matches["osm_area_m2"]
    out.loc[matches.index, "osm_hrm_match_count"] = matches["osm_hrm_match_count"]
    out.loc[matches.index, "osm_match_quality"] = matches["osm_match_quality"]
    print(f"OSM matches: {len(matches):,}")
    return out


# -----------------------------------------------------------------------------
# NSTDB height matching
# -----------------------------------------------------------------------------


def add_nstdb_matches(buildings: gpd.GeoDataFrame, args: HeightSelectionConfig) -> gpd.GeoDataFrame:
    """Attach the best NSTDB height candidate to each HRM footprint."""

    out = buildings.copy().reset_index(drop=True)
    for column in [
        "nstdb_idx",
        "nstdb_feat_code",
        "nstdb_feat_desc",
        "nstdb_zvalue_m",
        "nstdb_dem_m",
        "nstdb_height_m",
        "nstdb_geom_type",
        "nstdb_match_distance_m",
        "nstdb_match_method",
        "nstdb_overlap_m2",
        "nstdb_overlap_ratio",
    ]:
        out[column] = "" if column in {"nstdb_feat_code", "nstdb_feat_desc", "nstdb_geom_type", "nstdb_match_method"} else np.nan

    if not args.nstdb.exists():
        return out

    points = gpd.read_file(args.nstdb, layer="nstdb_points")
    points["nstdb_geom_type"] = "point"
    polygons = gpd.read_file(args.nstdb, layer="nstdb_polygons")
    polygons["nstdb_geom_type"] = "polygon"
    nstdb = pd.concat([points, polygons], ignore_index=True)
    nstdb = gpd.GeoDataFrame(nstdb, geometry="geometry", crs=points.crs)
    nstdb = nstdb[nstdb.geometry.notna()].copy()
    nstdb["ZVALUE"] = pd.to_numeric(nstdb["ZVALUE"], errors="coerce")
    nstdb = nstdb[nstdb["ZVALUE"].notna() & (nstdb["ZVALUE"] != INVALID_ZVALUE)].copy()

    with rasterio.open(args.dem) as dem:
        nstdb_dem = nstdb.to_crs(dem.crs)
        dem_values = [sample_at_representative_point(dem, geom) for geom in nstdb_dem.geometry]

    nstdb["nstdb_dem_m"] = dem_values
    nstdb["nstdb_height_m"] = nstdb["ZVALUE"] - nstdb["nstdb_dem_m"]
    nstdb = nstdb[
        nstdb["nstdb_height_m"].between(args.min_valid_height, args.max_valid_height, inclusive="both")
    ].copy()
    if nstdb.empty:
        return out
    nstdb = nstdb.reset_index(drop=False).rename(columns={"index": "nstdb_idx"})

    metric_crs = out.estimate_utm_crs()
    buildings_m = out.to_crs(metric_crs).reset_index(drop=False).rename(columns={"index": "hrm_idx"})
    buildings_m["hrm_area_m2"] = buildings_m.geometry.area
    nstdb_m = nstdb.to_crs(metric_crs)
    nstdb_m["nstdb_area_m2"] = nstdb_m.geometry.area
    nstdb_columns = [
        "nstdb_idx",
        "FEAT_CODE",
        "FEAT_DESC",
        "ZVALUE",
        "nstdb_dem_m",
        "nstdb_height_m",
        "nstdb_geom_type",
        "geometry",
    ]

    polygon_matches = gpd.GeoDataFrame()
    nstdb_polygons = nstdb_m[nstdb_m.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if not nstdb_polygons.empty:
        # Polygon intersections are stronger than nearest-neighbour matches:
        # they give geometric evidence that the NSTDB feature is the same object.
        polygon_candidates = gpd.sjoin(
            buildings_m[["hrm_idx", "BL_ID", "hrm_area_m2", "geometry"]],
            nstdb_polygons[nstdb_columns],
            how="inner",
            predicate="intersects",
        )
        if not polygon_candidates.empty:
            nstdb_geom = nstdb_polygons.set_index("nstdb_idx").geometry
            polygon_candidates["nstdb_overlap_m2"] = [
                row.geometry.intersection(nstdb_geom.loc[row["nstdb_idx"]]).area
                for _, row in polygon_candidates.iterrows()
            ]
            polygon_candidates["nstdb_overlap_ratio"] = polygon_candidates["nstdb_overlap_m2"] / polygon_candidates["hrm_area_m2"]
            polygon_candidates["nstdb_match_distance_m"] = 0.0
            polygon_candidates["nstdb_match_method"] = "polygon_intersection"
            polygon_matches = (
                polygon_candidates.sort_values(
                    ["hrm_idx", "nstdb_overlap_ratio", "nstdb_overlap_m2"],
                    ascending=[True, False, False],
                )
                .drop_duplicates("hrm_idx")
                .set_index("hrm_idx")
            )

    remaining = buildings_m if polygon_matches.empty else buildings_m[~buildings_m["hrm_idx"].isin(polygon_matches.index)]
    nearest_matches = gpd.GeoDataFrame()
    if not remaining.empty:
        # Nearest is kept as a fallback, with a short default radius to avoid
        # borrowing heights from neighbouring buildings in dense areas.
        nearest = gpd.sjoin_nearest(
            remaining[["hrm_idx", "BL_ID", "geometry"]],
            nstdb_m[nstdb_columns],
            how="left",
            max_distance=args.max_nstdb_distance_m,
            distance_col="nstdb_match_distance_m",
        )
        nearest = nearest[nearest["nstdb_idx"].notna()].copy()
        if not nearest.empty:
            nearest["nstdb_match_method"] = "nearest"
            nearest["nstdb_overlap_m2"] = np.nan
            nearest["nstdb_overlap_ratio"] = np.nan
            nearest_matches = (
                nearest.sort_values(["hrm_idx", "nstdb_match_distance_m"])
                .drop_duplicates("hrm_idx")
                .set_index("hrm_idx")
            )

    matches = pd.concat([polygon_matches, nearest_matches], axis=0, sort=False)
    if matches.empty:
        return out

    out.loc[matches.index, "nstdb_idx"] = matches["nstdb_idx"]
    out.loc[matches.index, "nstdb_feat_code"] = matches["FEAT_CODE"].fillna("")
    out.loc[matches.index, "nstdb_feat_desc"] = matches["FEAT_DESC"].fillna("")
    out.loc[matches.index, "nstdb_zvalue_m"] = matches["ZVALUE"]
    out.loc[matches.index, "nstdb_dem_m"] = matches["nstdb_dem_m"]
    out.loc[matches.index, "nstdb_height_m"] = matches["nstdb_height_m"]
    out.loc[matches.index, "nstdb_geom_type"] = matches["nstdb_geom_type"]
    out.loc[matches.index, "nstdb_match_distance_m"] = matches["nstdb_match_distance_m"]
    out.loc[matches.index, "nstdb_match_method"] = matches["nstdb_match_method"]
    out.loc[matches.index, "nstdb_overlap_m2"] = matches["nstdb_overlap_m2"]
    out.loc[matches.index, "nstdb_overlap_ratio"] = matches["nstdb_overlap_ratio"]
    print(f"NSTDB matches: {len(matches):,}")
    return out


# -----------------------------------------------------------------------------
# Candidate scoring utilities
# -----------------------------------------------------------------------------


def numeric(row: pd.Series, column: str) -> float:
    """Read a numeric row value while tolerating missing columns/empty strings."""

    try:
        return float(row.get(column, np.nan))
    except (TypeError, ValueError):
        return np.nan


def clip_score(score: float) -> float:
    """Keep confidence scores inside an open 0-1 range."""

    return float(min(0.98, max(0.02, score)))


def is_plausible(value: float, min_height: float, max_height: float) -> bool:
    """Check whether a candidate building height is physically plausible."""

    return np.isfinite(value) and min_height <= value <= max_height


def close_enough(a: float, b: float, args: HeightSelectionConfig) -> bool:
    """Return True when two sources agree within absolute or relative tolerance."""

    if not np.isfinite(a) or not np.isfinite(b):
        return False
    tolerance = max(args.agreement_abs_m, args.agreement_rel * np.nanmedian([abs(a), abs(b)]))
    return abs(a - b) <= tolerance


def conflict_level(a: float, b: float, args: HeightSelectionConfig) -> tuple[float, str]:
    """Label disagreement between DSM and external evidence."""

    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan, "none"
    diff = abs(a - b)
    if diff <= max(args.agreement_abs_m, args.agreement_rel * np.nanmedian([abs(a), abs(b)])):
        return diff, "none"
    if diff >= max(6.0, 0.40 * np.nanmedian([abs(a), abs(b)])):
        return diff, "strong"
    return diff, "moderate"


def weighted_median(values: list[float], weights: list[float]) -> float:
    """Choose a robust central value without averaging contradictory heights."""

    pairs = sorted([(v, w) for v, w in zip(values, weights) if np.isfinite(v) and w > 0])
    if not pairs:
        return np.nan
    midpoint = sum(weight for _, weight in pairs) / 2.0
    total = 0.0
    for value, weight in pairs:
        total += weight
        if total >= midpoint:
            return float(value)
    return float(pairs[-1][0])


def confidence_label(score: float) -> str:
    """Convert a numeric score to a compact label for the output layer."""

    if score >= 0.82:
        return "high"
    if score >= 0.65:
        return "medium_high"
    if score >= 0.45:
        return "medium"
    return "low"


def default_height(fcode) -> float:
    """Fallback height when DSM, OSM, and NSTDB are unusable."""

    return DEFAULT_HEIGHT_BY_FCODE.get(str(fcode).strip(), FALLBACK_DEFAULT_HEIGHT_M)


def percentile_value(row: pd.Series, base_column: str, percentile: int, prefer_inner: bool) -> float:
    """Read a full-footprint or inner-footprint percentile from a row."""

    prefixes = [f"inner_{base_column}", base_column] if prefer_inner else [base_column, f"inner_{base_column}"]
    for prefix in prefixes:
        value = numeric(row, f"{prefix}_p{percentile:02d}_m")
        if np.isfinite(value):
            return value
    return numeric(row, "height_m") if base_column == "height" and percentile == 95 else np.nan


# -----------------------------------------------------------------------------
# Source-specific candidates
# -----------------------------------------------------------------------------


def dsm_profile(row: pd.Series, args: HeightSelectionConfig) -> dict[str, object]:
    """Build the DSM candidate profile used by the final selector."""

    inner_pixels = numeric(row, "inner_dsm_pixel_count")
    inner_ratio = numeric(row, "inner_area_ratio")
    inner_height = numeric(row, "inner_height_p95_m")
    prefer_inner = (
        np.isfinite(inner_pixels)
        and inner_pixels >= 3
        and np.isfinite(inner_ratio)
        and inner_ratio >= 0.25
        and np.isfinite(inner_height)
    )
    h = {p: percentile_value(row, "height", p, prefer_inner) for p in PROFILE_PERCENTILES}
    z = {p: percentile_value(row, "roof_elevation", p, prefer_inner) for p in PROFILE_PERCENTILES}
    has_distribution = all(np.isfinite(h[p]) for p in h)

    # The profile class decides whether p95 is a good roof height or whether
    # the distribution suggests a high tail, mixed volumes, or edge pollution.
    if not is_plausible(h[95], args.min_valid_height, args.max_valid_height):
        shape = "dsm_implausible"
    elif not has_distribution:
        shape = "limited_dsm_stats"
    else:
        spread_50_98 = h[98] - h[50]
        gap_75_95 = h[95] - h[75]
        gap_90_98 = h[98] - h[90]
        if spread_50_98 <= max(3.0, 0.20 * h[95]) and gap_75_95 <= max(2.0, 0.12 * h[95]):
            shape = "stable_full_roof"
        elif gap_75_95 > max(4.0, 0.35 * h[95]) or gap_90_98 > max(5.0, 0.30 * h[95]):
            shape = "high_tail_or_multi_volume"
        elif h[95] - h[50] > max(5.0, 0.40 * h[95]):
            shape = "variable_roof"
        else:
            shape = "moderately_stable_roof"

    dominant = h[75] if shape in {"high_tail_or_multi_volume", "variable_roof"} else h[95]
    dominant_roof_elevation = z[75] if shape in {"high_tail_or_multi_volume", "variable_roof"} else z[95]
    if not is_plausible(dominant, args.min_valid_height, args.max_valid_height):
        dominant = h[95]
        dominant_roof_elevation = z[95]

    score = 0.72 if str(row.get("height_status", "")) == "ok" else 0.38
    pixels = numeric(row, "dsm_pixel_count")
    if np.isfinite(pixels):
        score += 0.10 if pixels >= 30 else 0.05 if pixels >= 10 else -0.20 if pixels < 3 else 0.0
    if prefer_inner:
        score += 0.05
    if str(row.get("ground_ref_source", "")) == "representative_point":
        score -= 0.05
    if not is_plausible(h[95], args.min_valid_height, args.max_valid_height):
        score -= 0.45
    elif shape == "stable_full_roof":
        score += 0.12
    elif shape == "moderately_stable_roof":
        score += 0.05
    elif shape == "high_tail_or_multi_volume":
        score -= 0.20
    elif shape == "variable_roof":
        score -= 0.12

    return {
        "roof": h[95],
        "roof_elevation": z[95],
        "dominant": dominant,
        "dominant_roof_elevation": dominant_roof_elevation,
        "score": clip_score(score),
        "shape": shape,
        "sampling": "inner_footprint" if prefer_inner else "full_footprint",
        **{f"h{p}": h[p] for p in h},
        **{f"z{p}": z[p] for p in z},
    }


def osm_candidate(row: pd.Series, args: HeightSelectionConfig) -> Candidate | None:
    """Score the matched OSM height, if it is plausible."""

    value = numeric(row, "osm_height_m")
    if not is_plausible(value, args.min_valid_height, args.max_valid_height):
        return None
    source = str(row.get("osm_height_source", ""))
    score = 0.66 if source == "osm_height" else 0.42 if source == "osm_levels" else 0.36
    reason_parts = ["OSM explicit height tag"] if source == "osm_height" else ["OSM building levels estimate"]
    overlap = numeric(row, "osm_overlap_ratio")
    if np.isfinite(overlap):
        score += min(0.18, max(0.0, (overlap - 0.20) / 0.80 * 0.18))
        if overlap < 0.35:
            score -= 0.08
            reason_parts.append("low HRM overlap")
    geometry_score = numeric(row, "osm_geometry_score")
    if np.isfinite(geometry_score):
        score += min(0.08, max(-0.08, (geometry_score - 0.50) * 0.16))
    osm_overlap = numeric(row, "osm_overlap_ratio_osm")
    if np.isfinite(osm_overlap) and osm_overlap < args.low_osm_overlap_ratio:
        score -= 0.18
        reason_parts.append("broad OSM polygon")
    match_count = numeric(row, "osm_hrm_match_count")
    if np.isfinite(match_count) and match_count > 1:
        score -= min(0.12, 0.03 * (match_count - 1))
        reason_parts.append("OSM polygon matches multiple HRM footprints")
    return Candidate(value, clip_score(score), "OSM", "; ".join(reason_parts))


def nstdb_candidate(row: pd.Series, args: HeightSelectionConfig) -> Candidate | None:
    """Score the matched NSTDB height, if it is plausible."""

    value = numeric(row, "nstdb_height_m")
    if not is_plausible(value, args.min_valid_height, args.max_valid_height):
        return None
    geom_type = str(row.get("nstdb_geom_type", ""))
    method = str(row.get("nstdb_match_method", ""))
    reason_parts = [f"NSTDB {geom_type or 'feature'} height"]
    score = 0.72 if method == "polygon_intersection" else 0.60 if geom_type == "polygon" else 0.55
    overlap = numeric(row, "nstdb_overlap_ratio")
    if method == "polygon_intersection" and np.isfinite(overlap):
        score += min(0.10, max(0.0, overlap * 0.10))
        if overlap < 0.35:
            score -= 0.08
            reason_parts.append("low polygon overlap")
    distance = numeric(row, "nstdb_match_distance_m")
    if np.isfinite(distance):
        if method != "polygon_intersection":
            score += max(0.0, 0.10 * (1.0 - distance / max(args.max_nstdb_distance_m, 1.0)))
            score -= min(0.25, max(0.0, distance / max(args.max_nstdb_distance_m, 1.0) * 0.25))
        if distance <= 3.0:
            score += 0.05
    return Candidate(value, clip_score(score), "NSTDB", "; ".join(reason_parts))


def external_candidate(osm: Candidate | None, nstdb: Candidate | None, args: HeightSelectionConfig) -> Candidate | None:
    """Return the best external candidate, or a consensus when OSM/NSTDB agree."""

    candidates = [candidate for candidate in [osm, nstdb] if candidate is not None]
    if not candidates:
        return None
    if osm is not None and nstdb is not None and close_enough(osm.value, nstdb.value, args):
        value = weighted_median([osm.value, nstdb.value], [osm.score, nstdb.score])
        return Candidate(value, clip_score(max(osm.score, nstdb.score) + 0.14), "OSM_NSTDB_CONSENSUS", "OSM and NSTDB agree")
    return max(candidates, key=lambda candidate: candidate.score)


def estimate_roof_elevation(selected: Candidate, dsm: dict[str, object], ground_ref: float) -> tuple[float, str]:
    """Estimate absolute roof altitude for later 3D scene generation."""

    if selected.source == "DSM_P95" and np.isfinite(float(dsm["roof_elevation"])):
        return float(dsm["roof_elevation"]), "dsm_roof_elevation_p95"
    if selected.source == "DSM_DOMINANT" and np.isfinite(float(dsm["dominant_roof_elevation"])):
        return float(dsm["dominant_roof_elevation"]), "dsm_dominant_roof_elevation"
    if np.isfinite(ground_ref) and np.isfinite(selected.value):
        return float(ground_ref + selected.value), "ground_ref_plus_selected_height"
    return np.nan, "missing"


# -----------------------------------------------------------------------------
# Final decision
# -----------------------------------------------------------------------------


def choose_height(row: pd.Series, args: HeightSelectionConfig) -> dict[str, object]:
    """Choose one final height from DSM, OSM, NSTDB, and fallback evidence."""

    dsm = dsm_profile(row, args)
    osm = osm_candidate(row, args)
    nstdb = nstdb_candidate(row, args)
    external = external_candidate(osm, nstdb, args)
    fallback = Candidate(default_height(row.get("FCODE")), 0.16, "DEFAULT_FCODE", "default by HRM FCODE")

    dsm_roof = Candidate(dsm["roof"], dsm["score"], "DSM_P95", f"{dsm['sampling']} {dsm['shape']}")
    dsm_dominant = Candidate(dsm["dominant"], max(0.02, dsm["score"] - 0.06), "DSM_DOMINANT", f"dominant DSM height from {dsm['shape']}")
    selected = fallback
    flag = "fallback"

    dsm_valid = is_plausible(dsm_roof.value, args.min_valid_height, args.max_valid_height)
    external_close_to_roof = external is not None and close_enough(external.value, dsm_roof.value, args)
    external_close_to_dominant = external is not None and close_enough(external.value, dsm_dominant.value, args)
    _, external_conflict_level = (
        conflict_level(external.value, dsm_roof.value, args) if external is not None else (np.nan, "none")
    )

    # Stable DSM is trusted most, but strong external disagreement is preserved
    # as a lower-confidence conflict rather than averaged away.
    if dsm_valid and dsm["shape"] == "stable_full_roof" and dsm_roof.score >= 0.78:
        selected = dsm_roof
        flag = "dsm_stable"
        if external is not None and not external_close_to_roof:
            score_cap = 0.68 if external_conflict_level == "strong" and external.score >= 0.65 else 0.78
            selected = Candidate(selected.value, min(selected.score, score_cap), selected.source, f"{selected.reason}; external sources disagree")
            flag = "strong_source_conflict_keep_stable_dsm" if external_conflict_level == "strong" else "source_conflict_keep_stable_dsm"
    elif dsm_valid and dsm["shape"] in {"high_tail_or_multi_volume", "variable_roof"}:
        # For mixed-height footprints, prefer the dominant DSM height when
        # external data agrees with it; otherwise avoid blindly using p95.
        if external is not None and external_close_to_dominant:
            value = weighted_median([dsm_dominant.value, external.value], [dsm_dominant.score, external.score])
            selected = Candidate(value, clip_score(max(dsm_dominant.score, external.score) + 0.08), "CONSENSUS_DOMINANT", "DSM dominant height agrees with external source")
            flag = "multi_volume_consensus_dominant"
        elif external is not None and external_close_to_roof:
            selected = Candidate(dsm_roof.value, clip_score(dsm_roof.score + 0.06), dsm_roof.source, f"{dsm_roof.reason}; confirmed by external source")
            flag = "dsm_roof_confirmed"
        elif dsm_dominant.score >= 0.50:
            selected = dsm_dominant
            flag = "multi_volume_dsm_dominant"
        elif external is not None:
            selected = external
            flag = "external_replaces_weak_dsm"
    elif dsm_valid and dsm_roof.score >= 0.55:
        if external is not None and not external_close_to_roof and external.score > dsm_roof.score + 0.15:
            selected = external
            flag = "external_replaces_unstable_dsm"
        else:
            selected = dsm_roof
            flag = "dsm_selected"
    elif external is not None and external.score >= 0.40:
        selected = external
        flag = "external_replaces_invalid_dsm"

    ground_ref = numeric(row, "ground_ref_m")
    roof_elevation, roof_elevation_method = estimate_roof_elevation(selected, dsm, ground_ref)

    return {
        "final_height_m": selected.value,
        "ground_ref_m": ground_ref,
        "roof_elevation_m": roof_elevation,
        "roof_elevation_method": roof_elevation_method,
        "final_height_source": selected.source,
        "final_height_confidence": confidence_label(selected.score),
        "height_selection_flag": flag,
        "external_dsm_conflict": external_conflict_level,
        "candidate_osm_m": osm.value if osm is not None else np.nan,
        "candidate_nstdb_m": nstdb.value if nstdb is not None else np.nan,
        "dsm_shape_class": dsm["shape"],
        "dsm_h50_m": dsm["h50"],
        "dsm_h75_m": dsm["h75"],
        "dsm_h90_m": dsm["h90"],
        "dsm_h95_m": dsm["h95"],
        "dsm_h98_m": dsm["h98"],
    }


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def compact_output(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only the columns needed for downstream use and manual inspection."""

    keep_columns = [
        "BL_ID",
        "FCODE",
        "final_height_m",
        "ground_ref_m",
        "roof_elevation_m",
        "roof_elevation_method",
        "final_height_source",
        "final_height_confidence",
        "height_selection_flag",
        "external_dsm_conflict",
        "dsm_shape_class",
        "dsm_h50_m",
        "dsm_h75_m",
        "dsm_h90_m",
        "dsm_h95_m",
        "dsm_h98_m",
        "candidate_osm_m",
        "candidate_nstdb_m",
        "osm_match_quality",
        "nstdb_match_method",
    ]
    columns = [column for column in keep_columns if column in buildings.columns]
    return buildings[columns + ["geometry"]].copy()


def write_summary(out_path: Path, buildings: gpd.GeoDataFrame) -> None:
    """Write a compact text summary of the selected height layer."""

    selected = buildings["final_height_m"].dropna()
    roof = buildings["roof_elevation_m"].dropna() if "roof_elevation_m" in buildings.columns else pd.Series(dtype="float64")
    lines = [
        "Selected Building Heights Summary",
        f"buildings_count: {len(buildings)}",
        "",
        "final_height_source_counts:",
    ]
    for source, count in buildings["final_height_source"].value_counts().items():
        lines.append(f"  {source}: {count}")
    lines.extend(["", "height_selection_flag_counts:"])
    for flag, count in buildings["height_selection_flag"].value_counts().items():
        lines.append(f"  {flag}: {count}")
    if "external_dsm_conflict" in buildings.columns:
        lines.extend(["", "external_dsm_conflict_counts:"])
        for flag, count in buildings["external_dsm_conflict"].value_counts().items():
            lines.append(f"  {flag}: {count}")
    if "osm_match_quality" in buildings.columns:
        osm_quality = buildings.loc[buildings["osm_match_quality"].astype(str) != "", "osm_match_quality"]
        if not osm_quality.empty:
            lines.extend(["", "osm_match_quality_counts:"])
            for quality, count in osm_quality.value_counts().items():
                lines.append(f"  {quality}: {count}")
    if "nstdb_match_method" in buildings.columns:
        nstdb_method = buildings.loc[buildings["nstdb_match_method"].astype(str) != "", "nstdb_match_method"]
        if not nstdb_method.empty:
            lines.extend(["", "nstdb_match_method_counts:"])
            for method, count in nstdb_method.value_counts().items():
                lines.append(f"  {method}: {count}")
    lines.extend(
        [
            "",
            "final_height_stats_m:",
            f"  min: {selected.min():.2f}",
            f"  p05: {selected.quantile(0.05):.2f}",
            f"  median: {selected.median():.2f}",
            f"  p95: {selected.quantile(0.95):.2f}",
            f"  max: {selected.max():.2f}",
        ]
    )
    if not roof.empty:
        lines.extend(
            [
                "",
                "roof_elevation_stats_m:",
                f"  min: {roof.min():.2f}",
                f"  median: {roof.median():.2f}",
                f"  p95: {roof.quantile(0.95):.2f}",
                f"  max: {roof.max():.2f}",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Pipeline entry point
# -----------------------------------------------------------------------------


def main() -> None:
    """Run the full height-selection pipeline."""

    args = HeightSelectionConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    buildings = gpd.read_file(args.footprints)
    if args.limit is not None:
        buildings = buildings.head(args.limit).copy()
    print(f"Loaded footprints: {len(buildings):,}")

    dsm_stats = calculate_dsm_dem_evidence(buildings, args)
    enriched = buildings.copy().reset_index(drop=True)
    for column in dsm_stats.columns:
        enriched[column] = dsm_stats[column].values
    enriched = add_osm_matches(enriched, args)
    enriched = add_nstdb_matches(enriched, args)

    selected = pd.DataFrame.from_records([choose_height(row, args) for _, row in enriched.iterrows()])
    for column in selected.columns:
        enriched[column] = selected[column].values
    output = compact_output(enriched)

    out_gpkg = args.out_dir / "building_heights_selected.gpkg"
    out_summary = args.out_dir / "building_heights_selected_summary.txt"
    if out_gpkg.exists():
        out_gpkg.unlink()
    output.to_file(out_gpkg, layer="buildings", driver="GPKG")
    write_summary(out_summary, output)
    print(f"Wrote: {out_gpkg}")
    print(f"Wrote: {out_summary}")


if __name__ == "__main__":
    main()
