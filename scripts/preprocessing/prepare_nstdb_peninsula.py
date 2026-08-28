"""
Filter NSTDB building layers to the Halifax peninsula buildings.

The NSTDB source contains many features far outside the study area. For height
estimation we only need NSTDB features close to the HRM peninsula footprints.
This script:
  - loads HRM peninsula footprints
  - reprojects them to the NSTDB CRS
  - buffers footprints to tolerate small NSTDB geometry offsets
  - keeps NSTDB points/polygons intersecting those buffered footprints
  - drops invalid ZVALUE=9999 records by default

Output:
  data/interim_data/nstdb_peninsula/nstdb_peninsula.gpkg
    layers: nstdb_points, nstdb_polygons, footprint_match_area
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


INVALID_ZVALUE = 9999.0
BUFFER_M = 25.0
KEEP_MISSING_Z = False


def data_root() -> Path:
    return Path(__file__).parents[2] / "data"


ROOT = data_root()
FOOTPRINTS_PATH = ROOT / "interim_data" / "footprints_peninsula_hrm" / "Building_Polygons_Peninsula.shp"
NSTDB_POINTS_PATH = ROOT / "raw_data" / "nstdb_shapefile" / "BL_POINT_10K.shp"
NSTDB_POLYGONS_PATH = ROOT / "raw_data" / "nstdb_shapefile" / "BL_POLY_10K.shp"
OUTPUT_DIR = ROOT / "interim_data" / "nstdb_peninsula"


def valid_zvalue(gdf: gpd.GeoDataFrame) -> pd.Series:
    # NSTDB uses 9999 as a placeholder for unknown/non-usable heights.
    z = pd.to_numeric(gdf["ZVALUE"], errors="coerce")
    return z.notna() & (z != INVALID_ZVALUE)


def filter_to_match_area(
    source_path: Path,
    match_area: gpd.GeoDataFrame,
    keep_missing_z: bool,
    layer_name: str,
) -> gpd.GeoDataFrame:
    print(f"Loading {layer_name}: {source_path}")
    # The bbox read avoids loading the full province-scale NSTDB layer.
    source = gpd.read_file(source_path, bbox=tuple(match_area.total_bounds))
    print(f"  Loaded by bbox: {len(source):,}")

    if source.crs != match_area.crs:
        source = source.to_crs(match_area.crs)

    if not keep_missing_z:
        before = len(source)
        source = source[valid_zvalue(source)].copy()
        print(f"  Kept valid ZVALUE: {len(source):,} / {before:,}")

    if source.empty:
        return source

    source = source.reset_index(drop=True)
    # Keep only features that intersect the buffered HRM footprint area. The
    # buffer compensates for small NSTDB positional offsets.
    matched_idx = gpd.sjoin(
        source[["FEAT_CODE", "FEAT_DESC", "ZVALUE", "geometry"]],
        match_area[["geometry"]],
        how="inner",
        predicate="intersects",
    ).index.unique()

    filtered = source.loc[matched_idx].copy()
    filtered = filtered.reset_index(drop=True)
    print(f"  Intersects buffered footprints: {len(filtered):,}")
    return filtered


def write_summary(
    out_path: Path,
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    buffer_m: float,
    keep_missing_z: bool,
) -> None:
    lines = [
        "NSTDB Peninsula Filter Summary",
        f"buffer_m: {buffer_m}",
        f"keep_missing_z: {keep_missing_z}",
        "",
        f"points_count: {len(points)}",
        f"polygons_count: {len(polygons)}",
        "",
        "points_by_feat_code:",
    ]
    if not points.empty:
        for code, count in points["FEAT_CODE"].value_counts().items():
            lines.append(f"  {code}: {count}")
    lines.append("")
    lines.append("polygons_by_feat_code:")
    if not polygons.empty:
        for code, count in polygons["FEAT_CODE"].value_counts().items():
            lines.append(f"  {code}: {count}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_gpkg = OUTPUT_DIR / "nstdb_peninsula.gpkg"
    out_summary = OUTPUT_DIR / "nstdb_peninsula_summary.txt"

    print(f"Loading HRM footprints: {FOOTPRINTS_PATH}")
    footprints = gpd.read_file(FOOTPRINTS_PATH)
    print(f"  Footprints: {len(footprints):,}")

    nstdb_probe = gpd.read_file(NSTDB_POINTS_PATH, rows=1)
    nstdb_crs = nstdb_probe.crs
    if nstdb_crs is None:
        raise ValueError(f"NSTDB points have no CRS: {NSTDB_POINTS_PATH}")

    footprints_nstdb = footprints.to_crs(nstdb_crs)
    match_area = footprints_nstdb[["geometry"]].copy()
    # Buffer is used only for matching. It never changes the final HRM
    # footprint geometry used for 3D extrusion.
    match_area["geometry"] = match_area.geometry.buffer(BUFFER_M)
    match_area = match_area.dissolve()
    match_area = match_area.explode(index_parts=False).reset_index(drop=True)

    print("Match area")
    print(f"  CRS: {match_area.crs}")
    print(f"  Buffer: {BUFFER_M} m")
    print(f"  Bounds: {tuple(round(v, 3) for v in match_area.total_bounds)}")

    points = filter_to_match_area(
        NSTDB_POINTS_PATH,
        match_area,
        KEEP_MISSING_Z,
        "NSTDB points",
    )
    polygons = filter_to_match_area(
        NSTDB_POLYGONS_PATH,
        match_area,
        KEEP_MISSING_Z,
        "NSTDB polygons",
    )

    if out_gpkg.exists():
        out_gpkg.unlink()
    match_area.to_file(out_gpkg, layer="footprint_match_area", driver="GPKG")
    points.to_file(out_gpkg, layer="nstdb_points", driver="GPKG")
    polygons.to_file(out_gpkg, layer="nstdb_polygons", driver="GPKG")
    write_summary(out_summary, points, polygons, BUFFER_M, KEEP_MISSING_Z)

    print(f"Wrote: {out_gpkg}")
    print(f"Wrote: {out_summary}")


if __name__ == "__main__":
    main()


