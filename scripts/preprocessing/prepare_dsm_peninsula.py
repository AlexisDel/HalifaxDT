"""
Extract the 1 m DSM raster for the Halifax peninsula.

The source DSM is a very large BigTIFF. This script never loads it entirely:
it reads only the raster window that covers the peninsula building footprints,
optionally plus a buffer.

Outputs:
  data/interim_data/dsm_peninsula_1m/HRM_LiDAR_DSM_2018_1m_peninsula.tif

Optional:
  set BUILDINGS_ONLY = True to also write a version masked outside HRM footprints.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.windows import from_bounds, transform as window_transform


def project_root() -> Path:
    # The script lives in scripts/preprocessing, so parents[2] is the project root.
    return Path(__file__).parents[2] / "data"


ROOT = project_root()
DSM_PATH = ROOT / "raw_data" / "HRM_LiDAR_DSM_2018_1m_wgs84" / "HRM_LiDAR_DSM_2018_1m_wgs84.tif"
FOOTPRINTS_PATH = ROOT / "interim_data" / "footprints_peninsula_hrm" / "Building_Polygons_Peninsula.shp"
OUTPUT_DIR = ROOT / "interim_data" / "dsm_peninsula_1m"
BUFFER_M = 50.0
BUILDINGS_ONLY = False


def crop_dsm(
    dsm_path: Path,
    footprints_path: Path,
    out_dir: Path,
    buffer_m: float,
    buildings_only: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_crop = out_dir / "HRM_LiDAR_DSM_2018_1m_peninsula.tif"
    out_masked = out_dir / "HRM_LiDAR_DSM_2018_1m_peninsula_buildings_only.tif"

    footprints = gpd.read_file(footprints_path)
    print(f"Loaded {len(footprints):,} footprint polygons")

    with rasterio.open(dsm_path) as src:
        if src.crs is None:
            raise ValueError(f"DSM has no CRS: {dsm_path}")

        # Reproject footprints to the DSM CRS before computing raster bounds.
        # The DSM is EPSG:3857, not geographic longitude/latitude.
        footprints_dsm = footprints.to_crs(src.crs)
        minx, miny, maxx, maxy = footprints_dsm.total_bounds
        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m

        window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        window = window.round_offsets().round_lengths()
        crop_transform = window_transform(window, src.transform)

        print("DSM source")
        print(f"  CRS: {src.crs}")
        print(f"  Size: {src.width:,} x {src.height:,}")
        print(f"  Resolution: {src.res}")
        print(f"  Nodata: {src.nodata}")
        print("Crop window")
        print(f"  Col offset: {window.col_off:,}")
        print(f"  Row offset: {window.row_off:,}")
        print(f"  Width: {window.width:,}")
        print(f"  Height: {window.height:,}")

        # Read only the requested BigTIFF window. This avoids loading the
        # original ~136 GB DSM into memory.
        data = src.read(1, window=window)
        profile = src.profile.copy()
        profile.update(
            {
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": crop_transform,
                "driver": "GTiff",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "compress": "deflate",
                "predictor": 3,
                "BIGTIFF": "IF_SAFER",
            }
        )

        with rasterio.open(out_crop, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"Wrote DSM crop: {out_crop}")

        valid = data
        if src.nodata is not None:
            valid = data[data != src.nodata]
        valid = valid[np.isfinite(valid)]
        if valid.size:
            print(
                "Crop values: "
                f"min={valid.min():.2f} "
                f"p50={np.percentile(valid, 50):.2f} "
                f"p95={np.percentile(valid, 95):.2f} "
                f"max={valid.max():.2f}"
            )

        if buildings_only:
            # Optional QA/output mode: keep DSM values only where HRM building
            # footprints exist and write nodata everywhere else.
            footprint_mask = features.rasterize(
                ((geom, 1) for geom in footprints_dsm.geometry if not geom.is_empty),
                out_shape=data.shape,
                transform=crop_transform,
                fill=0,
                dtype="uint8",
            )
            masked = data.copy()
            nodata = src.nodata
            if nodata is None:
                nodata = np.float32(np.nan)
                profile["nodata"] = nodata
            masked[footprint_mask == 0] = nodata
            with rasterio.open(out_masked, "w", **profile) as dst:
                dst.write(masked, 1)
            print(f"Wrote building-masked DSM: {out_masked}")


def main() -> None:
    crop_dsm(
        dsm_path=DSM_PATH,
        footprints_path=FOOTPRINTS_PATH,
        out_dir=OUTPUT_DIR,
        buffer_m=BUFFER_M,
        buildings_only=BUILDINGS_ONLY,
    )


if __name__ == "__main__":
    main()


