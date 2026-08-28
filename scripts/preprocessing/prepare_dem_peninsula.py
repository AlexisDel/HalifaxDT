"""
Extract the 5 m DEM raster for the Halifax peninsula.

By default this script uses the already cropped DSM bounds, so the DEM and DSM
cover the same geographic area. The DEM keeps its native 5 m resolution.

Output:
  data/interim_data/dem_peninsula_5m/HRM_LiDAR_DEM_2018_5m_peninsula.tif
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds, transform as window_transform


def data_root() -> Path:
    # The script lives in scripts/preprocessing, so parents[2] is the project root.
    return Path(__file__).parents[2] / "data"


ROOT = data_root()
DEM_PATH = ROOT / "raw_data" / "HRM_LiDAR_DEM_2018_5m_wgs84" / "HRM_LiDAR_DEM_2018_5m_wgs84.tif"
REFERENCE_DSM_PATH = ROOT / "interim_data" / "dsm_peninsula_1m" / "HRM_LiDAR_DSM_2018_1m_peninsula.tif"
OUTPUT_DIR = ROOT / "interim_data" / "dem_peninsula_5m"


def crop_dem(dem_path: Path, reference_dsm_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_crop = out_dir / "HRM_LiDAR_DEM_2018_5m_peninsula.tif"

    with rasterio.open(reference_dsm_path) as ref, rasterio.open(dem_path) as src:
        if src.crs != ref.crs:
            raise ValueError(
                f"DEM CRS ({src.crs}) does not match reference DSM CRS ({ref.crs})."
            )

        # Use the DSM crop bounds so both rasters cover the same peninsula area.
        # The DEM keeps its native 5 m resolution.
        rb = ref.bounds
        window = from_bounds(rb.left, rb.bottom, rb.right, rb.top, src.transform)
        window = window.round_offsets().round_lengths()
        crop_transform = window_transform(window, src.transform)

        print("DEM source")
        print(f"  CRS: {src.crs}")
        print(f"  Size: {src.width:,} x {src.height:,}")
        print(f"  Resolution: {src.res}")
        print(f"  Nodata: {src.nodata}")
        print("Reference DSM crop")
        print(f"  Bounds: {ref.bounds}")
        print("DEM crop window")
        print(f"  Col offset: {window.col_off:,}")
        print(f"  Row offset: {window.row_off:,}")
        print(f"  Width: {window.width:,}")
        print(f"  Height: {window.height:,}")

        # Read the matching DEM window only; the full DEM is much larger than
        # the peninsula study area.
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
        print(f"Wrote DEM crop: {out_crop}")

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


def main() -> None:
    crop_dem(
        dem_path=DEM_PATH,
        reference_dsm_path=REFERENCE_DSM_PATH,
        out_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()


