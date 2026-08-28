"""
Build a first terrain mesh for the Halifax Sionna scene.

The script reads the prepared 5 m DEM, reprojects it to UTM zone 20N
(EPSG:32620), converts x/y to a local meter-based scene frame, and writes a
PLY mesh that can later be referenced by a Mitsuba/Sionna scene XML.

Output:
  data/scenes/halifax_peninsula/meshes/terrain.ply
  data/scenes/halifax_peninsula/terrain_metadata.txt
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


DATA_ROOT = Path(__file__).parents[2] / "data"

DEM_PATH = DATA_ROOT / "interim_data" / "dem_peninsula_5m" / "HRM_LiDAR_DEM_2018_5m_peninsula.tif"
SCENE_DIR = DATA_ROOT / "scenes" / "halifax_peninsula"
MESH_DIR = SCENE_DIR / "meshes"
OUTPUT_PLY = MESH_DIR / "terrain.ply"
OUTPUT_METADATA = SCENE_DIR / "terrain_metadata.txt"

TARGET_CRS = "EPSG:32620"  # WGS84 / UTM zone 20N, suitable for Halifax.
TARGET_RESOLUTION_M = 5.0
LOCAL_ORIGIN_ROUNDING_M = 100.0


def read_dem_in_utm(dem_path: Path) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS, float | None]:
    """Read the DEM and reproject it to a regular UTM grid."""
    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError(f"DEM has no CRS: {dem_path}")

        transform, width, height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
            resolution=TARGET_RESOLUTION_M,
        )

        dem = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dem,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        return dem, transform, rasterio.crs.CRS.from_string(TARGET_CRS), src.nodata


def grid_coordinates(transform: rasterio.Affine, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return center coordinates for each raster column and row."""
    cols = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)

    # These formulas are the raster transform applied to pixel centers.
    xs = transform.c + (cols + 0.5) * transform.a + 0.5 * transform.b
    ys = transform.f + 0.5 * transform.d + (rows + 0.5) * transform.e
    return xs, ys


def local_origin(xs: np.ndarray, ys: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """Choose a stable local origin near the south-west corner of valid DEM data."""
    valid_rows, valid_cols = np.where(valid)
    if valid_rows.size == 0:
        raise ValueError("DEM has no valid terrain pixels after reprojection.")

    min_x = float(xs[valid_cols].min())
    min_y = float(ys[valid_rows].min())

    x0 = math.floor(min_x / LOCAL_ORIGIN_ROUNDING_M) * LOCAL_ORIGIN_ROUNDING_M
    y0 = math.floor(min_y / LOCAL_ORIGIN_ROUNDING_M) * LOCAL_ORIGIN_ROUNDING_M
    return x0, y0


def build_vertices_and_faces(
    dem: np.ndarray,
    transform: rasterio.Affine,
) -> tuple[np.ndarray, list[tuple[int, int, int]], dict[str, float | int]]:
    """Convert a DEM grid to vertices and upward-facing triangle faces."""
    height, width = dem.shape
    xs, ys = grid_coordinates(transform, width, height)

    valid = np.isfinite(dem)
    x0, y0 = local_origin(xs, ys, valid)

    vertex_index = np.full((height, width), -1, dtype=np.int32)
    valid_rows, valid_cols = np.where(valid)
    vertex_index[valid_rows, valid_cols] = np.arange(valid_rows.size, dtype=np.int32)

    vertices = np.column_stack(
        (
            xs[valid_cols] - x0,
            ys[valid_rows] - y0,
            dem[valid_rows, valid_cols],
        )
    ).astype(np.float32)

    faces: list[tuple[int, int, int]] = []
    for row in range(height - 1):
        for col in range(width - 1):
            p00 = int(vertex_index[row, col])
            p01 = int(vertex_index[row, col + 1])
            p10 = int(vertex_index[row + 1, col])
            p11 = int(vertex_index[row + 1, col + 1])

            if min(p00, p01, p10, p11) < 0:
                continue

            # Raster rows go north to south, so this winding keeps normals up.
            faces.append((p00, p10, p01))
            faces.append((p01, p10, p11))

    metadata = {
        "target_crs_epsg": 32620,
        "target_resolution_m": TARGET_RESOLUTION_M,
        "local_origin_x_utm_m": x0,
        "local_origin_y_utm_m": y0,
        "vertices": int(vertices.shape[0]),
        "faces": int(len(faces)),
        "local_min_x_m": float(vertices[:, 0].min()),
        "local_max_x_m": float(vertices[:, 0].max()),
        "local_min_y_m": float(vertices[:, 1].min()),
        "local_max_y_m": float(vertices[:, 1].max()),
        "min_z_m": float(vertices[:, 2].min()),
        "max_z_m": float(vertices[:, 2].max()),
    }
    return vertices, faces, metadata


def write_binary_ply(path: Path, vertices: np.ndarray, faces: list[tuple[int, int, int]]) -> None:
    """Write a binary little-endian triangle PLY without extra dependencies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Halifax terrain mesh generated from DEM\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertices.astype("<f4", copy=False).tofile(f)
        for face in faces:
            f.write(struct.pack("<Biii", 3, *face))


def write_metadata(path: Path, metadata: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Halifax terrain mesh metadata"]
    for key, value in metadata.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.3f}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(f"Reading DEM: {DEM_PATH}")
    dem, transform, crs, _ = read_dem_in_utm(DEM_PATH)
    print(f"Reprojected DEM CRS: {crs}")
    print(f"Reprojected DEM grid: {dem.shape[1]:,} x {dem.shape[0]:,}")
    print(f"Resolution: {TARGET_RESOLUTION_M:.1f} m")

    vertices, faces, metadata = build_vertices_and_faces(dem, transform)
    print(f"Vertices: {len(vertices):,}")
    print(f"Faces: {len(faces):,}")
    print(
        "Local extent: "
        f"x={metadata['local_min_x_m']:.1f}..{metadata['local_max_x_m']:.1f} m, "
        f"y={metadata['local_min_y_m']:.1f}..{metadata['local_max_y_m']:.1f} m, "
        f"z={metadata['min_z_m']:.1f}..{metadata['max_z_m']:.1f} m"
    )

    write_binary_ply(OUTPUT_PLY, vertices, faces)
    write_metadata(OUTPUT_METADATA, metadata)
    print(f"Wrote terrain mesh: {OUTPUT_PLY}")
    print(f"Wrote metadata: {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
