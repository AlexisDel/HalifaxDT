"""
Build building meshes for the Halifax Sionna scene.

The script reads the selected building materials and heights, samples the
prepared DEM at each footprint contour point, and writes one wall PLY per wall
material plus one roof PLY per roof material. Building bases follow the terrain
while roofs remain flat.

Output:
  data/scenes/halifax_peninsula/meshes/buildings_wall_<material>.ply
  data/scenes/halifax_peninsula/meshes/buildings_roof_<material>.ply
  data/scenes/halifax_peninsula/buildings_metadata.txt
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import triangulate

from build_terrain_mesh import (
    DATA_ROOT,
    DEM_PATH,
    MESH_DIR,
    SCENE_DIR,
    TARGET_CRS,
    TARGET_RESOLUTION_M,
    grid_coordinates,
    local_origin,
    read_dem_in_utm,
    write_binary_ply,
)


BUILDINGS_PATH = DATA_ROOT / "processed_data" / "building_materials_selected.gpkg"
OUTPUT_METADATA = SCENE_DIR / "buildings_metadata.txt"

EDGE_SAMPLE_SPACING_M = TARGET_RESOLUTION_M
DEFAULT_WALL_MATERIAL = "concrete"
DEFAULT_ROOF_MATERIAL = "asphalt"


def numeric(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def material_key(value: object, default: str) -> str:
    material = text(value).lower().replace(" ", "_").replace("-", "_")
    return material or default


def mesh_path(kind: str, material: str) -> Path:
    return MESH_DIR / f"buildings_{kind}_{material}.ply"


def iter_polygons(geometry: object) -> Iterable[Polygon]:
    """Yield valid polygons from a Polygon or MultiPolygon geometry."""
    if isinstance(geometry, Polygon):
        polygons = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
    else:
        polygons = []

    for polygon in polygons:
        if polygon.is_empty:
            continue
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if isinstance(polygon, MultiPolygon):
            fixed_polygons = list(polygon.geoms)
        else:
            fixed_polygons = [polygon]
        for fixed_polygon in fixed_polygons:
            if isinstance(fixed_polygon, Polygon) and not fixed_polygon.is_empty and fixed_polygon.area > 0:
                yield orient(fixed_polygon, sign=1.0)


def densify_ring(coords: Iterable[tuple[float, float]], spacing_m: float) -> list[tuple[float, float]]:
    """Add points along a ring so terrain sampling follows long footprint edges."""
    points = [(float(x), float(y)) for x, y, *_ in coords]
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        return points

    densified: list[tuple[float, float]] = []
    for i, start in enumerate(points):
        end = points[(i + 1) % len(points)]
        densified.append(start)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        segments = max(1, int(math.ceil(distance / spacing_m)))
        for step in range(1, segments):
            ratio = step / segments
            densified.append((start[0] + dx * ratio, start[1] + dy * ratio))

    return densified


def densify_polygon(polygon: Polygon, spacing_m: float) -> Polygon:
    exterior = densify_ring(polygon.exterior.coords, spacing_m)
    interiors = [densify_ring(ring.coords, spacing_m) for ring in polygon.interiors]
    return orient(Polygon(exterior, interiors), sign=1.0)


def sample_dem(dem: np.ndarray, transform: rasterio.Affine, x: float, y: float) -> float:
    """Bilinearly sample DEM elevation at a projected x/y coordinate."""
    col_corner, row_corner = (~transform) * (x, y)
    col = col_corner - 0.5
    row = row_corner - 0.5

    row0 = int(math.floor(row))
    col0 = int(math.floor(col))
    row1 = row0 + 1
    col1 = col0 + 1

    if 0 <= row0 and row1 < dem.shape[0] and 0 <= col0 and col1 < dem.shape[1]:
        values = np.array(
            [
                dem[row0, col0],
                dem[row0, col1],
                dem[row1, col0],
                dem[row1, col1],
            ],
            dtype=np.float64,
        )
        if np.isfinite(values).all():
            dc = col - col0
            dr = row - row0
            top = values[0] * (1.0 - dc) + values[1] * dc
            bottom = values[2] * (1.0 - dc) + values[3] * dc
            return float(top * (1.0 - dr) + bottom * dr)

    nearest_row = int(round(row))
    nearest_col = int(round(col))
    for radius in range(0, 4):
        min_row = max(0, nearest_row - radius)
        max_row = min(dem.shape[0], nearest_row + radius + 1)
        min_col = max(0, nearest_col - radius)
        max_col = min(dem.shape[1], nearest_col + radius + 1)
        window = dem[min_row:max_row, min_col:max_col]
        valid_rows, valid_cols = np.where(np.isfinite(window))
        if valid_rows.size:
            distances = (valid_rows + min_row - row) ** 2 + (valid_cols + min_col - col) ** 2
            nearest = int(np.argmin(distances))
            return float(window[valid_rows[nearest], valid_cols[nearest]])

    return np.nan


def roof_elevation(row: object) -> float:
    roof_z = numeric(row.get("roof_elevation_m"))
    if np.isfinite(roof_z):
        return roof_z

    ground_z = numeric(row.get("ground_ref_m"))
    height = numeric(row.get("final_height_m"))
    if np.isfinite(ground_z) and np.isfinite(height):
        return ground_z + height

    return np.nan


def triangle_area_xy(coords: list[tuple[float, float]]) -> float:
    (x0, y0), (x1, y1), (x2, y2) = coords
    return 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def add_building_polygon(
    polygon: Polygon,
    roof_z: float,
    dem: np.ndarray,
    transform: rasterio.Affine,
    x0: float,
    y0: float,
    wall_vertices: list[tuple[float, float, float]],
    wall_faces: list[tuple[int, int, int]],
    roof_vertices: list[tuple[float, float, float]],
    roof_faces: list[tuple[int, int, int]],
) -> bool:
    polygon = densify_polygon(polygon, EDGE_SAMPLE_SPACING_M)
    rings = [polygon.exterior, *polygon.interiors]
    roof_mesh_by_xy: dict[tuple[float, float], int] = {}
    sampled_rings: list[tuple[int, list[tuple[float, float]], list[float]]] = []

    def xy_key(x: float, y: float) -> tuple[float, float]:
        return round(x, 3), round(y, 3)

    def add_wall_vertex(x: float, y: float, z: float) -> int:
        wall_vertices.append((float(x - x0), float(y - y0), float(z)))
        return len(wall_vertices) - 1

    def add_roof_vertex(x: float, y: float, z: float) -> int:
        roof_vertices.append((float(x - x0), float(y - y0), float(z)))
        return len(roof_vertices) - 1

    for ring_index, ring in enumerate(rings):
        coords = [(float(x), float(y)) for x, y, *_ in ring.coords[:-1]]
        if len(coords) < 3:
            continue

        base_values: list[float] = []
        for x, y in coords:
            base_z = sample_dem(dem, transform, x, y)
            if not np.isfinite(base_z):
                return False
            base_values.append(base_z)
        sampled_rings.append((ring_index, coords, base_values))

    if not sampled_rings:
        return False

    for ring_index, coords, base_values in sampled_rings:
        base_ids: list[int] = []
        wall_roof_ids: list[int] = []

        for (x, y), base_z in zip(coords, base_values):
            base_ids.append(add_wall_vertex(x, y, base_z))
            wall_roof_ids.append(add_wall_vertex(x, y, roof_z))
            key = xy_key(x, y)
            if key not in roof_mesh_by_xy:
                roof_mesh_by_xy[key] = add_roof_vertex(x, y, roof_z)

        for i, base_a in enumerate(base_ids):
            base_b = base_ids[(i + 1) % len(base_ids)]
            roof_a = wall_roof_ids[i]
            roof_b = wall_roof_ids[(i + 1) % len(wall_roof_ids)]

            if ring_index == 0:
                wall_faces.append((base_a, base_b, roof_a))
                wall_faces.append((base_b, roof_b, roof_a))
            else:
                wall_faces.append((base_a, roof_a, base_b))
                wall_faces.append((base_b, roof_a, roof_b))

    for triangle in triangulate(polygon):
        if triangle.is_empty or not polygon.covers(triangle.representative_point()):
            continue

        coords = [(float(x), float(y)) for x, y, *_ in triangle.exterior.coords[:-1]]
        if len(coords) != 3:
            continue

        face: list[int] = []
        for x, y in coords:
            key = xy_key(x, y)
            if key not in roof_mesh_by_xy:
                roof_mesh_by_xy[key] = add_roof_vertex(x, y, roof_z)
            face.append(roof_mesh_by_xy[key])

        if triangle_area_xy(coords) < 0:
            roof_faces.append((face[0], face[2], face[1]))
        else:
            roof_faces.append((face[0], face[1], face[2]))

    return True


def build_building_mesh(
    buildings: gpd.GeoDataFrame,
    dem: np.ndarray,
    transform: rasterio.Affine,
    x0: float,
    y0: float,
) -> tuple[dict[str, dict[str, object]], dict[str, int | float | str]]:
    meshes: dict[str, dict[str, object]] = {}
    skipped_missing_height = 0
    skipped_geometry = 0
    skipped_no_terrain = 0
    polygons_written = 0

    def get_mesh(kind: str, material: str) -> dict[str, object]:
        key = f"{kind}:{material}"
        if key not in meshes:
            meshes[key] = {
                "kind": kind,
                "material": material,
                "vertices": [],
                "faces": [],
                "polygons_written": 0,
            }
        return meshes[key]

    for _, row in buildings.iterrows():
        height = numeric(row.get("final_height_m"))
        roof_z = roof_elevation(row)
        if not np.isfinite(height) or height <= 0 or not np.isfinite(roof_z):
            skipped_missing_height += 1
            continue

        wall_material = material_key(row.get("wall_material"), DEFAULT_WALL_MATERIAL)
        roof_material = material_key(row.get("roof_material"), DEFAULT_ROOF_MATERIAL)
        wall_mesh = get_mesh("wall", wall_material)
        roof_mesh = get_mesh("roof", roof_material)

        polygons = list(iter_polygons(row.geometry))
        if not polygons:
            skipped_geometry += 1
            continue

        for polygon in polygons:
            success = add_building_polygon(
                polygon,
                roof_z,
                dem,
                transform,
                x0,
                y0,
                wall_mesh["vertices"],  # type: ignore[arg-type]
                wall_mesh["faces"],  # type: ignore[arg-type]
                roof_mesh["vertices"],  # type: ignore[arg-type]
                roof_mesh["faces"],  # type: ignore[arg-type]
            )
            if success:
                polygons_written += 1
                wall_mesh["polygons_written"] = int(wall_mesh["polygons_written"]) + 1
                roof_mesh["polygons_written"] = int(roof_mesh["polygons_written"]) + 1
            else:
                skipped_no_terrain += 1

    mesh_keys_to_remove = [
        key for key, mesh in meshes.items() if not mesh["vertices"] or not mesh["faces"]
    ]
    for key in mesh_keys_to_remove:
        del meshes[key]

    if not meshes:
        raise ValueError("No building mesh vertices were generated.")

    total_vertices = 0
    total_faces = 0
    vertex_arrays: list[np.ndarray] = []
    for mesh in meshes.values():
        vertex_array = np.asarray(mesh["vertices"], dtype=np.float32)
        mesh["vertex_array"] = vertex_array
        total_vertices += int(vertex_array.shape[0])
        total_faces += len(mesh["faces"])  # type: ignore[arg-type]
        vertex_arrays.append(vertex_array)

    all_vertices = np.vstack(vertex_arrays)
    metadata = {
        "buildings_read": int(len(buildings)),
        "polygons_written": int(polygons_written),
        "skipped_missing_height": int(skipped_missing_height),
        "skipped_geometry": int(skipped_geometry),
        "skipped_no_terrain": int(skipped_no_terrain),
        "mesh_parts": int(len(meshes)),
        "vertices": int(total_vertices),
        "faces": int(total_faces),
        "local_origin_x_utm_m": float(x0),
        "local_origin_y_utm_m": float(y0),
        "local_min_x_m": float(all_vertices[:, 0].min()),
        "local_max_x_m": float(all_vertices[:, 0].max()),
        "local_min_y_m": float(all_vertices[:, 1].min()),
        "local_max_y_m": float(all_vertices[:, 1].max()),
        "min_z_m": float(all_vertices[:, 2].min()),
        "max_z_m": float(all_vertices[:, 2].max()),
    }
    return meshes, metadata


def write_building_meshes(meshes: dict[str, dict[str, object]]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, mesh in sorted(meshes.items()):
        kind = str(mesh["kind"])
        material = str(mesh["material"])
        path = mesh_path(kind, material)
        write_binary_ply(
            path,
            mesh["vertex_array"],  # type: ignore[arg-type]
            mesh["faces"],  # type: ignore[arg-type]
        )
        paths[key] = path
    return paths


def write_metadata(
    path: Path,
    metadata: dict[str, int | float | str],
    meshes: dict[str, dict[str, object]],
    mesh_paths: dict[str, Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Halifax building mesh metadata"]
    for key, value in metadata.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.3f}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("mesh_parts_detail:")
    for key, mesh in sorted(meshes.items()):
        vertex_array = mesh["vertex_array"]
        faces = mesh["faces"]
        lines.append(
            "  "
            f"{key}: "
            f"file={mesh_paths[key].relative_to(SCENE_DIR).as_posix()}, "
            f"vertices={len(vertex_array)}, "
            f"faces={len(faces)}, "
            f"polygons_written={mesh['polygons_written']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(f"Reading DEM: {DEM_PATH}")
    dem, transform, crs, _ = read_dem_in_utm(DEM_PATH)
    height, width = dem.shape
    xs, ys = grid_coordinates(transform, width, height)
    valid = np.isfinite(dem)
    x0, y0 = local_origin(xs, ys, valid)

    print(f"Reading buildings: {BUILDINGS_PATH}")
    buildings = gpd.read_file(BUILDINGS_PATH, layer="buildings")
    if buildings.crs is None:
        raise ValueError(f"Buildings have no CRS: {BUILDINGS_PATH}")
    buildings = buildings.to_crs(TARGET_CRS)

    print(f"Building mesh CRS: {crs}")
    print(f"Local origin: x0={x0:.1f}, y0={y0:.1f}")

    meshes, metadata = build_building_mesh(buildings, dem, transform, x0, y0)
    print(f"Mesh parts: {metadata['mesh_parts']:,}")
    print(f"Vertices: {metadata['vertices']:,}")
    print(f"Faces: {metadata['faces']:,}")
    print(f"Polygons written: {metadata['polygons_written']:,}")
    print(
        "Local extent: "
        f"x={metadata['local_min_x_m']:.1f}..{metadata['local_max_x_m']:.1f} m, "
        f"y={metadata['local_min_y_m']:.1f}..{metadata['local_max_y_m']:.1f} m, "
        f"z={metadata['min_z_m']:.1f}..{metadata['max_z_m']:.1f} m"
    )

    mesh_paths = write_building_meshes(meshes)
    write_metadata(OUTPUT_METADATA, metadata, meshes, mesh_paths)
    for key, path in sorted(mesh_paths.items()):
        print(f"Wrote {key}: {path}")
    print(f"Wrote metadata: {OUTPUT_METADATA}")
    print("Run scripts/sionna/build_scene_xml.py to assemble the Sionna XML scene.")


if __name__ == "__main__":
    main()
