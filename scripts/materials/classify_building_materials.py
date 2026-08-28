"""
Assign likely radio-material classes to Halifax building footprints.

This script keeps the height-selection output untouched and writes a new
GeoPackage with material columns. The material choice is a transparent scoring
model: every source adds direct points to one or more likely materials, then the
material with the highest score wins.

Output:
  data/processed_data/building_materials_selected.gpkg
  data/processed_data/building_materials_summary.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

DEFAULT_BUILDINGS_PATH = DATA_ROOT / "processed_data" / "building_heights_selected.gpkg"
DEFAULT_OUTPUT_PATH = DATA_ROOT / "processed_data" / "building_materials_selected.gpkg"
DEFAULT_SUMMARY_PATH = DATA_ROOT / "processed_data" / "building_materials_summary.txt"
DEFAULT_OSM_PATHS = [
    DATA_ROOT / "interim_data" / "osm_buildings_height_levels.geojson",
]

TARGET_CRS = "EPSG:32620"
WGS84_CRS = "EPSG:4326"

MATERIALS = ("concrete", "brick", "wood", "metal", "glass")
ROOF_MATERIALS = ("concrete", "asphalt", "metal", "wood", "glass")


@dataclass
class Evidence:
    material: str
    points: float
    reason: str
    source: str


@dataclass
class MaterialDecision:
    wall_material: str
    roof_material: str
    confidence: str
    confidence_score: float
    source: str
    reason: str
    wall_scores: dict[str, float] = field(default_factory=dict)
    roof_scores: dict[str, float] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", type=Path, default=DEFAULT_BUILDINGS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--osm",
        type=Path,
        nargs="*",
        default=DEFAULT_OSM_PATHS,
        help="OSM GeoJSON/GPKG files used for explicit material tags and building type hints.",
    )
    parser.add_argument("--min-osm-overlap", type=float, default=0.15)
    parser.add_argument("--disable-zone-priors", action="store_true")
    return parser.parse_args()


def text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def numeric(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_json(mapping: dict[str, float]) -> str:
    return json.dumps({key: round(float(value), 4) for key, value in mapping.items()}, sort_keys=True)


def normalize_material(value: object) -> str | None:
    raw = text(value).lower().replace("_", " ").replace("-", " ")
    if not raw:
        return None

    if any(token in raw for token in ("reinforced concrete", "concrete", "cement", "cinder block")):
        return "concrete"
    if any(token in raw for token in ("brick", "masonry", "stone", "granite", "limestone", "sandstone")):
        return "brick"
    if any(token in raw for token in ("wood", "timber", "log")):
        return "wood"
    if any(token in raw for token in ("metal", "steel", "aluminium", "aluminum", "tin", "corrugated")):
        return "metal"
    if "glass" in raw:
        return "glass"

    return None


def normalize_roof_material(value: object) -> str | None:
    raw = text(value).lower().replace("_", " ").replace("-", " ")
    if not raw:
        return None

    if any(token in raw for token in ("concrete", "cement")):
        return "concrete"
    if any(token in raw for token in ("asphalt", "shingle", "tar", "bitumen", "tile", "slate")):
        return "asphalt"
    if any(token in raw for token in ("metal", "steel", "aluminium", "aluminum", "tin", "copper")):
        return "metal"
    if any(token in raw for token in ("wood", "timber")):
        return "wood"
    if "glass" in raw:
        return "glass"

    return None


def parse_year(value: object) -> float:
    raw = text(value)
    if not raw:
        return np.nan
    match = re.search(r"(18|19|20)\d{2}", raw)
    if not match:
        return np.nan
    year = float(match.group(0))
    if 1800 <= year <= 2030:
        return year
    return np.nan


def add_evidence(
    evidence: list[Evidence],
    material: str,
    points: float,
    reason: str,
    source: str,
) -> None:
    if material not in MATERIALS:
        return
    evidence.append(Evidence(material, points, reason, source))


def score_evidence(evidence: Iterable[Evidence]) -> dict[str, float]:
    scores = {material: 0.0 for material in MATERIALS}
    for item in evidence:
        scores[item.material] += item.points
    return scores


def classify_confidence(top_score: float, second_score: float, source: str) -> tuple[str, float]:
    if source == "osm_explicit_material":
        return "high", 0.92

    gap = max(0.0, top_score - second_score)
    dominance = gap / max(top_score, 1e-9)
    strength = min(1.0, top_score / 4.0)
    confidence_score = float(np.clip(0.25 + 0.35 * dominance + 0.40 * strength, 0.20, 0.86))

    if confidence_score >= 0.70:
        return "medium_high", confidence_score
    if confidence_score >= 0.55:
        return "medium", confidence_score
    if confidence_score >= 0.40:
        return "low_medium", confidence_score
    return "low", confidence_score


def polygon_shape_metrics(geometry: object) -> tuple[float, float, float]:
    """Return area, compactness, elongation for a projected geometry."""
    if geometry is None or geometry.is_empty:
        return np.nan, np.nan, np.nan

    area = float(geometry.area)
    perimeter = float(geometry.length)
    compactness = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else np.nan

    elongation = np.nan
    try:
        rect = geometry.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
        lengths = [
            math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
            for i in range(4)
        ]
        long_side = max(lengths)
        short_side = min(length for length in lengths if length > 0)
        elongation = long_side / short_side if short_side > 0 else np.nan
    except Exception:
        pass

    return area, compactness, elongation


def load_osm_evidence(paths: list[Path]) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        frame = gpd.read_file(path)
        if frame.empty or "geometry" not in frame:
            continue
        if frame.crs is None:
            frame = frame.set_crs(WGS84_CRS)
        keep = [
            column
            for column in frame.columns
            if column == "geometry"
            or column in {
                "@id",
                "building",
                "building:material",
                "facade:material",
                "material",
                "roof:material",
                "roof:shape",
                "building:year_built",
                "year_of_construction",
                "start_date",
            }
        ]
        frames.append(frame[keep].copy())

    if not frames:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=WGS84_CRS)

    osm = pd.concat(frames, ignore_index=True)
    if "@id" in osm.columns:
        osm = osm.drop_duplicates("@id", keep="first")
    osm = gpd.GeoDataFrame(osm, geometry="geometry", crs=frames[0].crs).to_crs(TARGET_CRS)
    osm = osm[osm.geometry.notna() & ~osm.geometry.is_empty].copy()
    return osm


def first_available(row: pd.Series, columns: Iterable[str]) -> object:
    for column in columns:
        if column in row and text(row.get(column)):
            return row.get(column)
    return ""


def attach_osm_matches(
    buildings: gpd.GeoDataFrame,
    osm: gpd.GeoDataFrame,
    min_overlap: float,
) -> gpd.GeoDataFrame:
    out = buildings.copy()
    for column in [
        "osm_material",
        "osm_roof_material",
        "osm_building_type",
        "osm_roof_shape",
        "osm_year_built",
        "osm_overlap_ratio",
    ]:
        out[column] = np.nan if column in {"osm_year_built", "osm_overlap_ratio"} else ""

    if osm.empty:
        return out

    buildings_for_join = out[["geometry"]].copy()
    buildings_for_join["building_index"] = out.index
    candidates = gpd.sjoin(buildings_for_join, osm, how="inner", predicate="intersects")
    if candidates.empty:
        return out

    candidates["hrm_area_m2"] = out.loc[candidates["building_index"], "geometry"].area.to_numpy()
    osm_geometries = osm.geometry
    overlaps: list[float] = []
    for _, row in candidates.iterrows():
        hrm_geom = out.at[row["building_index"], "geometry"]
        osm_geom = osm_geometries.loc[row["index_right"]]
        overlaps.append(float(hrm_geom.intersection(osm_geom).area))
    candidates["overlap_m2"] = overlaps
    candidates["osm_overlap_ratio"] = candidates["overlap_m2"] / candidates["hrm_area_m2"].replace(0, np.nan)
    candidates = candidates[candidates["osm_overlap_ratio"] >= min_overlap].copy()
    if candidates.empty:
        return out

    candidates = candidates.sort_values(["building_index", "osm_overlap_ratio"], ascending=[True, False])
    best = candidates.drop_duplicates("building_index", keep="first")

    for _, row in best.iterrows():
        idx = row["building_index"]
        out.at[idx, "osm_material"] = normalize_material(
            first_available(row, ["building:material", "facade:material", "material"])
        ) or ""
        out.at[idx, "osm_roof_material"] = normalize_roof_material(row.get("roof:material")) or ""
        out.at[idx, "osm_building_type"] = text(row.get("building"))
        out.at[idx, "osm_roof_shape"] = text(row.get("roof:shape"))
        out.at[idx, "osm_year_built"] = parse_year(
            first_available(row, ["building:year_built", "year_of_construction", "start_date"])
        )
        out.at[idx, "osm_overlap_ratio"] = float(row["osm_overlap_ratio"])

    return out


def classify_zone(lon: float, lat: float) -> str:
    """Approximate peninsula zones used only as low-confidence priors."""
    if not np.isfinite(lon) or not np.isfinite(lat):
        return "unknown"

    if -63.610 <= lon <= -63.575 and 44.628 <= lat <= 44.643:
        return "institutional_south_end"
    if -63.590 <= lon <= -63.560 and 44.640 <= lat <= 44.655:
        return "downtown_core"
    if lon >= -63.570 and 44.620 <= lat <= 44.660:
        return "waterfront_port"
    if lat < 44.640:
        return "south_end_residential"
    if lat >= 44.655:
        return "north_end_mixed"
    return "central_residential_mixed"


def add_building_type_evidence(evidence: list[Evidence], building_type: str) -> None:
    value = building_type.lower()
    if not value:
        return

    if any(token in value for token in ("industrial", "warehouse", "hangar", "shed", "storage")):
        add_evidence(evidence, "metal", 2.6, f"OSM building={building_type}", "osm_building_type")
        add_evidence(evidence, "concrete", 0.45, f"OSM building={building_type}", "osm_building_type")
    elif any(token in value for token in ("house", "detached", "semidetached", "residential", "terrace", "bungalow")):
        add_evidence(evidence, "wood", 1.8, f"OSM building={building_type}", "osm_building_type")
        add_evidence(evidence, "brick", 0.45, f"OSM building={building_type}", "osm_building_type")
    elif any(token in value for token in ("apartments", "hotel", "office", "commercial", "retail")):
        add_evidence(evidence, "concrete", 1.5, f"OSM building={building_type}", "osm_building_type")
        add_evidence(evidence, "brick", 0.54, f"OSM building={building_type}", "osm_building_type")
    elif any(token in value for token in ("school", "university", "hospital", "public", "civic")):
        add_evidence(evidence, "concrete", 1.95, f"OSM building={building_type}", "osm_building_type")
        add_evidence(evidence, "brick", 0.45, f"OSM building={building_type}", "osm_building_type")
    elif value not in {"yes", "building"}:
        add_evidence(evidence, "concrete", 0.28, f"OSM building={building_type}", "osm_building_type")


def add_fcode_evidence(evidence: list[Evidence], fcode: str) -> None:
    value = fcode.lower()
    if not value:
        return

    if any(token in value for token in ("ind", "warehouse", "storage", "shed", "garage")):
        add_evidence(evidence, "metal", 1, f"FCODE={fcode}", "fcode")
    elif any(token in value for token in ("res", "house", "dwelling")):
        add_evidence(evidence, "wood", 0.72, f"FCODE={fcode}", "fcode")
        add_evidence(evidence, "brick", 0.245, f"FCODE={fcode}", "fcode")
    elif any(token in value for token in ("school", "hospital", "institution", "public", "comm")):
        add_evidence(evidence, "concrete", 0.765, f"FCODE={fcode}", "fcode")


def add_geometry_evidence(
    evidence: list[Evidence],
    height_m: float,
    area_m2: float,
    compactness: float,
    elongation: float,
) -> None:
    if np.isfinite(height_m):
        if height_m >= 30.0:
            add_evidence(evidence, "concrete", 2.88, f"height {height_m:.1f} m", "height")
            add_evidence(evidence, "glass", 0.4, f"height {height_m:.1f} m", "height")
        elif height_m >= 18.0:
            add_evidence(evidence, "concrete", 2.04, f"height {height_m:.1f} m", "height")
        elif height_m >= 10.0:
            add_evidence(evidence, "concrete", 0.54, f"height {height_m:.1f} m", "height")
            add_evidence(evidence, "brick", 0.42, f"height {height_m:.1f} m", "height")
        else:
            add_evidence(evidence, "wood", 0.42, f"height {height_m:.1f} m", "height")
            add_evidence(evidence, "brick", 0.21, f"height {height_m:.1f} m", "height")

    if np.isfinite(area_m2):
        if np.isfinite(height_m) and area_m2 >= 1800.0 and height_m <= 14.0:
            add_evidence(evidence, "metal", 2.17, f"large low footprint {area_m2:.0f} m2", "footprint_area")
            add_evidence(evidence, "concrete", 0.4, f"large low footprint {area_m2:.0f} m2", "footprint_area")
        elif np.isfinite(height_m) and area_m2 >= 1000.0 and height_m <= 12.0:
            add_evidence(evidence, "metal", 1.21, f"large low footprint {area_m2:.0f} m2", "footprint_area")
        elif np.isfinite(height_m) and area_m2 <= 300.0 and height_m <= 10.0:
            add_evidence(evidence, "wood", 1.45, f"small low footprint {area_m2:.0f} m2", "footprint_area")
        elif np.isfinite(height_m) and area_m2 <= 140.0 and height_m <= 8.0:
            add_evidence(evidence, "wood", 1.8, f"very small low footprint {area_m2:.0f} m2", "footprint_area")

    if (
        np.isfinite(area_m2)
        and np.isfinite(height_m)
        and np.isfinite(compactness)
        and np.isfinite(elongation)
        and area_m2 >= 900.0
        and height_m <= 12.0
        and compactness <= 0.55
        and elongation >= 2.0
    ):
        add_evidence(evidence, "metal", 0.936, "large elongated low footprint", "footprint_shape")


def add_zone_evidence(evidence: list[Evidence], zone: str) -> None:
    if zone == "downtown_core":
        add_evidence(evidence, "brick", 0.84, "downtown core prior", "zone")
        add_evidence(evidence, "concrete", 0.48, "downtown core prior", "zone")
    elif zone == "waterfront_port":
        add_evidence(evidence, "metal", 0.9, "waterfront/port prior", "zone")
        add_evidence(evidence, "concrete", 0.504, "waterfront/port prior", "zone")
    elif zone == "institutional_south_end":
        add_evidence(evidence, "concrete", 0.96, "institutional south end prior", "zone")
        add_evidence(evidence, "brick", 0.28, "institutional south end prior", "zone")
    elif zone == "south_end_residential":
        add_evidence(evidence, "wood", 0.756, "south end residential prior", "zone")
        add_evidence(evidence, "brick", 0.315, "south end residential prior", "zone")
    elif zone == "north_end_mixed":
        add_evidence(evidence, "wood", 0.456, "north end mixed prior", "zone")
        add_evidence(evidence, "brick", 0.36, "north end mixed prior", "zone")
    elif zone == "central_residential_mixed":
        add_evidence(evidence, "wood", 0.35, "central residential/mixed prior", "zone")
        add_evidence(evidence, "brick", 0.256, "central residential/mixed prior", "zone")


def add_age_evidence(evidence: list[Evidence], year: float, zone: str) -> None:
    if not np.isfinite(year):
        return

    if year < 1940:
        add_evidence(evidence, "brick", 0.644, f"older OSM year {year:.0f}", "age")
        if zone in {"south_end_residential", "north_end_mixed", "central_residential_mixed"}:
            add_evidence(evidence, "wood", 0.42, f"older residential OSM year {year:.0f}", "age")
    elif year >= 1980:
        add_evidence(evidence, "concrete", 0.42, f"newer OSM year {year:.0f}", "age")
        add_evidence(evidence, "glass", 0.15, f"newer OSM year {year:.0f}", "age")


def choose_roof_material(row: pd.Series, wall_material: str, area_m2: float, height_m: float) -> tuple[str, dict[str, float]]:
    scores = {material: 0.0 for material in ROOF_MATERIALS}
    osm_roof = text(row.get("osm_roof_material"))
    roof_shape = text(row.get("osm_roof_shape")).lower()

    if osm_roof in scores:
        scores[osm_roof] += 9.0

    if wall_material == "metal":
        scores["metal"] += 3.0
    elif wall_material == "wood":
        scores["asphalt"] += 2.5
        scores["wood"] += 0.5
    elif wall_material == "brick":
        scores["asphalt"] += 1.8
        scores["concrete"] += 0.8
    elif wall_material in {"concrete", "glass"}:
        scores["concrete"] += 2.2
        scores["asphalt"] += 0.7

    if np.isfinite(area_m2) and np.isfinite(height_m) and area_m2 >= 1200.0 and height_m <= 12.0:
        scores["metal"] += 1.5
    if "flat" in roof_shape:
        scores["concrete"] += 1.2
    elif any(token in roof_shape for token in ("gabled", "hipped", "pitched")):
        scores["asphalt"] += 1.2

    selected = max(scores.items(), key=lambda item: item[1])[0]
    return selected, scores


def decide_material(row: pd.Series, use_zone_priors: bool) -> MaterialDecision:
    evidence: list[Evidence] = []
    height_m = numeric(row.get("final_height_m"))
    area_m2 = numeric(row.get("footprint_area_m2"))
    compactness = numeric(row.get("footprint_compactness"))
    elongation = numeric(row.get("footprint_elongation"))
    zone = text(row.get("material_zone")) or "unknown"

    add_evidence(evidence, "concrete", 0.125, "baseline fallback", "default")

    osm_material = text(row.get("osm_material"))
    if osm_material in MATERIALS:
        scores = {material: 0.0 for material in MATERIALS}
        scores[osm_material] = 9.5
        roof_material, roof_scores = choose_roof_material(row, osm_material, area_m2, height_m)
        return MaterialDecision(
            wall_material=osm_material,
            roof_material=roof_material,
            confidence="high",
            confidence_score=0.92,
            source="osm_explicit_material",
            reason=f"OSM explicit material={osm_material}",
            wall_scores=scores,
            roof_scores=roof_scores,
        )

    add_building_type_evidence(evidence, text(row.get("osm_building_type")))
    add_fcode_evidence(evidence, text(row.get("FCODE")))
    add_geometry_evidence(evidence, height_m, area_m2, compactness, elongation)
    if use_zone_priors:
        add_zone_evidence(evidence, zone)
    add_age_evidence(evidence, numeric(row.get("osm_year_built")), zone)

    scores = score_evidence(evidence)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    wall_material = ordered[0][0]
    top_score = ordered[0][1]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0

    source_priority = [
        "osm_explicit_material",
        "osm_building_type",
        "fcode",
        "height",
        "footprint_area",
        "footprint_shape",
        "zone",
        "age",
        "default",
    ]
    winning_reasons = [item for item in evidence if item.material == wall_material]
    sources = {item.source for item in winning_reasons}
    source = next((candidate for candidate in source_priority if candidate in sources), "heuristic")

    confidence, confidence_score = classify_confidence(top_score, second_score, source)
    roof_material, roof_scores = choose_roof_material(row, wall_material, area_m2, height_m)

    reason = "; ".join(item.reason for item in winning_reasons if item.source != "default")
    if not reason:
        reason = "baseline fallback"

    return MaterialDecision(
        wall_material=wall_material,
        roof_material=roof_material,
        confidence=confidence,
        confidence_score=confidence_score,
        source=source,
        reason=reason,
        wall_scores=scores,
        roof_scores=roof_scores,
    )


def enrich_materials(
    buildings: gpd.GeoDataFrame,
    osm_paths: list[Path],
    min_osm_overlap: float,
    use_zone_priors: bool,
) -> gpd.GeoDataFrame:
    buildings = buildings.to_crs(TARGET_CRS).copy()

    metrics = buildings.geometry.apply(polygon_shape_metrics)
    buildings["footprint_area_m2"] = [item[0] for item in metrics]
    buildings["footprint_compactness"] = [item[1] for item in metrics]
    buildings["footprint_elongation"] = [item[2] for item in metrics]

    centroids_wgs84 = buildings.geometry.centroid.to_crs(WGS84_CRS)
    buildings["material_zone"] = [
        classify_zone(point.x, point.y) if use_zone_priors else "disabled"
        for point in centroids_wgs84
    ]

    osm = load_osm_evidence(osm_paths)
    buildings = attach_osm_matches(buildings, osm, min_osm_overlap)

    decisions = [decide_material(row, use_zone_priors) for _, row in buildings.iterrows()]
    buildings["wall_material"] = [decision.wall_material for decision in decisions]
    buildings["roof_material"] = [decision.roof_material for decision in decisions]
    buildings["material_source"] = [decision.source for decision in decisions]
    buildings["material_confidence"] = [decision.confidence for decision in decisions]
    buildings["material_confidence_score"] = [decision.confidence_score for decision in decisions]
    buildings["material_reason"] = [decision.reason for decision in decisions]

    for material in MATERIALS:
        buildings[f"score_{material}"] = [decision.wall_scores[material] for decision in decisions]

    buildings["wall_material_scores"] = [safe_json(decision.wall_scores) for decision in decisions]
    buildings["roof_material_scores"] = [safe_json(decision.roof_scores) for decision in decisions]
    return buildings


def write_summary(path: Path, buildings: gpd.GeoDataFrame, osm_paths: list[Path], use_zone_priors: bool) -> None:
    lines = ["Halifax building material classification summary"]
    lines.append(f"buildings: {len(buildings)}")
    lines.append(f"zone_priors_enabled: {use_zone_priors}")
    lines.append("osm_inputs:")
    for osm_path in osm_paths:
        lines.append(f"  {osm_path}: {'found' if osm_path.exists() else 'missing'}")

    lines.append("")
    lines.append("wall_material_counts:")
    for key, value in buildings["wall_material"].value_counts(dropna=False).sort_index().items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("roof_material_counts:")
    for key, value in buildings["roof_material"].value_counts(dropna=False).sort_index().items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("material_source_counts:")
    for key, value in buildings["material_source"].value_counts(dropna=False).sort_index().items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("material_confidence_counts:")
    for key, value in buildings["material_confidence"].value_counts(dropna=False).sort_index().items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("material_zone_counts:")
    for key, value in buildings["material_zone"].value_counts(dropna=False).sort_index().items():
        lines.append(f"  {key}: {value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    print(f"Reading buildings: {args.buildings}")
    buildings = gpd.read_file(args.buildings, layer="buildings")
    if buildings.crs is None:
        raise ValueError(f"Buildings have no CRS: {args.buildings}")

    use_zone_priors = not args.disable_zone_priors
    enriched = enrich_materials(
        buildings,
        osm_paths=args.osm,
        min_osm_overlap=args.min_osm_overlap,
        use_zone_priors=use_zone_priors,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing material GeoPackage: {args.output}")
    enriched.to_file(args.output, layer="buildings", driver="GPKG")

    print(f"Writing summary: {args.summary}")
    write_summary(args.summary, enriched, args.osm, use_zone_priors)

    print("Wall material counts:")
    for key, value in enriched["wall_material"].value_counts().sort_index().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
