from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
OUTPUT_GEOJSON = DATA_ROOT / "interim_data" / "osm_buildings_height_levels.geojson"

# Overpass expects polygon coordinates as "lat lon lat lon ...".
PENINSULA_POLYGON_LAT_LON = [
    (44.6410835, -63.6194204),
    (44.6643012, -63.6308541),
    (44.6817374, -63.6220173),
    (44.6736948, -63.5992474),
    (44.6413834, -63.5531831),
    (44.6169733, -63.5565099),
    (44.6133803, -63.5641958),
    (44.6253175, -63.5844221),
    (44.6410626, -63.6194298),
    (44.6410835, -63.6194204),
]


def build_overpass_query() -> str:
    polygon = " ".join(
        f"{lat:.7f} {lon:.7f}" for lat, lon in PENINSULA_POLYGON_LAT_LON
    )
    return f"""[out:json][timeout:180];

(
  way["building"](poly:"{polygon}");
  relation["building"](poly:"{polygon}");
);

out body geom;
"""


def fetch_overpass_json(query: str) -> dict[str, Any]:
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(
        OVERPASS_ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": "halifax-twin-osm-downloader",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Overpass HTTP error {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Overpass API: {exc}") from exc


def coords_from_overpass_geometry(points: list[dict[str, float]]) -> list[list[float]]:
    return [[float(point["lon"]), float(point["lat"])] for point in points]


def close_ring(coords: list[list[float]]) -> list[list[float]]:
    if coords and coords[0] != coords[-1]:
        coords = [*coords, coords[0]]
    return coords


def assemble_rings(segments: list[list[list[float]]]) -> list[list[list[float]]]:
    remaining = [segment for segment in segments if len(segment) >= 2]
    rings: list[list[list[float]]] = []

    while remaining:
        ring = remaining.pop(0)
        changed = True
        while changed and ring[0] != ring[-1]:
            changed = False
            for index, segment in enumerate(remaining):
                if ring[-1] == segment[0]:
                    ring.extend(segment[1:])
                elif ring[-1] == segment[-1]:
                    ring.extend(reversed(segment[:-1]))
                elif ring[0] == segment[-1]:
                    ring = [*segment[:-1], *ring]
                elif ring[0] == segment[0]:
                    ring = [*reversed(segment[1:]), *ring]
                else:
                    continue

                remaining.pop(index)
                changed = True
                break

        ring = close_ring(ring)
        if len(ring) >= 4:
            rings.append(ring)

    return rings


def way_geometry(element: dict[str, Any]) -> dict[str, Any] | None:
    coords = coords_from_overpass_geometry(element.get("geometry", []))
    if len(coords) < 2:
        return None
    if len(coords) >= 4 and coords[0] == coords[-1]:
        return {"type": "Polygon", "coordinates": [coords]}
    return {"type": "LineString", "coordinates": coords}


def relation_geometry(element: dict[str, Any]) -> dict[str, Any] | None:
    outer_segments = []
    for member in element.get("members", []):
        if member.get("type") != "way":
            continue
        if member.get("role") not in ("", "outer"):
            continue
        geometry = member.get("geometry")
        if not geometry:
            continue
        outer_segments.append(coords_from_overpass_geometry(geometry))

    rings = assemble_rings(outer_segments)
    if not rings:
        return None
    if len(rings) == 1:
        return {"type": "Polygon", "coordinates": [rings[0]]}
    return {
        "type": "MultiPolygon",
        "coordinates": [[ring] for ring in rings],
    }


def element_to_feature(element: dict[str, Any]) -> dict[str, Any] | None:
    element_type = element.get("type")
    if element_type == "way":
        geometry = way_geometry(element)
    elif element_type == "relation":
        geometry = relation_geometry(element)
    else:
        geometry = None

    if geometry is None:
        return None

    properties = dict(element.get("tags", {}))
    properties["@id"] = f"{element_type}/{element['id']}"
    properties["@type"] = element_type

    return {
        "type": "Feature",
        "properties": properties,
        "geometry": geometry,
    }


def overpass_to_geojson(overpass_data: dict[str, Any]) -> dict[str, Any]:
    features = []
    skipped = 0
    for element in overpass_data.get("elements", []):
        feature = element_to_feature(element)
        if feature is None:
            skipped += 1
            continue
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "metadata": {
            "source": "OpenStreetMap via Overpass API",
            "query": build_overpass_query(),
            "features": len(features),
            "skipped_elements_without_geometry": skipped,
        },
        "features": features,
    }


def main() -> None:
    print("Downloading OSM buildings from Overpass...")
    overpass_data = fetch_overpass_json(build_overpass_query())
    geojson = overpass_to_geojson(overpass_data)

    OUTPUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_GEOJSON, "w", encoding="utf-8") as file:
        json.dump(geojson, file, ensure_ascii=False)

    print(f"Saved {len(geojson['features']):,} buildings to:")
    print(OUTPUT_GEOJSON)


if __name__ == "__main__":
    main()
